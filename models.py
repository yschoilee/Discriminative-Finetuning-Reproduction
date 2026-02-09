import os

import clip  # openai/CLIP
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class DiscriTuneTrainer(nn.Module):
    """
    Trainer (REINFORCE):
      - caption generation
      - reward
      - running-mean baseline
      - REINFORCE loss
    """

    def __init__(self, captioner_model, retriever_model, moving_average_decay=0.9):
        super().__init__()
        self.captioner = captioner_model
        self.retriever = retriever_model

        for param in self.retriever.parameters():
            param.requires_grad = False

        self.register_buffer("running_mean_reward", torch.tensor(0.0))
        self.decay = moving_average_decay

    def train(self, mode: bool = True):
        super().train(mode)

        # retriever & captioner's CLIP stay frozen/eval
        if self.retriever is not None:
            self.retriever.eval()

        clip_model = getattr(self.captioner, "clip_model", None)
        if clip_model is not None:
            clip_model.eval()

        return self

    def forward(self, images, global_step=None):
        """
        images: PIL.Image 또는 List[PIL.Image]
        """
        # 1) caption generation (greedy policy)
        generated_captions, token_log_probs = (
            self.captioner.generate_captions_with_log_probs(images)
        )

        # 2) reward
        with torch.no_grad():
            if not isinstance(images, (list, tuple)):
                images_list = [images]
            else:
                images_list = list(images)

            text_tokens = clip.tokenize(
                generated_captions, context_length=77, truncate=True
            ).to(self.retriever.device)
            _, logits_per_text = self.retriever.compute_logits(images_list, text_tokens)
            log_prob_rewards = F.log_softmax(logits_per_text, dim=1)  # (B, B)
            current_rewards = torch.diag(log_prob_rewards)  # (B,)

        # 3) baseline update
        if self.training:
            batch_mean_reward = current_rewards.mean()
            with torch.no_grad():
                self.running_mean_reward.mul_(self.decay).add_(
                    batch_mean_reward * (1.0 - self.decay)
                )

        advantage = (current_rewards - self.running_mean_reward).detach()

        # 4) REINFORCE loss
        sentence_log_probs = token_log_probs.sum(dim=1)  # (B,)
        loss = -(advantage * sentence_log_probs).mean()

        return loss, current_rewards.mean()


# Neural Retriever - OpenAI CLIP
class CLIPRetriever(nn.Module):
    """
    OpenAI CLIP (원 구현) 기반 Retriever.
    입력: PIL.Image 또는 List[PIL.Image]
    """

    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE

        model, preprocess = clip.load(config.OPENAI_CLIP, device=self.device, jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.preprocess = preprocess

    def train(self, mode: bool = True):
        # Retriever는 항상 eval 모드로 유지한다.
        super().train(False)
        if self.model is not None:
            self.model.eval()
        return self

    @torch.no_grad()
    def encode_image(self, images_pil):
        if not isinstance(images_pil, (list, tuple)):
            images_pil = [images_pil]
        image_tensor = torch.stack(
            [self.preprocess(im) for im in images_pil], dim=0
        ).to(self.device)
        return self.model.encode_image(image_tensor)

    @torch.no_grad()
    def encode_text(self, captions):
        if isinstance(captions, str):
            captions = [captions]
        text_tokens = clip.tokenize(captions, context_length=77, truncate=True).to(
            self.device
        )
        return self.model.encode_text(text_tokens)

    @torch.no_grad()
    def compute_logits(
        self,
        images,
        texts,
        context_length: int = 77,
        truncate: bool = True,
    ):
        # images -> tensor
        if torch.is_tensor(images):
            image_tensor = images.to(self.device)
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
            image_tensor = torch.stack(
                [self.preprocess(im) for im in images], dim=0
            ).to(self.device)

        # texts -> tokens
        if torch.is_tensor(texts):
            text_tokens = texts.to(self.device)
        else:
            if isinstance(texts, str):
                texts = [texts]
            text_tokens = clip.tokenize(
                texts, context_length=context_length, truncate=truncate
            ).to(self.device)

        return self.model(image_tensor, text_tokens)


# Neural Captioner - ClipCap (OpenAI CLIP + trainable mapper + trainable GPT-2)


class ClipCapMapper(nn.Module):
    # coco,conceptual weights 체크포인트 호환을 위해 hidden_dim=3840을 기본값으로 사용
    def __init__(self, input_dim=512, output_dim=768, prefix_len=10, hidden_dim=3840):
        super().__init__()
        self.prefix_len = prefix_len
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, prefix_len * output_dim),
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, self.prefix_len, self.output_dim)


class ClipCapPolicy(nn.Module):
    """
    ClipCap 형태:
      - OpenAI CLIP image encoder (frozen) + preprocess (PIL -> tensor)
      - Mapper (trainable)
      - GPT-2 LM (trainable)
    입력: PIL.Image 또는 List[PIL.Image]
    """

    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.max_len = config.MAX_TOKEN_LEN

        # OpenAI CLIP (frozen) + preprocess
        clip_model, preprocess = clip.load(
            config.OPENAI_CLIP, device=self.device, jit=False
        )
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_model = clip_model
        self.preprocess = preprocess

        # GPT-2 (trainable)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.GPT2_MODEL_NAME).to(
            self.device
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.GPT2_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Mapper (trainable)
        self.mapper = ClipCapMapper(
            input_dim=512,
            output_dim=768,
            prefix_len=10,
            hidden_dim=3840,
        ).to(self.device)

        # ClipCap 체크포인트에서 GPT-2와 mapper를 로드
        ckpt_path = getattr(config, "WEIGHTS_PATH", None)
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_clipcap_checkpoint(ckpt_path)
        elif ckpt_path:
            raise FileNotFoundError(f"WEIGHTS_PATH not found: {ckpt_path}")

        # Stop ids: '.' / ' .'
        self.eos_id = self.tokenizer.eos_token_id
        self.stop_ids = set()
        if self.eos_id is not None:
            self.stop_ids.add(self.eos_id)
        for s in [".", " ."]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                self.stop_ids.add(ids[0])

    # keep frozen CLIP encoder in eval mode, but allow mapper/gpt2 to train

    def train(self, mode: bool = True):
        super().train(mode)
        if self.clip_model is not None:
            self.clip_model.eval()
        return self

    def load_clipcap_checkpoint(
        self, ckpt_path: str, strict_gpt: bool = False, strict_mapper: bool = True
    ):
        sd = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(sd, dict):
            raise ValueError(f"Unexpected checkpoint type: {type(sd)}")

        # 1) GPT
        gpt_sd = {k[len("gpt.") :]: v for k, v in sd.items() if k.startswith("gpt.")}
        if len(gpt_sd) == 0:
            raise RuntimeError(
                "No 'gpt.' keys found in checkpoint (unexpected format)."
            )
        missing_gpt, unexpected_gpt = self.gpt2.load_state_dict(
            gpt_sd, strict=strict_gpt
        )

        # 2) Mapper
        mapper_sd = {
            k[len("clip_project.") :]: v
            for k, v in sd.items()
            if k.startswith("clip_project.")
        }
        if len(mapper_sd) == 0:
            raise RuntimeError(
                "No 'clip_project.' keys found in checkpoint (unexpected format)."
            )
        missing_map, unexpected_map = self.mapper.load_state_dict(
            mapper_sd, strict=strict_mapper
        )

        print(f"[ClipCap init] loaded: {ckpt_path}")
        print(
            f"  GPT   keys={len(gpt_sd)} | missing={len(missing_gpt)} | unexpected={len(unexpected_gpt)}"
        )
        print(
            f"  MAP   keys={len(mapper_sd)} | missing={len(missing_map)} | unexpected={len(unexpected_map)}"
        )

        if len(missing_map) != 0 or len(unexpected_map) != 0:
            raise RuntimeError(
                f"Mapper load mismatch.\nmissing={missing_map}\nunexpected={unexpected_map}"
            )

    def generate_captions_with_log_probs(self, images_pil):
        if not isinstance(images_pil, (list, tuple)):
            images_pil = [images_pil]
        batch_size = len(images_pil)
        device = self.device

        # 1) OpenAI CLIP image features (frozen)
        with torch.no_grad():
            image_tensor = torch.stack(
                [self.preprocess(im) for im in images_pil], dim=0
            ).to(device)
            image_feats = self.clip_model.encode_image(image_tensor)  # (B, 512)

        # CLIP이 FP16이어도 Mapper dtype에 맞춤
        mapper_dtype = next(self.mapper.parameters()).dtype
        image_feats = image_feats.to(dtype=mapper_dtype)

        # 2) Prefix mapping (trainable)
        prefix_embeds = self.mapper(image_feats)  # (B, prefix_len, 768)

        token_ids_list = []
        token_log_probs_list = []
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 3) t=0: prefix로 LM을 한 번 돌려 past를 만든다
        out = self.gpt2(inputs_embeds=prefix_embeds, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  # (B, vocab)

        if past_key_values is None:
            raise RuntimeError(
                "past_key_values is None. Check transformers version / use_cache behavior."
            )

        for t in range(self.max_len):
            # greedy token 선택
            action = torch.argmax(logits, dim=-1)  # (B,)

            # 선택 토큰의 log-prob
            log_probs_all = F.log_softmax(logits, dim=-1)
            chosen_log_prob = log_probs_all.gather(1, action.unsqueeze(1)).squeeze(
                1
            )  # (B,)
            chosen_log_prob = chosen_log_prob.float()

            # 종료된 샘플은 이후 신호 누적 방지
            chosen_log_prob = chosen_log_prob * alive.float()

            token_ids_list.append(action.unsqueeze(1))
            token_log_probs_list.append(chosen_log_prob)

            # stop 조건 갱신
            finished = torch.zeros_like(alive)
            for sid in self.stop_ids:
                finished |= action == sid
            alive = alive & (~finished)

            if not alive.any():
                break

            # 다음 입력 토큰: 종료된 애들은 eos로 밀어 넣어 shape 안정화
            next_input_ids = action.unsqueeze(1)
            if self.eos_id is not None:
                next_input_ids = (
                    torch.where(  # alive인 sample은 next_input_ids, 끝나면 EOS
                        alive.unsqueeze(1),
                        next_input_ids,
                        torch.full_like(next_input_ids, self.eos_id),  # EOS
                    )
                )

            # 4) t>=1: past_key_values를 써서 prefix 조건을 유지
            out = self.gpt2(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

        # pad to (B, max_len)
        T = len(token_ids_list)
        if T == 0:
            # 이론상 거의 없지만 안전 처리
            pad_id = self.eos_id if self.eos_id is not None else 0
            all_token_ids = torch.full(
                (batch_size, self.max_len), pad_id, device=device, dtype=torch.long
            )
            token_log_probs = torch.zeros(
                (batch_size, self.max_len), device=device, dtype=torch.float
            )
            generated_captions = self.tokenizer.batch_decode(
                all_token_ids, skip_special_tokens=True
            )
            return generated_captions, token_log_probs

        all_token_ids = torch.cat(token_ids_list, dim=1)  # (B, T)
        token_log_probs = torch.stack(token_log_probs_list, dim=1)  # (B, T)

        if T < self.max_len:
            pad_len = self.max_len - T
            pad_id = self.eos_id if self.eos_id is not None else 0

            all_token_ids = torch.cat(
                [
                    all_token_ids,
                    torch.full(
                        (batch_size, pad_len),
                        pad_id,
                        device=device,
                        dtype=all_token_ids.dtype,
                    ),
                ],
                dim=1,
            )
            token_log_probs = torch.cat(
                [
                    token_log_probs,
                    torch.zeros(
                        (batch_size, pad_len),
                        device=device,
                        dtype=token_log_probs.dtype,
                    ),
                ],
                dim=1,
            )

        generated_captions = self.tokenizer.batch_decode(
            all_token_ids, skip_special_tokens=True
        )
        return generated_captions, token_log_probs

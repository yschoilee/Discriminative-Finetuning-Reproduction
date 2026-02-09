import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import clip  # openai/CLIP

# Beam search
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tqdm import tqdm

from retrieval import (
    build_image_embedding_cache,
    get_image_feat_from_cache,
    seed_from_key,
)


# -----------------------------
# Beam search for ClipCap (test)
# Official implementation ported from rmokady/CLIP_prefix_caption (predict.py).
# -----------------------------
@torch.no_grad()
def clipcap_beam_search_single(
    captioner,
    image_pil: Image.Image,
    beam_size: int = 5,
    max_len: int = 40,
    temperature: float = 1.0,
    stop_ids: Optional[Set[int]] = None,  # REQUIRED
) -> str:
    """
    based on ClipCap official beam search (rmokady/CLIP_prefix_caption)
    """
    if stop_ids is None or len(stop_ids) == 0:
        raise ValueError("stop_ids must be a non-empty set of token ids.")

    device = captioner.device
    gpt2 = captioner.gpt2
    tokenizer = captioner.tokenizer

    # 1) image -> prefix embeddings
    image_tensor = torch.stack([captioner.preprocess(image_pil)], dim=0).to(device)
    image_feats = captioner.clip_model.encode_image(image_tensor)  # (1, 512)
    mapper_dtype = next(captioner.mapper.parameters()).dtype
    image_feats = image_feats.to(dtype=mapper_dtype)
    embed = captioner.mapper(image_feats)  # (1, prefix_len, 768)

    # normalize/validate stop_ids
    stop_ids = {int(x) for x in stop_ids}

    # 2) beam initialization
    tokens = None
    scores = None

    seq_lengths = torch.ones(beam_size, device=device)  # 각 beam의 현재 길이
    is_stopped = torch.zeros(
        beam_size, device=device, dtype=torch.bool
    )  # stop token boolean mask

    generated = embed  # (1, prefix_len, dim)

    # 3) generate token
    for _ in range(int(max_len)):
        outputs = gpt2(inputs_embeds=generated)
        logits = outputs.logits[:, -1, :]
        logits = logits / (temperature if temperature > 0 else 1.0)  # length normalized
        logits = logits.softmax(-1).log()

        if scores is None:  # first step
            scores, next_tokens = logits.topk(beam_size, -1)  # (1, k)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens = next_tokens.permute(1, 0)  # (k, 1)
            scores = scores.squeeze(0)  # (k,)
            tokens = next_tokens  # (k, 1)
        else:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, pad_id] = 0

            scores_sum = scores[:, None] + logits  # (k, vocab)
            seq_lengths[~is_stopped] += 1

            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                beam_size, -1
            )

            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)

            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)

            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]

        next_token_ids = next_tokens.squeeze(1)  # (k,)
        next_token_embed = gpt2.transformer.wte(next_token_ids).unsqueeze(
            1
        )  # (k,1,dim)
        generated = torch.cat((generated, next_token_embed), dim=1)

        # stop check
        next_tok = next_tokens.squeeze(1)  # (k,)
        is_stop_now = torch.zeros_like(is_stopped)
        for sid in stop_ids:
            is_stop_now |= next_tok.eq(sid)
        is_stopped = is_stopped | is_stop_now

        if is_stopped.all():
            break

    # 4) finalize
    scores = scores / seq_lengths
    output_list = tokens.detach().cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts[0].strip()


@torch.no_grad()
def generate_captions_beam(
    captioner,
    images_pil: Sequence[Image.Image],
    beam_size: int = 5,
    max_len: int = 40,
    temperature: float = 1.0,
    stop_ids: Set[int] = None,
) -> List[str]:
    return [
        clipcap_beam_search_single(
            captioner=captioner,
            image_pil=im,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
            stop_ids=stop_ids,
        )
        for im in images_pil
    ]


# Retrieval evaluation: P@1@100
@torch.no_grad()
def retrieval_p_at_1_100(
    retriever,
    dataset_samples_paths: List[str],
    images_batch: List[Image.Image],
    captions_batch: List[str],
    batch_start_index: int = 0,
    target_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    unique_distractors: bool = True,
    seed: int = 42,
    image_emb_cache: Optional[Dict[str, torch.Tensor]] = None,
    dtype: torch.dtype = torch.float32,
) -> float:

    device = retriever.device
    B = len(images_batch)
    if B == 0:
        return 0.0

    # fix pool order
    dataset_samples_paths = sorted(dataset_samples_paths)

    text_tokens = clip.tokenize(captions_batch, context_length=77, truncate=True).to(
        device
    )

    base_exclude = set(exclude_paths) if exclude_paths else set()
    correct = 0

    for b in range(B):
        per_exclude = set(base_exclude)

        # Choose per-sample deterministic RNG
        if target_paths is not None:
            target_key = str(target_paths[b])
            per_exclude.add(target_key)
            rng_b = random.Random(seed_from_key(target_key, seed))
        else:
            # fallback (deterministic w.r.t. iteration order only)
            global_index = batch_start_index + b
            rng_b = random.Random(seed + global_index)

        available = [p for p in dataset_samples_paths if p not in per_exclude]
        if len(available) < 99:
            raise ValueError(
                f"Not enough available distractors: need 99, got {len(available)} "
                f"(pool={len(dataset_samples_paths)}, excluded={len(per_exclude)})"
            )

        if unique_distractors:
            distract_paths = rng_b.sample(available, 99)
        else:
            # allow duplicates (still deterministic)
            distract_paths = [
                available[rng_b.randrange(0, len(available))] for _ in range(99)
            ]

        # Safety check: target must not leak into distractors
        if target_paths is not None and str(target_paths[b]) in distract_paths:
            raise RuntimeError(
                "Target leaked into distractors (path mismatch or pool issue)."
            )

        use_cache = (image_emb_cache is not None) and (target_paths is not None)

        if use_cache:
            # Candidate paths: [target] + 99 distractors
            tgt_path = str(target_paths[b])
            cand_paths = [tgt_path] + [str(p) for p in distract_paths]

            # Text feature for this caption
            if "text_feats" not in locals():
                text_feats = retriever.model.encode_text(text_tokens)  # (B, D)
                text_feats = F.normalize(text_feats.float(), dim=-1)

            tfeat = text_feats[b]  # (D,)

            # Gather candidate image feats from cache (CPU), then move to device
            cand_feats_cpu = [
                get_image_feat_from_cache(p, retriever, image_emb_cache, dtype=dtype)
                for p in cand_paths
            ]
            cand_feats = torch.stack(cand_feats_cpu, dim=0).to(
                device=device, dtype=torch.float32
            )  # (100, D)

            scale = retriever.model.logit_scale.exp()
            scores = (scale * (cand_feats @ tfeat)).to(torch.float32)  # (100,)
            pred_idx = int(torch.argmax(scores).item())
        else:
            # Original slow path (PIL open + preprocess every time)
            cand_pils: List[Image.Image] = [images_batch[b]]
            for p in distract_paths:
                with Image.open(p) as im:
                    cand_pils.append(im.convert("RGB"))

            image_tensor = torch.stack(
                [retriever.preprocess(im) for im in cand_pils], dim=0
            ).to(device)

            if hasattr(retriever, "compute_logits"):
                _, logits_per_text = retriever.compute_logits(
                    image_tensor, text_tokens[b : b + 1]
                )
            else:
                _, logits_per_text = retriever.model(
                    image_tensor, text_tokens[b : b + 1]
                )

            scores = logits_per_text.squeeze(0)  # (100,)
            pred_idx = int(torch.argmax(scores).item())

        if pred_idx == 0:
            correct += 1

    return correct / B


@dataclass
class CiderState:
    gts: Dict[int, List[str]]
    res: Dict[int, List[str]]
    added: int = 0
    skipped_no_ref: int = 0
    collisions: int = 0


def _wrap_for_ptb(d: dict) -> dict:
    return {k: [{"caption": s} for s in v] for k, v in d.items()}


def cider_add_batch(
    state: CiderState,
    caps: List[str],
    caps_ref: Optional[List[List[str]]],
    img_ids: Optional[List[int]],
    seen: int,
    require_ref: bool = True,
) -> None:
    for i in range(len(caps)):
        # key 선택
        if img_ids is not None and img_ids[i] is not None:
            key = int(img_ids[i])
        else:
            key = int(seen + i)

        refs_i = []
        if caps_ref is not None and i < len(caps_ref) and isinstance(caps_ref[i], list):
            refs_i = caps_ref[i]
        if not isinstance(refs_i, list):
            refs_i = []
        refs_i = [r for r in refs_i if isinstance(r, str) and len(r.strip()) > 0]

        if require_ref and len(refs_i) == 0:
            state.skipped_no_ref += 1
            continue

        if key in state.gts or key in state.res:
            state.collisions += 1
            key = int(seen + i)

        state.gts[key] = refs_i
        state.res[key] = [caps[i]]
        state.added += 1


def cider_finalize(state: CiderState) -> Tuple[Optional[float], str]:
    if len(state.gts) == 0 or len(state.res) == 0:
        return (
            None,
            f"no_valid_refs (added={state.added}, skipped={state.skipped_no_ref})",
        )
    if set(state.gts.keys()) != set(state.res.keys()):
        return None, f"key_mismatch (gts={len(state.gts)}, res={len(state.res)})"

    tokenizer = PTBTokenizer()

    gts_ptb = _wrap_for_ptb(state.gts)
    res_ptb = _wrap_for_ptb(state.res)

    gts_tok = tokenizer.tokenize(gts_ptb)
    res_tok = tokenizer.tokenize(res_ptb)

    cider = Cider()
    score, _ = cider.compute_score(gts_tok, res_tok)
    msg = f"added={state.added} skipped_no_ref={state.skipped_no_ref} collisions={state.collisions}"
    return float(score), msg


def print_stop_debug(captioner):
    tok = captioner.tokenizer
    dot = tok.encode(".")
    space_dot = tok.encode(" .")
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)

    print("[Tokenizer Debug]")
    print("  encode('.')      =", dot, "len=", len(dot), "decode=", tok.decode(dot))
    print(
        "  encode(' .')     =",
        space_dot,
        "len=",
        len(space_dot),
        "decode=",
        tok.decode(space_dot),
    )
    print(
        "  eos_token_id     =",
        eos,
        "decode=",
        (tok.decode([eos]) if eos is not None else None),
    )
    print(
        "  pad_token_id     =",
        pad,
        "decode=",
        (tok.decode([pad]) if pad is not None else None),
    )
    print("  decode([0])      =", tok.decode([0]))


@torch.no_grad()
def evaluate(
    config, captioner, retriever, test_loader, test_all_paths: List[str]
) -> Dict[str, object]:
    do_p1 = bool(getattr(config, "EVAL_DO_P1", False))
    do_cider = bool(getattr(config, "EVAL_CIDER", False))

    captioner.eval()
    retriever.eval()

    print_stop_debug(captioner)

    p1_scores: List[float] = []

    cider_require_ref = bool(getattr(config, "EVAL_CIDER_REQUIRE_REF", True))
    cider_state = CiderState(gts={}, res={})

    seen = 0
    exclude_self = bool(getattr(config, "EVAL_EXCLUDE_SELF", True))
    unique_dist = bool(getattr(config, "EVAL_UNIQUE_DISTRACTORS", True))
    base_seed = int(getattr(config, "SEED", 42))

    tok = captioner.tokenizer
    stop_ids = set()
    if tok.eos_token_id is not None:
        stop_ids.add(tok.eos_token_id)
    for s in [".", " ."]:
        ids = tok.encode(s)
        if len(ids) == 1:
            stop_ids.add(ids[0])

    # build image embedding cache once to speed up P@1 (선택 가능)
    use_emb_cache = bool(getattr(config, "EVAL_USE_EMB_CACHE", True)) and do_p1
    image_emb_cache: Optional[Dict[str, torch.Tensor]] = None
    cache_dtype = torch.float32
    if use_emb_cache:
        cache_bs = int(
            getattr(
                config,
                "EVAL_CACHE_BATCH_SIZE",
                getattr(config, "EVAL_EMB_CACHE_BS", 256),
            )
        )

        cache_dtype_name = str(
            getattr(
                config,
                "EVAL_CACHE_DTYPE",
                getattr(config, "EVAL_EMB_CACHE_DTYPE", "float32"),
            )
        ).lower()

        if cache_dtype_name in {"fp16", "float16", "half"}:
            cache_dtype = torch.float16
        else:
            cache_dtype = torch.float32

        image_emb_cache = build_image_embedding_cache(
            retriever=retriever,
            all_paths=test_all_paths,
            batch_size=cache_bs,
            dtype=cache_dtype,
            verbose=True,
        )

    for images, caps_ref, img_ids, paths in tqdm(
        test_loader, desc="Eval(test)", leave=False
    ):
        # beam decoding captions
        caps = generate_captions_beam(
            captioner=captioner,
            images_pil=images,
            beam_size=int(getattr(config, "TEST_BEAM_SIZE", 5)),
            max_len=int(getattr(config, "MAX_TOKEN_LEN", 40)),
            stop_ids=stop_ids,
        )
        """
        lens = [len(c.strip()) for c in caps]
        print(
            f"[Caps] mean_chars={sum(lens) / len(lens):.1f} | empty={(sum(l == 0 for l in lens))}/{len(lens)}" 
        )
        """

        # 1. P@1@100
        if do_p1:
            p1 = retrieval_p_at_1_100(
                retriever=retriever,
                dataset_samples_paths=test_all_paths,
                images_batch=list(images),
                captions_batch=caps,
                batch_start_index=seen,
                target_paths=(list(paths) if exclude_self else None),
                unique_distractors=unique_dist,
                seed=base_seed,
                image_emb_cache=image_emb_cache,
                dtype=cache_dtype,
            )
            p1_scores.append(p1)

        # 2. CIDEr
        if do_cider:
            cider_add_batch(
                state=cider_state,
                caps=caps,
                caps_ref=caps_ref,
                img_ids=img_ids,
                seen=seen,
                require_ref=cider_require_ref,
            )

        seen += len(images)

    out: Dict[str, object] = {}

    if do_p1:
        avg_p1 = float(sum(p1_scores) / max(1, len(p1_scores)))
        print(f"[Eval:beam] Retrieval P@1@100 = {avg_p1 * 100:.2f}%")
        out["p_at_1_100"] = avg_p1

    if do_cider:
        cider_score, msg = cider_finalize(cider_state)
        out["cider"] = cider_score
        out["cider_meta"] = {
            "added": cider_state.added,
            "skipped_no_ref": cider_state.skipped_no_ref,
            "collisions": cider_state.collisions,
        }
        print(
            f"[Eval] CIDEr = {cider_score:.6f} (x100={cider_score * 100:.2f}) | {msg}"
        )

    return out

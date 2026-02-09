import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import (
    CocoKarpathyDataset,
    ConceptualCaptionsDataset,
    collate_fn_pil,
    collate_fn_pil_with_ids,
)
from evaluate import evaluate
from models import ClipCapPolicy, CLIPRetriever, DiscriTuneTrainer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_limit(samples_list, limit: int):
    """limit <= 0 or None -> no limit"""
    if limit is None:
        return
    if isinstance(limit, int) and limit > 0:
        del samples_list[limit:]


def _device_string(config: Config) -> str:
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda"
        return f"{config.DEVICE} ({name})"
    return config.DEVICE


def print_run_header(config: Config):
    print("\n" + "=" * 80)
    print("[Run Settings]")
    print(f"  RUN_MODE      : {getattr(config, 'RUN_MODE', 'N/A')}")
    print(f"  train_dataset : {getattr(config, 'EXPERIMENT_TYPE', 'N/A')}")
    print(f"  device        : {_device_string(config)}")
    print(f"  seed          : {config.SEED}")
    print(f"  train_bs      : {config.BATCH_SIZE}")
    print(f"  eval_bs       : {config.EVAL_BATCH_SIZE}")
    print(f"  epochs        : {getattr(config, 'EPOCHS', 'N/A')}")
    print(
        f"  decoding      : train={config.TRAIN_DECODING} | test={config.TEST_DECODING}(beam={config.TEST_BEAM_SIZE})"
    )
    print(f"  max_token_len : {config.MAX_TOKEN_LEN}")
    print(
        f"  eval protocol : P@1@100 (exclude_self={config.EVAL_EXCLUDE_SELF}, unique_dist={config.EVAL_UNIQUE_DISTRACTORS})"
    )
    print(
        f"  debug sizes   : train_coco={getattr(config, 'DEBUG_COCO_TRAIN_SAMPLES', None)}, "
        f"train_concap={getattr(config, 'DEBUG_CC_TRAIN_SAMPLES', None)}, "
        f"coco_eval={getattr(config, 'DEBUG_COCO_EVAL_SAMPLES', None)}, "
        f"cc_eval={getattr(config, 'DEBUG_CC_EVAL_SAMPLES', None)}"
    )
    print(
        f"  eval ckpt in/out   : {config.DISCRITUNE_EVAL_CKPT_IN} / {config.DISCRITUNE_EVAL_CKPT_OUT}"
    )
    print(
        f"  resume ckpt in/out : {config.DISCRITUNE_RESUME_CKPT_IN} / {config.DISCRITUNE_RESUME_CKPT_OUT}"
    )
    print("=" * 80 + "\n")


def train_one_epoch(
    config, trainer, dataloader, optimizer, epoch_idx: int, global_step: int
):
    trainer.train()

    epoch_loss_sum, epoch_reward_sum = 0.0, 0.0
    progress = tqdm(dataloader, desc=f"Train {epoch_idx + 1}/{config.EPOCHS}")

    for images, _caps in progress:
        optimizer.zero_grad()

        loss, reward = trainer(images, global_step=global_step)

        loss.backward()
        optimizer.step()
        global_step += 1

        loss_val = float(loss.item())
        reward_val = float(reward.item())
        epoch_loss_sum += loss_val
        epoch_reward_sum += reward_val
        progress.set_postfix({"loss": f"{loss_val:.4f}", "reward": f"{reward_val:.4f}"})

    avg_loss = epoch_loss_sum / max(1, len(dataloader))
    avg_reward = epoch_reward_sum / max(1, len(dataloader))
    print(
        f"[Train epoch {epoch_idx + 1}] avg_loss={avg_loss:.4f} avg_reward={avg_reward:.4f}"
    )
    return avg_loss, avg_reward, global_step


def _summarize_eval(tag: str, out: dict, pool_size: int, config: Config):
    p1 = out.get("p_at_1_100", None)
    if isinstance(p1, (int, float)):
        p1_str = f"{p1:.4f} ({p1 * 100:.2f}%)"
    else:
        p1_str = "N/A"

    print(
        f"[Summary] {tag} | P@1@100={p1_str} | pool={pool_size} | "
        f"beam={config.TEST_BEAM_SIZE} | max_len={config.MAX_TOKEN_LEN} | "
        f"eval_bs={getattr(config, 'EVAL_BATCH_SIZE', config.BATCH_SIZE)}"
    )

    cider = out.get("cider", None)
    if isinstance(cider, (int, float)):
        print(f"[Summary] {tag} | CIDEr={cider:.4f}")
    elif cider is None and ("cider" in out):
        print(f"[Summary] {tag} | CIDEr=None")

    meta = out.get("cider_meta", None)
    if isinstance(meta, dict):
        print(
            f"[Summary] {tag} | CIDEr_meta: "
            f"added={meta.get('added', 0)} "
            f"skipped_no_ref={meta.get('skipped_no_ref', 0)} "
            f"collisions={meta.get('collisions', 0)}"
        )


def run_eval(config, captioner, retriever, loader, all_paths, tag: str):
    print(f"\n========== {tag} ==========")
    out = evaluate(config, captioner, retriever, loader, all_paths)
    _summarize_eval(tag=tag, out=out, pool_size=len(all_paths), config=config)
    return out


def save_discritune_ckpt(path: str, captioner: ClipCapPolicy):
    """학습되는 gpt2 + mapper만 저장 (clip encoder 등은 제외)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = captioner.state_dict()
    small = {
        k: v.cpu()
        for k, v in sd.items()
        if (k.startswith("gpt2.") or k.startswith("mapper."))
    }
    torch.save(small, path)
    print(f"[Saved ckpt] {path} (keys={len(small)})")


def load_discritune_ckpt(path: str, captioner: ClipCapPolicy):
    if not os.path.exists(path):
        raise FileNotFoundError(f"DISCRITUNE_EVAL_CKPT_IN not found: {path}")
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = captioner.load_state_dict(sd, strict=False)
    print(f"[Loaded ckpt] {path} | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (head):", missing[:10])
    if unexpected:
        print("  unexpected (head):", unexpected[:10])


def save_discritune_resume_ckpt_aprime(
    path: str,
    captioner: ClipCapPolicy,
    trainer: DiscriTuneTrainer,
    epoch_abs: int = 0,
    global_step: int = 0,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = captioner.state_dict()
    small = {
        k: v.cpu()
        for k, v in sd.items()
        if (k.startswith("gpt2.") or k.startswith("mapper."))
    }

    ckpt = {
        "format": "discritune_aprime_v1",
        "captioner": small,
        "trainer": {
            "running_mean_reward": trainer.running_mean_reward.detach().cpu(),
            "decay": float(getattr(trainer, "decay", 0.0)),
        },
        "meta": {"epoch_abs": int(epoch_abs), "global_step": int(global_step)},
    }
    torch.save(ckpt, path)
    print(
        f"[Saved RESUME(A′) ckpt] {path} | epoch_abs={epoch_abs} step={global_step} | keys={len(small)}"
    )


def load_discritune_resume_ckpt_aprime(
    path: str, captioner: ClipCapPolicy, trainer: DiscriTuneTrainer
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at path={path}")

    obj = torch.load(path, map_location="cpu")
    if not (isinstance(obj, dict) and obj.get("format") == "discritune_aprime_v1"):
        raise RuntimeError("Resume ckpt is not discritune_aprime_v1 format.")

    small = obj.get("captioner", {})
    missing, unexpected = captioner.load_state_dict(small, strict=False)
    print(
        f"[Loaded RESUME(A′) ckpt] {path} | missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print("  missing (head):", missing[:10])
    if unexpected:
        print("  unexpected (head):", unexpected[:10])

    rm = obj.get("trainer", {}).get("running_mean_reward", None)
    if rm is not None:
        trainer.running_mean_reward.data.copy_(
            rm.to(trainer.running_mean_reward.device)
        )
        print(f"[Resume] running_mean_reward={trainer.running_mean_reward.item():.6f}")

    meta = obj.get("meta", {})
    epoch_abs = int(meta.get("epoch_abs", 0))
    global_step = int(meta.get("global_step", 0))
    return epoch_abs, global_step


def build_data(config: Config):
    mode = config.RUN_MODE
    train_mode = config.EXPERIMENT_TYPE

    need_coco_eval = mode in {"E1", "E3", "ALL"}
    need_cc_eval = mode in {"E2", "E4", "ALL"}
    need_coco_train = mode in {"TRAIN", "ALL"} and train_mode == "clipcap_coco"
    need_concap_train = mode in {"TRAIN", "ALL"} and train_mode == "clipcap_concap"

    coco_train_limit = getattr(config, "DEBUG_COCO_TRAIN_SAMPLES", None)
    cc_train_limit = getattr(config, "DEBUG_CC_TRAIN_SAMPLES", None)
    coco_eval_limit = getattr(config, "DEBUG_COCO_EVAL_SAMPLES", None)
    cc_eval_limit = getattr(config, "DEBUG_CC_EVAL_SAMPLES", None)

    coco_train_loader = None
    cc_train_loader = None
    coco_test_loader = None
    cc_val_loader = None

    coco_train_all_paths = None
    cc_train_all_paths = None
    coco_eval_all_paths = None
    cc_eval_all_paths = None

    # ---- COCO train ----
    if need_coco_train:
        coco_train_set = CocoKarpathyDataset(
            karpathy_json=config.COCO_KARPATHY_JSON,
            images_root=config.COCO_IMAGES_ROOT,
            split="train",
            include_restval_in_train=False,
            return_ids=False,
        )
        _apply_limit(coco_train_set.samples, coco_train_limit)
        print(
            f"[Data] COCO train samples: {len(coco_train_set)} (limit={coco_train_limit})"
        )

        coco_train_loader = DataLoader(
            coco_train_set,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn_pil,
            drop_last=True,
        )
        coco_train_all_paths = [str(s["image_path"]) for s in coco_train_set.samples]

    # ---- ConCap train ----
    if need_concap_train:
        cc_train_set = ConceptualCaptionsDataset(
            out_dir=config.CC_TRAIN_DIR,
            return_ids=False,
            use_caption_as_ref=False,
        )
        _apply_limit(cc_train_set.samples, cc_train_limit)
        print(
            f"[Data] CC train   samples: {len(cc_train_set)} (limit={cc_train_limit})"
        )

        cc_train_loader = DataLoader(
            cc_train_set,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn_pil,
            drop_last=True,
        )
        cc_train_all_paths = [str(s["image_path"]) for s in cc_train_set.samples]

    # ---- COCO test ----
    if need_coco_eval:
        coco_test_set = CocoKarpathyDataset(
            karpathy_json=config.COCO_KARPATHY_JSON,
            images_root=config.COCO_IMAGES_ROOT,
            split="test",
            include_restval_in_train=False,
            return_ids=True,
        )
        _apply_limit(coco_test_set.samples, coco_eval_limit)
        print(
            f"[Data] COCO test  samples: {len(coco_test_set)} (limit={coco_eval_limit})"
        )
        if len(coco_test_set.samples) < 100:
            raise RuntimeError(
                "COCO test eval pool must be >= 100. Increase DEBUG_COCO_EVAL_SAMPLES."
            )

        coco_test_loader = DataLoader(
            coco_test_set,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn_pil_with_ids,
            drop_last=False,
        )
        coco_eval_all_paths = [str(s["image_path"]) for s in coco_test_set.samples]

    # ---- CC val ----
    if need_cc_eval:
        use_cc_ref = bool(getattr(config, "EVAL_USE_CC_REF", False))
        cc_val_set = ConceptualCaptionsDataset(
            out_dir=config.CC_VAL_DIR,
            return_ids=True,
            use_caption_as_ref=use_cc_ref,
        )
        _apply_limit(cc_val_set.samples, cc_eval_limit)
        print(f"[Data] CC val     samples: {len(cc_val_set)} (limit={cc_eval_limit})")
        if len(cc_val_set.samples) < 100:
            raise RuntimeError(
                "CC val eval pool must be >= 100. Increase DEBUG_CC_EVAL_SAMPLES."
            )

        cc_val_loader = DataLoader(
            cc_val_set,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn_pil_with_ids,
            drop_last=False,
        )
        cc_eval_all_paths = [str(s["image_path"]) for s in cc_val_set.samples]

    return (
        coco_train_loader,
        cc_train_loader,
        coco_test_loader,
        cc_val_loader,
        coco_train_all_paths,
        coco_eval_all_paths,
        cc_train_all_paths,
        cc_eval_all_paths,
    )


def main():
    experiment_type = input("E1, E2, E3, E4, TRAIN, ALL 중 입력하세요.")
    train_dataset = input("[checkpoint dataset] coco, concap 중 하나를 입력하세요.")
    config = Config(experiment_type, train_dataset)
    if config.RUN_MODE == "TRAIN":
        resume_ans = input("DiscriTune 이어서 학습할까요? (y/n): ").strip().lower()
    else:
        eval_mode = (
            input("[eval mode] p1 / cider / both 중 입력하세요: ").strip().lower()
        )
        if eval_mode not in {"p1", "cider", "both"}:
            raise ValueError("Invalid eval mode. Choose among: p1, cider, both")
        config.EVAL_DO_P1 = eval_mode in {"p1", "both"}
        config.EVAL_CIDER = eval_mode in {"cider", "both"}
        config.EVAL_USE_CC_REF = config.EVAL_CIDER

    set_seed(config.SEED)
    print_run_header(config)

    (
        coco_train_loader,
        cc_train_loader,
        coco_test_loader,
        cc_val_loader,
        coco_train_all_paths,
        coco_eval_all_paths,
        cc_train_all_paths,
        cc_eval_all_paths,
    ) = build_data(config)

    captioner = ClipCapPolicy(config)
    retriever = CLIPRetriever(config)

    trainer = DiscriTuneTrainer(
        captioner_model=captioner,
        retriever_model=retriever,
        moving_average_decay=config.BASELINE_DECAY,
    ).to(config.DEVICE)

    if config.RUN_MODE in {"TRAIN", "ALL"}:
        print("[Train] Using in-batch reward.")

    optimizer = torch.optim.Adam(
        params=[p for p in trainer.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
    )

    epoch_abs0, global_step0 = 0, 0
    if config.RUN_MODE == "TRAIN":
        if resume_ans not in {"y", "n"}:
            raise ValueError("Invalid input. Please enter 'y' or 'n'.")

        if resume_ans == "y":
            epoch_abs0, global_step0 = load_discritune_resume_ckpt_aprime(
                config.DISCRITUNE_RESUME_CKPT_IN, captioner, trainer
            )
            print(
                f"[Resume TRAIN] from {config.DISCRITUNE_RESUME_CKPT_IN} | epoch_abs0={epoch_abs0} step0={global_step0}"
            )

    # -------- EVAL ONLY --------
    if config.RUN_MODE == "E1":
        run_eval(
            config,
            captioner,
            retriever,
            coco_test_loader,
            coco_eval_all_paths,
            tag=f"E1 (before DiscriTune): {train_dataset} -> coco test",
        )
        return

    if config.RUN_MODE == "E2":
        run_eval(
            config,
            captioner,
            retriever,
            cc_val_loader,
            cc_eval_all_paths,
            tag=f"E2 (before DiscriTune): {train_dataset} -> CC val",
        )
        return

    if config.RUN_MODE == "E3":
        load_discritune_ckpt(config.DISCRITUNE_EVAL_CKPT_IN, captioner)
        run_eval(
            config,
            captioner,
            retriever,
            coco_test_loader,
            coco_eval_all_paths,
            tag=f"E3 (after DiscriTune): {train_dataset} -> COCO test",
        )
        return

    if config.RUN_MODE == "E4":
        load_discritune_ckpt(config.DISCRITUNE_EVAL_CKPT_IN, captioner)
        run_eval(
            config,
            captioner,
            retriever,
            cc_val_loader,
            cc_eval_all_paths,
            tag=f"E4 (after DiscriTune): {train_dataset} -> CC val",
        )
        return

    # -------- TRAIN (or ALL) --------
    def _train_loop(train_loader, train_tag: str):
        nonlocal epoch_abs0, global_step0
        last_avg_loss, last_avg_reward = None, None

        for epoch in range(config.EPOCHS):
            last_avg_loss, last_avg_reward, global_step0 = train_one_epoch(
                config, trainer, train_loader, optimizer, epoch, global_step0
            )

            save_discritune_resume_ckpt_aprime(
                config.DISCRITUNE_RESUME_CKPT_OUT,
                captioner,
                trainer,
                epoch_abs=epoch_abs0 + (epoch + 1),
                global_step=global_step0,
            )

            save_discritune_ckpt(config.DISCRITUNE_EVAL_CKPT_OUT, captioner)

        if last_avg_loss is not None and last_avg_reward is not None:
            print(
                f"[Summary] TRAIN({train_tag}) | epochs={config.EPOCHS} | train_samples={len(train_loader.dataset)} | "
                f"bs={config.BATCH_SIZE} | last_avg_loss={last_avg_loss:.4f} | last_avg_reward={last_avg_reward:.4f} | "
                f"saved={config.DISCRITUNE_EVAL_CKPT_OUT}"
            )

    if config.RUN_MODE == "TRAIN":
        if config.EXPERIMENT_TYPE == "clipcap_coco":
            assert coco_train_loader is not None, "coco_train_loader is None"
            _train_loop(coco_train_loader, "coco")
            return
        if config.EXPERIMENT_TYPE == "clipcap_concap":
            assert cc_train_loader is not None, "cc_train_loader is None"
            _train_loop(cc_train_loader, "concap")
            return
        raise ValueError(f"Unknown EXPERIMENT_TYPE: {config.EXPERIMENT_TYPE}")

    if config.RUN_MODE == "ALL":
        run_eval(
            config,
            captioner,
            retriever,
            coco_test_loader,
            coco_eval_all_paths,
            tag=f"E1 (before DiscriTune): {train_dataset} -> COCO test",
        )
        run_eval(
            config,
            captioner,
            retriever,
            cc_val_loader,
            cc_eval_all_paths,
            tag=f"E2 (before DiscriTune): {train_dataset} -> CC val",
        )

        if config.EXPERIMENT_TYPE == "clipcap_coco":
            assert coco_train_loader is not None, "coco_train_loader is None"
            _train_loop(coco_train_loader, "coco")
        else:
            assert cc_train_loader is not None, "cc_train_loader is None"
            _train_loop(cc_train_loader, "concap")

        run_eval(
            config,
            captioner,
            retriever,
            coco_test_loader,
            coco_eval_all_paths,
            tag=f"E3 (after DiscriTune): {train_dataset} -> COCO test",
        )
        run_eval(
            config,
            captioner,
            retriever,
            cc_val_loader,
            cc_eval_all_paths,
            tag=f"E4 (after DiscriTune): {train_dataset} -> CC val",
        )
        return

    raise ValueError(f"Unknown RUN_MODE: {config.RUN_MODE}")


if __name__ == "__main__":
    main()

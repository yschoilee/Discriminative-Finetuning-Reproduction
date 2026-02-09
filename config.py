import json
import os

import torch


class Config:
    def __init__(self, experiment_type="ALL", train_dataset="coco"):
        # RUN MODE (E1/E2/TRAIN/E3/E4) 선택

        # "E1": (Pre) COCO -> COCO test (in-domain)
        # "E2": (Pre) COCO -> CC val     (cross-domain)
        # "TRAIN": DiscriTune finetune on COCO train, save ckpt
        # "E3": (Post) load ckpt -> COCO test
        # "E4": (Post) load ckpt -> CC val
        # "ALL": (선택) E1->E2->TRAIN->E3->E4 한 번에
        if experiment_type not in ["E1", "E2", "E3", "E4", "TRAIN", "ALL"]:
            raise ValueError()

        self.RUN_MODE = experiment_type

        if train_dataset == "coco":
            self.EXPERIMENT_TYPE = "clipcap_coco"
        elif train_dataset == "concap":
            self.EXPERIMENT_TYPE = "clipcap_concap"
        else:
            raise ValueError()

        # 결과 저장
        self.RUNS_DIR = "./runs"
        self.SAVE_RESULTS = True

        # self.TRAIN_DATASET = "coco_karpathy"
        # self.EVAL_DATASET = "coco_karpathy"

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = 42
        self.NUM_WORKERS = 2

        self.OPENAI_CLIP = "ViT-B/32"
        self.GPT2_MODEL_NAME = "gpt2"
        self.MAPPER_TYPE = "MLP"

        self.OPTIMIZER = "Adam"
        self.LEARNING_RATE = 1e-7
        self.USE_SCHEDULER = False

        self.BATCH_SIZE = 100

        self.MAX_TOKEN_LEN = 40

        self.TRAIN_DECODING = "greedy"
        self.TEST_DECODING = "beam_search"
        self.TEST_BEAM_SIZE = 5

        self.TRAIN_USE_INBATCH = True

        # Evaluation protocol
        self.EVAL_EXCLUDE_SELF = True
        self.EVAL_UNIQUE_DISTRACTORS = True
        self.EVAL_BATCH_SIZE = 64

        self.EVAL_DO_P1 = True
        self.EVAL_CIDER = True
        self.EVAL_CIDER_REQUIRE_REF = True
        self.EVAL_USE_CC_REF = True

        # DiscriTune
        self.REWARD_TYPE = "log_prob"
        self.USE_BASELINE = True
        self.BASELINE_DECAY = 0.9

        self.EVAL_CACHE_BATCH_SIZE = 256
        self.EVAL_CACHE_DTYPE = "float32"  # or "float16"

        # 샘플 제한
        self.DEBUG_COCO_TRAIN_SAMPLES = None
        self.DEBUG_CC_TRAIN_SAMPLES = None
        self.DEBUG_COCO_EVAL_SAMPLES = None
        self.DEBUG_CC_EVAL_SAMPLES = None

        self._set_experiment_specifics()

        # COCO paths
        self.COCO_KARPATHY_JSON = "/content/data/coco_images/dataset_coco.json"
        self.COCO_IMAGES_ROOT = "/content/data/coco_images"

        # Conceptual Captions paths
        self.CC_TRAIN_DIR = "/content/data/conceptual_captions/train"
        self.CC_VAL_DIR = "/content/data/conceptual_captions/val"

        self._validate_paths()
        self._validate_run_mode()

    def _set_experiment_specifics(self):
        if "clipcap_coco" in self.EXPERIMENT_TYPE:
            self.EPOCHS = 20
            self.MODEL_ARCH = "clipcap"
            self.WEIGHTS_PATH = "./weights/coco_weights.pt"
            self.DISCRITUNE_EVAL_CKPT_OUT = (
                "./weights/discritune_coco8.pt"  # TRAIN에서 저장
            )
            self.DISCRITUNE_RESUME_CKPT_OUT = "./weights/discritune_coco_resume3.pt"
            self.DISCRITUNE_RESUME_CKPT_IN = "./weights/discritune_coco_resume3.pt"
            self.DISCRITUNE_EVAL_CKPT_IN = (
                "./weights/discritune_coco8.pt"  # E3/E4에서 로드
            )
        elif "clipcap_concap" in self.EXPERIMENT_TYPE:
            self.EPOCHS = 2
            self.MODEL_ARCH = "clipcap"
            self.WEIGHTS_PATH = "./weights/conceptual_weights.pt"
            self.DISCRITUNE_EVAL_CKPT_OUT = (
                "./weights/discritune_concap4.pt"  # TRAIN에서 저장
            )
            self.DISCRITUNE_RESUME_CKPT_OUT = "./weights/discritune_concap_resume1.pt"
            self.DISCRITUNE_RESUME_CKPT_IN = "./weights/discritune_concap_resume1.pt"
            self.DISCRITUNE_EVAL_CKPT_IN = (
                "./weights/discritune_concap4.pt"  # E3/E4에서 로드
            )

    def _validate_run_mode(self):
        valid = {"E1", "E2", "TRAIN", "E3", "E4", "ALL"}
        if self.RUN_MODE not in valid:
            raise ValueError(
                f"Invalid RUN_MODE={self.RUN_MODE}. Choose from {sorted(valid)}"
            )

        # E3/E4는 ckpt가 필요함.
        if self.RUN_MODE in {"E3", "E4"}:
            if not self.DISCRITUNE_EVAL_CKPT_IN or (
                not os.path.exists(self.DISCRITUNE_EVAL_CKPT_IN)
            ):
                print(
                    f"[Warning] DISCRITUNE_EVAL_CKPT_IN not found: {self.DISCRITUNE_EVAL_CKPT_IN}"
                )

    def _validate_paths(self):
        # COCO TRAIN
        if self.RUN_MODE in {"TRAIN", "ALL"} and self.EXPERIMENT_TYPE == "clipcap_coco":
            if self.COCO_IMAGES_ROOT and (not os.path.exists(self.COCO_IMAGES_ROOT)):
                print(f"[Warning] COCO_IMAGES_ROOT not found: {self.COCO_IMAGES_ROOT}")
            if self.COCO_KARPATHY_JSON and (
                not os.path.exists(self.COCO_KARPATHY_JSON)
            ):
                print(
                    f"[Warning] COCO_KARPATHY_JSON not found: {self.COCO_KARPATHY_JSON}"
                )

        # COCO evaluate
        if self.RUN_MODE in {"E1", "E3", "ALL"}:
            if self.COCO_IMAGES_ROOT and (not os.path.exists(self.COCO_IMAGES_ROOT)):
                print(f"[Warning] COCO_IMAGES_ROOT not found: {self.COCO_IMAGES_ROOT}")
            if self.COCO_KARPATHY_JSON and (
                not os.path.exists(self.COCO_KARPATHY_JSON)
            ):
                print(
                    f"[Warning] COCO_KARPATHY_JSON not found: {self.COCO_KARPATHY_JSON}"
                )

        # Concap TRAIN
        if (
            self.RUN_MODE in {"TRAIN", "ALL"}
            and self.EXPERIMENT_TYPE == "clipcap_concap"
        ):
            if self.CC_TRAIN_DIR and (not os.path.exists(self.CC_TRAIN_DIR)):
                print(f"[Warning] CC_TRAIN_DIR not found: {self.CC_TRAIN_DIR}")

        # Concap evaluate
        if self.RUN_MODE in {"E2", "E4", "ALL"}:
            if self.CC_VAL_DIR and (not os.path.exists(self.CC_VAL_DIR)):
                print(f"[Warning] CC_VAL_DIR not found: {self.CC_VAL_DIR}")

        # Weights
        if self.WEIGHTS_PATH and (not os.path.exists(self.WEIGHTS_PATH)):
            print(f"[Warning] WEIGHTS_PATH not found: {self.WEIGHTS_PATH}")

    def to_dict(self):
        return dict(self.__dict__)

    def dump_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

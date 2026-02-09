import json
import os
from typing import Any, Dict, List

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class CocoKarpathyDataset(Dataset):
    def __init__(
        self,
        karpathy_json: str,
        images_root: str,
        split: str,
        include_restval_in_train: bool = False,
        return_ids: bool = False,
    ):
        assert split in {"train", "val", "test"}
        self.return_ids = return_ids
        self.images_root = images_root

        if not os.path.exists(karpathy_json):
            raise FileNotFoundError(f"Karpathy json not found: {karpathy_json}")
        if not os.path.exists(images_root):
            raise FileNotFoundError(f"COCO images root not found: {images_root}")

        with open(karpathy_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "images" not in data:
            raise ValueError("Karpathy json must contain top-level key 'images'")

        def match_split(s: str) -> bool:
            if split == "train":
                return s == "train" or (include_restval_in_train and s == "restval")
            return s == split

        samples: List[Dict[str, Any]] = []
        for img in data["images"]:
            s = img.get("split", None)
            if not isinstance(s, str) or not match_split(s):
                continue

            filename = img.get("filename", None) or img.get("file_name", None)
            if not isinstance(filename, str):
                continue

            filepath = img.get("filepath", "")
            if isinstance(filepath, str) and filepath:
                image_path = os.path.join(images_root, filepath, filename)
            else:
                image_path = os.path.join(images_root, filename)

            image_id = img.get("cocoid", None)
            if image_id is None:
                image_id = img.get("imgid", None)
            if image_id is None:
                image_id = img.get("id", None)

            captions: List[str] = []
            sents = img.get("sentences", None)
            if isinstance(sents, list):
                for sent in sents:
                    raw = sent.get("raw", None)
                    if isinstance(raw, str):
                        captions.append(raw)

            samples.append(
                {
                    "image_path": image_path,
                    "captions": captions,
                    "image_id": int(image_id) if image_id is not None else None,
                }
            )

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples loaded. Check split={split} and json format."
            )

        samples.sort(key=lambda x: x["image_path"])
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        path = entry["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to load image: {path} ({e})")

        if self.return_ids:
            return img, entry["captions"], entry["image_id"], path
        return img, entry["captions"]


class ConceptualCaptionsDataset(Dataset):
    def __init__(
        self,
        out_dir: str,
        return_ids: bool = False,
        use_caption_as_ref: bool = True,
    ):
        self.return_ids = return_ids

        if not os.path.exists(out_dir):
            raise FileNotFoundError(f"CC out_dir not found: {out_dir}")

        self.images_root = os.path.join(out_dir, "images")
        self.meta_path = os.path.join(out_dir, "meta.jsonl")

        if not os.path.exists(self.images_root):
            raise FileNotFoundError(f"CC images dir not found: {self.images_root}")
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"CC meta.jsonl not found: {self.meta_path}")

        samples: List[Dict[str, Any]] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                image_name = obj.get("image", None)
                if not isinstance(image_name, str):
                    continue

                image_path = os.path.join(self.images_root, image_name)
                if not os.path.exists(image_path):
                    continue

                idx_val = obj.get("idx", None)
                try:
                    image_id = int(idx_val) if idx_val is not None else None
                except Exception:
                    image_id = None

                caption = obj.get("caption", "")
                if not isinstance(caption, str):
                    caption = ""

                captions: List[str]
                if use_caption_as_ref and caption:
                    captions = [caption]
                else:
                    captions = []

                samples.append(
                    {
                        "image_path": image_path,
                        "captions": captions,
                        "image_id": image_id,
                    }
                )

        if len(samples) == 0:
            raise RuntimeError(f"No CC samples loaded from: {self.meta_path}")

        samples.sort(key=lambda x: x["image_path"])
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        path = entry["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to load image: {path} ({e})")

        if self.return_ids:
            return img, entry["captions"], entry["image_id"], path
        return img, entry["captions"]


def collate_fn_pil(batch):
    # train용: List[PIL], List[List[str]]
    images, caps = [], []
    for img, c in batch:
        images.append(img)
        caps.append(c)
    return images, caps


def collate_fn_pil_with_ids(batch):
    # eval용: List[PIL], List[List[str]], List[int|None]
    images, caps, ids, paths = [], [], [], []
    for img, c, i, p in batch:
        images.append(img)
        caps.append(c)
        ids.append(i)
        paths.append(p)
    return images, caps, ids, paths

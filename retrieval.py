import hashlib
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------
def seed_from_key(key: str, base_seed: int = 42) -> int:
    h = hashlib.sha1(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


# ---------------------------------------------------------------------
# Image embedding cache
# ---------------------------------------------------------------------
@torch.no_grad()
def build_image_embedding_cache(
    retriever,
    all_paths: List[str],
    batch_size: int = 256,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a path -> normalized CLIP image embedding cache (stored on CPU).

    Requires retriever to have:
      - retriever.device
      - retriever.preprocess(PIL.Image) -> Tensor[C,H,W]
      - retriever.model.encode_image(image_tensor) -> Tensor[N,D]
    """
    import warnings

    from PIL import ImageFile

    # 깨진/잘린 이미지 때문에 PIL이 죽지 않도록(현실적으로 ConCap에 종종 있음)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 너무 큰 이미지(폭탄/메모리 폭증) 스킵 기준
    MAX_PIXELS = 80_000_000  # ~80MP

    device = retriever.device
    cache: Dict[str, torch.Tensor] = {}
    paths = [str(p) for p in all_paths]

    it = range(0, len(paths), batch_size)
    if verbose:
        from tqdm import tqdm

        it = tqdm(it, desc="[Cache] Build image embeddings", leave=False)

    for start in it:
        chunk = paths[start : start + batch_size]

        imgs: List[Image.Image] = []
        valid_paths: List[str] = []

        for p in chunk:
            try:
                # DecompressionBombWarning을 "경고"가 아니라 "예외"로 취급해서 안전하게 스킵
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        if w * h > MAX_PIXELS:
                            # 초대형 이미지는 스킵
                            continue
                        imgs.append(im)
                        valid_paths.append(p)
            except (Image.DecompressionBombWarning, OSError, ValueError):
                # 폭탄 경고 / 깨진 파일 / 디코딩 오류 등은 스킵
                continue

        if len(imgs) == 0:
            # 이 chunk에서 유효 이미지가 하나도 없으면 그냥 넘어감
            continue

        image_tensor = torch.stack([retriever.preprocess(im) for im in imgs], dim=0).to(
            device
        )
        feats = retriever.model.encode_image(image_tensor)  # (N, D)
        feats = F.normalize(feats.float(), dim=-1).to(dtype=dtype).cpu()

        # zip 대상은 chunk가 아니라 valid_paths여야 함(스킵한 만큼 길이가 달라짐)
        for p, f in zip(valid_paths, feats):
            cache[p] = f.contiguous()

    return cache


@torch.no_grad()
def get_image_feat_from_cache(
    path: str,
    retriever,
    cache: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Fetch normalized image feature from cache. If missing, compute and insert.

    Returns:
        Tensor[D] on CPU (normalized)
    """
    p = str(path)
    if p in cache:
        return cache[p]

    device = retriever.device
    with Image.open(p) as im:
        im = im.convert("RGB")

    image_tensor = torch.stack([retriever.preprocess(im)], dim=0).to(device)
    feat = retriever.model.encode_image(image_tensor)  # (1, D)
    feat = (
        F.normalize(feat.float(), dim=-1).to(dtype=dtype).cpu().squeeze(0).contiguous()
    )

    cache[p] = feat
    return cache[p]

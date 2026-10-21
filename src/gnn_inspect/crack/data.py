"""METU concrete-crack dataset (Özgenel 2018, CC BY 4.0): 20k Positive / 20k Negative 227x227 patches.

Raw files live in data/raw/ (untrusted, git-ignored) and are only ever opened as images.
Images are center-cropped to 224 and cached as one uint8 array per split.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "metu_concrete"
CACHE = ROOT / "data" / "cache"
SEED = 0


def make_split(n_train: int = 8000, seed: int = SEED) -> dict:
    """Stratified 70/10/20 split of all 40k files; training then uses an n_train subset of the 70% part.

    Val and test use the full 10% / 20% parts. Test is never touched during training or model selection.
    """
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for label, cls in enumerate(["Negative", "Positive"]):
        files = sorted((RAW / cls).glob("*.jpg"))
        idx = rng.permutation(len(files))
        n = len(files)
        cut_tr, cut_va = int(0.7 * n), int(0.8 * n)
        parts = {"train": idx[:cut_tr][: n_train // 2], "val": idx[cut_tr:cut_va], "test": idx[cut_va:]}
        for k, ids in parts.items():
            out[k] += [(str(files[i].relative_to(RAW)), label) for i in ids]
    for k in out:
        rng.shuffle(out[k])
    return out


def load_split(name: str, n_train: int = 8000):
    """Return (uint8 images N,224,224,3 memory-mapped, int64 labels N)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{name}{n_train if name == 'train' else ''}"
    xf, yf = CACHE / f"{tag}_x.npy", CACHE / f"{tag}_y.npy"
    if not xf.exists():
        items = make_split(n_train)[name]
        x = np.lib.format.open_memmap(xf, mode="w+", dtype=np.uint8, shape=(len(items), 224, 224, 3))
        for i, (rel, _) in enumerate(items):
            x[i] = np.asarray(Image.open(RAW / rel).convert("RGB"))[1:225, 1:225]
        x.flush()
        np.save(yf, np.array([l for _, l in items], dtype=np.int64))
        (CACHE / f"{tag}_files.json").write_text(json.dumps([r for r, _ in items]))
    return np.load(xf, mmap_mode="r"), np.load(yf)

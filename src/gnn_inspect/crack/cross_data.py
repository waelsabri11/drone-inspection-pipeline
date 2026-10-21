"""Out-of-domain patch sets built from public segmentation datasets (never used for training).

Labelling rule (both datasets): slide a non-overlapping 128x128 window over each image; a patch is
POSITIVE if >= 1% of its pixels are crack in the ground-truth mask (>= 164 px), NEGATIVE if it has 0 crack
pixels, and is DROPPED otherwise (ambiguous hairline fragments). Patches are bilinearly upsampled to 224.
The source image id is kept so confidence intervals can be bootstrapped over images, not patches.

- DeepCrack (Liu et al., Neurocomputing 2019): 537 images 544x384 with masks, mixed concrete/brick/stone
  scenes. https://github.com/yhlleo/DeepCrack  (no license file; academic use, citation requested)
- CrackForest (Shi et al., IEEE T-ITS 2016): 118 road-pavement images 480x320 with masks.
  https://github.com/cuilimeng/CrackForest-dataset  (no license file; citation requested)
"""

import numpy as np
import scipy.io as sio
from PIL import Image

from .data import CACHE, ROOT

RAW = ROOT / "data" / "raw"
WIN, MIN_FRAC = 128, 0.01


def patch_label(frac: float) -> int | None:
    """Crack-pixel fraction of a window -> 1 (positive), 0 (negative) or None (ambiguous, dropped)."""
    if frac >= MIN_FRAC:
        return 1
    return 0 if frac == 0 else None


def _sources(name):
    if name == "deepcrack":
        for sub in ("train", "test"):
            for f in sorted((RAW / "deepcrack" / f"{sub}_img").glob("*.jpg")):
                m = np.asarray(Image.open(RAW / "deepcrack" / f"{sub}_lab" / (f.stem + ".png")).convert("L")) > 127
                yield f"{sub}/{f.stem}", f, m
    else:
        for f in sorted((RAW / "crackforest" / "groundTruth").glob("*.mat")):
            g = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["groundTruth"]
            yield f.stem, RAW / "crackforest" / "image" / (f.stem + ".jpg"), np.asarray(g.Segmentation) == 2


def load_cross(name: str):
    """(uint8 patches N,224,224,3; labels N; source-image index N)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"cross_{name}.npz"
    if not f.exists():
        xs, ys, ids = [], [], []
        for k, (_, img, mask) in enumerate(_sources(name)):
            im = Image.open(img).convert("RGB")
            h, w = mask.shape
            assert im.size == (w, h), (img, im.size, mask.shape)
            for y0 in range(0, h - WIN + 1, WIN):
                for x0 in range(0, w - WIN + 1, WIN):
                    frac = mask[y0 : y0 + WIN, x0 : x0 + WIN].mean()
                    lab = patch_label(frac)
                    if lab is None:
                        continue
                    xs.append(np.asarray(im.crop((x0, y0, x0 + WIN, y0 + WIN)).resize((224, 224), Image.BILINEAR)))
                    ys.append(lab)
                    ids.append(k)
        np.savez(f, x=np.stack(xs), y=np.array(ys), g=np.array(ids))
    d = np.load(f)
    return d["x"], d["y"], d["g"]

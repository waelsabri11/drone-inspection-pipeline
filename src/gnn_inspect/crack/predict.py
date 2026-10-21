"""Cache the detector's hard predictions on the val / test patch pools (used by the pipeline)."""

import numpy as np
import torch

from .data import ROOT, load_split
from .models import build
from .train import device, predict


def pool_predictions(arch: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(true labels, predicted labels) for every patch of a split; cached in checkpoints/."""
    f = ROOT / "checkpoints" / f"{arch}_{split}_pred.npy"
    x, y = load_split(split)
    if not f.exists():
        m = build(arch, pretrained=False)
        m.load_state_dict(torch.load(ROOT / "checkpoints" / f"{arch}.pt"))
        dev = device()
        np.save(f, predict(m.to(dev), x, dev))
    return y, np.load(f)

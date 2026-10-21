"""Evaluate every trained checkpoint, with NO fine-tuning, on out-of-domain crack patches.

    python -m gnn_inspect.crack.cross

Per (model, augmentation, seed): precision / recall / F1 at the fixed 0.5 threshold and PR-AUC.
Confidence intervals: 1000-replicate bootstrap over SOURCE IMAGES (patches of one photo stay together);
in each replicate the metric is averaged over seeds, so CIs reflect test-set sampling, while the
across-seed std is reported separately.
"""

import json
import re

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .cross_data import load_cross
from .data import ROOT
from .metrics import prf
from .models import build, prep
from .train import device

DATASETS = ("deepcrack", "crackforest")
ARCHS = ("resnet18", "deit_tiny")


@torch.no_grad()
def probs(path, arch, x, dev, bs=128):
    m = build(arch, pretrained=False)
    m.load_state_dict(torch.load(path))
    m.to(dev).eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(m(prep(torch.from_numpy(x[i : i + bs]).to(dev))).softmax(1)[:, 1].cpu())
    return torch.cat(out).numpy()


def metrics(y, p):
    r = prf(y, (p >= 0.5).astype(int))
    return {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "pr_auc": float(average_precision_score(y, p))}


def main():
    dev = device()
    ck = ROOT / "checkpoints"
    runs = {}
    for arch in ARCHS:
        for aug in ("light", "strong"):
            for seed in range(3):
                tag = arch if (aug == "light" and seed == 0) else f"{arch}_{aug}_s{seed}"
                if (ck / f"{tag}.pt").exists():
                    runs[(arch, aug, seed)] = ck / f"{tag}.pt"
    out = {"runs": {}, "summary": {}}
    P = {}
    data = {d: load_cross(d) for d in DATASETS}
    for key, path in runs.items():
        for d in DATASETS:
            P[key + (d,)] = probs(path, key[0], data[d][0], dev)
            out["runs"]["|".join(map(str, key)) + "|" + d] = metrics(data[d][1], P[key + (d,)])
    rng = np.random.default_rng(0)
    for arch in ARCHS:
        for aug in ("light", "strong"):
            seeds = [s for s in range(3) if (arch, aug, s) in runs]
            if not seeds:
                continue
            for d in DATASETS:
                _, y, g = data[d]
                ids = np.unique(g)
                by = {i: np.flatnonzero(g == i) for i in ids}
                point = [out["runs"][f"{arch}|{aug}|{s}|{d}"] for s in seeds]
                boots = {k: [] for k in point[0]}
                for _ in range(1000):
                    sel = np.concatenate([by[i] for i in rng.choice(ids, len(ids))])
                    if y[sel].min() == y[sel].max():
                        continue
                    ms = [metrics(y[sel], P[(arch, aug, s, d)][sel]) for s in seeds]
                    for k in boots:
                        boots[k].append(np.mean([m[k] for m in ms]))
                out["summary"][f"{arch}|{aug}|{d}"] = {
                    "n_seeds": len(seeds), "n_patches": int(len(y)), "n_images": int(len(ids)), "prevalence": float(y.mean()),
                    **{k: {"mean": float(np.mean([p[k] for p in point])), "seed_std": float(np.std([p[k] for p in point])),
                           "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]} for k in boots}}
                print(arch, aug, d, {k: round(v["mean"], 3) for k, v in out["summary"][f"{arch}|{aug}|{d}"].items() if isinstance(v, dict)}, flush=True)
    (ROOT / "results" / "crack").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "crack" / "cross_dataset.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

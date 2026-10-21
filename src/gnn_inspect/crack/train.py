"""Fine-tune ResNet-18 and DeiT-Tiny on the crack patches; report test P/R/F1 + confusion matrix.

    python -m gnn_inspect.crack.train --arch resnet18
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import ROOT, SEED, load_split
from .metrics import prf
from .models import build, prep

LR = {"resnet18": 5e-4, "deit_tiny": 1e-4}


def _moderate_one(img):
    """img: uint8 (H,W,3) tensor -> uint8. Moderate domain-shift recipe, fixed a priori (not tuned on any
    external data). Crop/blur/erasing are kept mild so thin cracks stay visible."""
    import torchvision.transforms.v2 as T
    x = img.permute(2, 0, 1)
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(1, 2))
    tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        T.RandomApply([T.JPEG((40, 95))], p=0.5),
    ])
    x = tf(x).float()
    if torch.rand(1).item() < 0.3:  # light gaussian sensor noise
        x = x + torch.randn_like(x) * torch.empty(1).uniform_(2, 8).item()
    x = x.clamp(0, 255).to(torch.uint8)
    x = T.RandomErasing(p=0.2, scale=(0.02, 0.08), value=0)(x)  # small patch, light
    return x.permute(1, 2, 0)


def augment_batch(xb):
    return torch.stack([_moderate_one(im) for im in xb])


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model, x, dev, bs=128):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = prep(torch.from_numpy(np.ascontiguousarray(x[i : i + bs])).to(dev))
        out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "deit_tiny"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--aug", choices=["light", "moderate"], default="light")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = a.arch if (a.aug == "light" and a.seed == SEED) else f"{a.arch}_{a.aug}_s{a.seed}"
    dev = device()
    xtr, ytr = load_split("train", a.n_train)
    xva, yva = load_split("val")
    xte, yte = load_split("test")
    print(f"{a.arch} on {dev}: train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)

    model = build(a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[a.arch], weight_decay=0.05)
    steps = a.epochs * (len(ytr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR[a.arch], total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(a.seed)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g).numpy()
        for i in range(len(ytr) // a.bs):
            idx = np.sort(perm[i * a.bs : (i + 1) * a.bs])
            xb = torch.from_numpy(np.ascontiguousarray(xtr[idx]))
            if a.aug == "moderate":
                xb = augment_batch(xb)
            xb = xb.to(dev)
            if a.aug == "light":
                if torch.rand(1, generator=g).item() < 0.5:
                    xb = xb.flip(2)  # horizontal flip of the whole batch
                if torch.rand(1, generator=g).item() < 0.5:
                    xb = xb.flip(1)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            loss = F.cross_entropy(model(prep(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        vf1 = prf(yva, predict(model, xva, dev))["f1"]
        print(f"epoch {ep+1} loss {loss.item():.4f} val_f1 {vf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best:  # model selection on validation only
            best = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = prf(yte, predict(model, xte, dev))
    res.update(arch=a.arch, val_f1=best, train_seconds=time.time() - t0, device=str(dev),
               n_train=len(ytr), n_val=len(yva), n_test=len(yte), epochs=a.epochs, seed=a.seed, aug=a.aug)
    print(json.dumps(res))
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save(best_state, ROOT / "checkpoints" / f"{tag}.pt")
    out = ROOT / "results" / "crack"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

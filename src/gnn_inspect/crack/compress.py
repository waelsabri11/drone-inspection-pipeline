"""Post-training quantization (int8) and magnitude pruning of the trained crack classifiers.

Variants per model, all evaluated on the SAME held-out test split on CPU:
  torch fp32 | onnxruntime fp32 | ORT dynamic int8 | ORT static int8 (QDQ, calibrated on 256 val images)
  | PyTorch dynamic int8 (nn.Linear) | global magnitude pruning 50% / 80% (+1 epoch masked fine-tune)

    python -m gnn_inspect.crack.compress --arch resnet18
"""

import argparse
import gzip
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from torch.nn.utils import prune

from .data import ROOT, SEED, load_split
from .metrics import prf
from .models import build, prep
from .train import LR, device

warnings.filterwarnings("ignore")
THREADS = 4
WORK = ROOT / "checkpoints" / "onnx"


def mb(path) -> float:
    return Path(path).stat().st_size / 1e6


def state_mb(model) -> tuple[float, float]:
    """(raw fp32 state-dict MB, gzip-compressed MB). The latter shows what sparsity buys in storage."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    return len(raw) / 1e6, len(gzip.compress(raw, compresslevel=6)) / 1e6


def nonzero_frac(model) -> float:
    ws = [m.weight for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    return float(sum((w != 0).sum().item() for w in ws) / sum(w.numel() for w in ws))


def torch_logits(model, x, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(prep(torch.from_numpy(np.ascontiguousarray(x[i : i + bs])))).argmax(1))
    return torch.cat(out).numpy()


def ort_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_pred(sess, x, bs=64):
    out = []
    for i in range(0, len(x), bs):
        xb = prep(torch.from_numpy(np.ascontiguousarray(x[i : i + bs]))).numpy()
        out.append(sess.run(None, {"input": xb})[0].argmax(1))
    return np.concatenate(out)


def latency_ms(fn, batch, warm=5, runs=30):
    for _ in range(warm):
        fn(batch)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(batch)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


class Calib(CalibrationDataReader):
    def __init__(self, x):
        self.it = iter([{"input": prep(torch.from_numpy(np.ascontiguousarray(x[i : i + 1]))).numpy()} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def measure_torch(model, xte, yte, lat_x):
    model.eval()
    f = lambda b: model(b).detach()
    with torch.no_grad():
        l1 = latency_ms(f, lat_x[:1])
        l32 = latency_ms(f, lat_x, runs=10)
    return prf(yte, torch_logits(model, xte)), l1, l32


def measure_ort(path, xte, yte, lat_x):
    s = ort_session(path)
    f = lambda b: s.run(None, {"input": b})
    l1 = latency_ms(f, lat_x[:1].numpy())
    l32 = latency_ms(f, lat_x.numpy(), runs=10)
    return prf(yte, ort_pred(s, xte)), l1, l32


def finetune_masked(model, xtr, ytr, arch, dev, epochs=1, bs=64):
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR[arch] / 5, weight_decay=0.0)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr), generator=g).numpy()
        for i in range(len(ytr) // bs):
            idx = np.sort(perm[i * bs : (i + 1) * bs])
            xb = torch.from_numpy(np.ascontiguousarray(xtr[idx])).to(dev)
            loss = F.cross_entropy(model(prep(xb)), torch.from_numpy(ytr[idx]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "deit_tiny"], required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    xte, yte = load_split("test")
    xva, _ = load_split("val")
    xtr, ytr = load_split("train")
    lat_x = prep(torch.from_numpy(np.ascontiguousarray(xte[:32])))
    WORK.mkdir(parents=True, exist_ok=True)
    state = torch.load(ROOT / "checkpoints" / f"{a.arch}.pt")

    def fresh():
        m = build(a.arch, pretrained=False)
        m.load_state_dict(state)
        return m.eval()

    rows = []

    def add(name, kind, metrics, l1, l32, size, **extra):
        r = dict(variant=name, kind=kind, size_mb=size, lat_bs1_ms=l1, lat_bs32_ms=l32, **{k: metrics[k] for k in ("precision", "recall", "f1", "accuracy", "confusion")}, **extra)
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    m = fresh()
    met, l1, l32 = measure_torch(m, xte, yte, lat_x)
    add("PyTorch fp32", "baseline", met, l1, l32, state_mb(m)[0])

    # --- ONNX export + ORT variants
    fp32 = WORK / f"{a.arch}_fp32.onnx"
    torch.onnx.export(m, torch.zeros(1, 3, 224, 224), str(fp32), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    # shape inference needed by the quantizer
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(fp32)), fp32)
    met, l1, l32 = measure_ort(fp32, xte, yte, lat_x)
    add("ONNX Runtime fp32", "baseline", met, l1, l32, mb(fp32))

    dyn = WORK / f"{a.arch}_dyn_int8.onnx"
    quantize_dynamic(str(fp32), str(dyn), weight_type=QuantType.QInt8)
    met, l1, l32 = measure_ort(dyn, xte, yte, lat_x)
    add("ORT dynamic int8", "quantization", met, l1, l32, mb(dyn))

    st = WORK / f"{a.arch}_static_int8.onnx"
    rng = np.random.default_rng(SEED)
    calib = xva[np.sort(rng.choice(len(xva), 256, replace=False))]
    quantize_static(str(fp32), str(st), Calib(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    met, l1, l32 = measure_ort(st, xte, yte, lat_x)
    add("ORT static int8 (QDQ)", "quantization", met, l1, l32, mb(st))

    # --- PyTorch dynamic int8 on Linear layers
    torch.backends.quantized.engine = "qnnpack"  # only engine in Apple-silicon builds
    q = torch.ao.quantization.quantize_dynamic(fresh(), {torch.nn.Linear}, dtype=torch.qint8)
    met, l1, l32 = measure_torch(q, xte, yte, lat_x)
    add("PyTorch dynamic int8 (Linear)", "quantization", met, l1, l32, state_mb(q)[0])

    # --- global magnitude pruning, then 1 epoch of masked fine-tuning on the train subset
    dev = device()
    for sparsity in (0.5, 0.8):
        p = fresh()
        params = [(mod, "weight") for mod in p.modules()
                  if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) and mod.weight.shape[0] != 2]  # keep classifier dense
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
        met0 = prf(yte, torch_logits(p, xte))  # no fine-tune
        p = finetune_masked(p, xtr, ytr, a.arch, dev)
        for mod, n in params:
            prune.remove(mod, n)
        met, l1, l32 = measure_torch(p, xte, yte, lat_x)
        raw, gz = state_mb(p)
        add(f"magnitude prune {int(sparsity*100)}% (no fine-tune)", "pruning", met0, float("nan"), float("nan"), raw, gzip_mb=gz,
            nonzero_weights=nonzero_frac(p))
        add(f"magnitude prune {int(sparsity*100)}% + 1 ep fine-tune", "pruning", met, l1, l32, raw, gzip_mb=gz,
            nonzero_weights=nonzero_frac(p))

    out = ROOT / "results" / "crack"
    (out / f"{a.arch}_compression.json").write_text(json.dumps(
        {"arch": a.arch, "threads": THREADS, "n_test": len(yte), "latency_note": "median ms per call, CPU, batch 1 and 32, 224x224",
         "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

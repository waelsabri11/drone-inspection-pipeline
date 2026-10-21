"""Does fusing attention / LayerNorm / GELU (onnxruntime.transformers optimizer) rescue ViT int8 latency?

Hypothesis from the compression table: int8 gave no speed-up for DeiT-Tiny because Q/DQ nodes surround
unfused LayerNorm / softmax / GELU. We fuse first, then quantize (dynamic and static), and re-measure with
the SAME protocol as compress.py (CPU, 4 threads, median ms, batch 1 and 32, 8,000 test images).

    python -m gnn_inspect.crack.vit_fused
"""

import json
from collections import Counter

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.transformers import optimizer

from .compress import SEED, WORK, Calib, THREADS, measure_ort, mb
from .data import ROOT, load_split
from .models import prep

FUSED_OPS = ("Attention", "MultiHeadAttention", "LayerNormalization", "SkipLayerNormalization", "Gelu", "FastGelu", "BiasGelu")


def ops(path):
    c = Counter(n.op_type for n in onnx.load(path).graph.node)
    return {k: c[k] for k in FUSED_OPS if c[k]} | {"total_nodes": sum(c.values()), "Softmax": c["Softmax"], "Erf": c["Erf"]}


def main():
    torch.set_num_threads(THREADS)
    xte, yte = load_split("test")
    xva, _ = load_split("val")
    lat_x = prep(torch.from_numpy(np.ascontiguousarray(xte[:32])))
    fp32 = WORK / "deit_tiny_fp32.onnx"
    fused = WORK / "deit_tiny_fused_fp32.onnx"
    m = optimizer.optimize_model(str(fp32), model_type="vit", num_heads=3, hidden_size=192, opt_level=1)
    m.save_model_to_file(str(fused))
    variants = {"ORT fp32 (unfused)": fp32, "ORT fp32 (fused)": fused}

    dyn = WORK / "deit_tiny_fused_dyn_int8.onnx"
    quantize_dynamic(str(fused), str(dyn), weight_type=QuantType.QInt8)
    variants["fused + dynamic int8"] = dyn
    st = WORK / "deit_tiny_fused_static_int8.onnx"
    rng = np.random.default_rng(SEED)
    calib = xva[np.sort(rng.choice(len(xva), 256, replace=False))]
    try:
        quantize_static(str(fused), str(st), Calib(calib), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
        variants["fused + static int8 (QDQ)"] = st
    except Exception as e:  # report, don't hide
        print("static quantization of fused graph failed:", repr(e)[:300])

    # Re-measure the unfused int8 variants with this same process for a like-for-like comparison.
    variants["unfused + dynamic int8"] = WORK / "deit_tiny_dyn_int8.onnx"
    variants["unfused + static int8 (QDQ)"] = WORK / "deit_tiny_static_int8.onnx"
    rows = []
    for name, p in variants.items():
        met, l1, l32 = measure_ort(p, xte, yte, lat_x)
        r = {"variant": name, "size_mb": mb(p), "f1": met["f1"], "precision": met["precision"], "recall": met["recall"],
             "confusion": met["confusion"], "lat_bs1_ms": l1, "lat_bs32_ms": l32, "ops": ops(str(p))}
        rows.append(r)
        print(json.dumps(r), flush=True)
    (ROOT / "results" / "crack" / "deit_fused_int8.json").write_text(json.dumps({"threads": THREADS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

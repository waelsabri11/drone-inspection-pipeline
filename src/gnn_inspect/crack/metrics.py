import numpy as np


def confusion(y, p) -> np.ndarray:
    m = np.zeros((2, 2), dtype=int)
    for t, q in zip(y, p):
        m[int(t), int(q)] += 1
    return m


def prf(y, p) -> dict:
    """Metrics for the positive (crack) class."""
    m = confusion(y, p)
    tp, fp, fn = m[1, 1], m[0, 1], m[1, 0]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": float((m[0, 0] + tp) / m.sum()),
            "confusion": m.tolist()}

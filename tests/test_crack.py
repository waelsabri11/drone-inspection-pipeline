import numpy as np
import pytest

from gnn_inspect.crack.data import RAW, make_split
from gnn_inspect.crack.metrics import confusion, prf


def test_prf_known_values():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0, 0]
    r = prf(y, p)
    assert confusion(y, p).tolist() == [[4, 1], [1, 2]]
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3) and r["accuracy"] == pytest.approx(6 / 8)


def test_prf_degenerate_no_positive_predictions():
    r = prf([1, 0], [0, 0])
    assert r["precision"] == 0 and r["recall"] == 0 and r["f1"] == 0


@pytest.mark.skipif(not (RAW / "Positive").exists(), reason="dataset not downloaded (make data)")
def test_split_is_disjoint_stratified_and_deterministic():
    a, b = make_split(), make_split()
    assert a == b
    files = {k: {f for f, _ in v} for k, v in a.items()}
    assert not (files["train"] & files["val"]) and not (files["train"] & files["test"]) and not (files["val"] & files["test"])
    for k in a:
        labels = np.array([l for _, l in a[k]])
        assert abs(labels.mean() - 0.5) < 0.01

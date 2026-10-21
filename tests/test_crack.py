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


def test_cross_dataset_patch_labelling_rule():
    from gnn_inspect.crack.cross_data import patch_label

    assert patch_label(0.0) == 0
    assert patch_label(0.0001) is None  # hairline fragment: dropped
    assert patch_label(0.0099) is None
    assert patch_label(0.01) == 1 and patch_label(0.5) == 1


def test_strong_augmentation_preserves_shape_and_dtype():
    import torch

    from gnn_inspect.crack.train import strong_aug

    x = torch.randint(0, 255, (3, 224, 224, 3), dtype=torch.uint8)
    y = strong_aug(x)
    assert y.shape == x.shape and y.dtype == torch.uint8

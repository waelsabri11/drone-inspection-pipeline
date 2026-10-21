import numpy as np
from scipy.stats import spearmanr


def ndcg_at_k(scores: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """NDCG@k with the true severity as (linear) gain."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argsort(-scores, kind="stable")[:k]
    ideal = np.sort(truth)[::-1][:k]
    return float((truth[top] * disc[: len(top)]).sum() / (ideal * disc[: len(ideal)]).sum())


def spearman(scores: np.ndarray, truth: np.ndarray) -> float:
    return float(spearmanr(scores, truth).statistic)

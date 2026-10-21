"""Node prioritisation: centrality measures combined with detected defect evidence."""

import numpy as np

from . import baselines
from .data import Site


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    r = v.max() - v.min()
    return (v - v.min()) / r if r > 0 else np.zeros_like(v)


def centrality_table(site: Site) -> dict[str, np.ndarray]:
    """The four centralities used to prioritise inspection points."""
    return {k: f(site) for k, f in baselines.CENTRALITY.items()}


def combined_priority(site: Site, evidence: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Defect evidence (e.g. detector crack fraction, smoothed over neighbours) blended with
    normalised betweenness: (1-alpha)*evidence + alpha*betweenness. Nodes on many shortest paths
    are structurally more loaded, so they get a boost when evidence is equal."""
    ev = np.asarray(evidence, float)
    neigh = np.array([ev[list(site.graph[i])].mean() for i in range(site.n)])
    return (1 - alpha) * _norm(0.5 * ev + 0.5 * neigh) + alpha * _norm(site.betweenness)

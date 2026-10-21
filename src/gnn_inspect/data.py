"""Synthetic drone-inspection sites.

A site is a graph of waypoints (nodes) linked by traversable segments (edges).
Each waypoint hides a true defect severity; the drone only sees a noisy prior
from an earlier coarse survey, on part of the nodes.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

FEATURES = ["x", "y", "prior", "prior_observed", "inspect_cost", "dist_to_depot"]


@dataclass
class Site:
    graph: nx.Graph          # edge attribute "length"
    pos: np.ndarray          # (n, 2) in the unit square
    dist: np.ndarray         # (n, n) shortest-path lengths
    severity: np.ndarray     # (n,) true defect severity (hidden from methods)
    prior: np.ndarray        # (n,) noisy survey score, 0 where unobserved
    observed: np.ndarray     # (n,) 1 if the survey covered the node
    cost: np.ndarray         # (n,) battery spent inspecting the node
    betweenness: np.ndarray  # (n,) normalized betweenness (also drives load defects)

    @property
    def n(self) -> int:
        return len(self.pos)

    def features(self) -> np.ndarray:
        d0 = self.dist[0] / self.dist[0].max()
        return np.column_stack(
            [self.pos, self.prior, self.observed, self.cost / 0.05, d0]
        ).astype(np.float32)

    def to_data(self) -> Data:
        edges = np.array(self.graph.edges()).T
        edge_index = np.concatenate([edges, edges[::-1]], axis=1)
        return Data(
            x=torch.from_numpy(self.features()),
            edge_index=torch.from_numpy(edge_index).long(),
            y=torch.from_numpy(self.severity.astype(np.float32)),
        )


def _build_graph(rng: np.random.Generator, n: int, k: int):
    pos = rng.random((n, 2))
    pos = pos[np.argsort(pos.sum(1))]  # node 0 (depot) is the corner closest to the origin
    euclid = cdist(pos, pos)
    pairs = set()
    for i in range(n):
        for j in np.argsort(euclid[i])[1 : k + 1]:
            pairs.add((min(i, j), max(i, j)))
    mst = minimum_spanning_tree(euclid).tocoo()  # guarantees connectivity
    pairs |= {(min(i, j), max(i, j)) for i, j in zip(mst.row, mst.col)}

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j in sorted(pairs):
        # lengths >= euclidean distance, so the euclidean A* heuristic stays admissible
        g.add_edge(int(i), int(j), length=float(euclid[i, j] * rng.uniform(1.0, 1.3)))
    return g, pos


def make_site(
    rng: np.random.Generator,
    n_range=(50, 80),
    k: int = 4,
    load_weight: float = 0.5,
    survey_coverage: float = 0.6,
    survey_noise: float = 0.35,
) -> Site:
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    g, pos = _build_graph(rng, n, k)
    adj = nx.to_scipy_sparse_array(g, weight="length", nodelist=range(n))
    dist = shortest_path(adj, method="D", directed=False)

    bc = np.array(list(nx.betweenness_centrality(g, weight="length").values()))
    bc = bc / bc.max()

    # cracks: a few clusters spreading along the structure
    centers = rng.choice(n, size=int(rng.integers(2, 5)), replace=False)
    amp = rng.uniform(0.5, 1.5, size=len(centers))
    cracks = (amp[:, None] * np.exp(-dist[centers] ** 2 / (2 * 0.12**2))).sum(0)
    severity = cracks + load_weight * bc + 0.05 * np.abs(rng.normal(size=n))
    severity[0] = 0.0  # depot

    observed = (rng.random(n) < survey_coverage).astype(float)
    noisy = severity / severity.max() + rng.normal(0, survey_noise, size=n)
    prior = np.clip(noisy, 0, None) * observed

    cost = rng.uniform(0.01, 0.05, size=n)
    cost[0] = 0.0
    return Site(g, pos, dist, severity, prior, observed, cost, bc)


def make_dataset(seed: int, n_graphs: int, **kw) -> list[Site]:
    rng = np.random.default_rng(seed)
    return [make_site(rng, **kw) for _ in range(n_graphs)]


def split(sites: list, n_train: int, n_val: int):
    """Split by whole graphs: no node of a test site is ever seen in training."""
    return sites[:n_train], sites[n_train : n_train + n_val], sites[n_train + n_val :]

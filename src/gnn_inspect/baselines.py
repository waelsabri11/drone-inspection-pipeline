"""Hand-crafted rankings: the centrality measures from the original study, plus
non-learned baselines that use the survey prior."""

import networkx as nx
import numpy as np

from .data import Site


def _arr(d: dict, n: int) -> np.ndarray:
    return np.array([d[i] for i in range(n)], dtype=float)


def _filled_prior(site: Site) -> np.ndarray:
    seen = site.observed > 0
    return np.where(seen, site.prior, site.prior[seen].mean() if seen.any() else 0.0)


def degree(site: Site) -> np.ndarray:
    return _arr(dict(site.graph.degree()), site.n)


def betweenness(site: Site) -> np.ndarray:
    return site.betweenness


def closeness(site: Site) -> np.ndarray:
    return _arr(nx.closeness_centrality(site.graph, distance="length"), site.n)


def _proximity_graph(site: Site) -> nx.Graph:
    g = site.graph.copy()
    for _, _, e in g.edges(data=True):
        e["w"] = 1.0 / e["length"]
    return g


def pagerank(site: Site) -> np.ndarray:
    return _arr(nx.pagerank(_proximity_graph(site), weight="w"), site.n)


def prior(site: Site) -> np.ndarray:
    return _filled_prior(site)


def smoothed_prior(site: Site) -> np.ndarray:
    p = _filled_prior(site)
    neigh = np.array([p[list(site.graph[i])].mean() for i in range(site.n)])
    return 0.5 * p + 0.5 * neigh


def personalized_pagerank(site: Site) -> np.ndarray:
    p = _filled_prior(site) + 1e-6
    pers = {i: p[i] for i in range(site.n)}
    return _arr(nx.pagerank(_proximity_graph(site), weight="w", personalization=pers), site.n)


CENTRALITY = {
    "degree": degree,
    "betweenness": betweenness,
    "closeness": closeness,
    "pagerank": pagerank,
}
PRIOR_BASED = {
    "prior": prior,
    "smoothed_prior": smoothed_prior,
    "personalized_pagerank": personalized_pagerank,
}

"""Battery-constrained inspection itinerary driven by a node ranking."""

import heapq

import numpy as np

from .data import Site


def astar(site: Site, source: int, target: int) -> tuple[float, list[int]]:
    """A* on the site graph with the (admissible) euclidean heuristic."""
    h = lambda u: float(np.linalg.norm(site.pos[u] - site.pos[target]))
    best = {source: 0.0}
    parent = {source: None}
    heap = [(h(source), 0.0, source)]
    while heap:
        _, g, u = heapq.heappop(heap)
        if u == target:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return g, path[::-1]
        if g > best[u]:
            continue
        for v, e in site.graph[u].items():
            ng = g + e["length"]
            if ng < best.get(v, np.inf):
                best[v], parent[v] = ng, u
                heapq.heappush(heap, (ng + h(v), ng, v))
    return np.inf, []


def full_tour_cost(site: Site) -> float:
    """Reference length: walking the MST twice plus inspecting every node."""
    import networkx as nx

    mst = nx.minimum_spanning_tree(site.graph, weight="length")
    return 2 * mst.size(weight="length") + site.cost.sum()


def plan(site: Site, scores: np.ndarray, budget: float) -> tuple[float, list[int]]:
    """Greedy itinerary from the depot: repeatedly inspect the unvisited waypoint
    with the best score per battery cost (travel + inspection) that still leaves
    enough battery to fly back. Travel uses shortest paths (equal to A* lengths).

    Returns (true defect severity found, ordered list of inspected waypoints).
    """
    s = np.clip(np.asarray(scores, dtype=float), 0, None)
    s = s / s.max() + 1e-9 if s.max() > 0 else s + 1e-9
    d = site.dist
    visited = np.zeros(site.n, dtype=bool)
    visited[0] = True
    cur, left, found, route = 0, budget, 0.0, [0]
    while True:
        step = d[cur] + site.cost
        ok = (~visited) & (step + d[:, 0] <= left + 1e-12)
        if not ok.any():
            break
        j = int(np.argmax(np.where(ok, s / np.maximum(step, 1e-9), -np.inf)))
        left -= step[j]
        found += site.severity[j]
        visited[j] = True
        route.append(j)
        cur = j
    route.append(0)
    return found, route

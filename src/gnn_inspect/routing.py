"""Route solvers on a site graph: greedy + A* legs vs OR-Tools (near-exact, local search).

Three problems, all with the depot at node 0 and shortest-path travel between waypoints:
- orienteering: pick nodes to inspect under a battery limit, maximizing collected score;
- TSP: shortest closed tour through a given set of nodes;
- cover-VRP: visit a given set of nodes with several battery-limited drones, minimizing total length.
"""

import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .data import Site
from .planner import astar

SCALE = 10_000  # lengths -> ints for OR-Tools


def _solve(dist_int, n_veh, caps, penalties, time_limit, first="PATH_CHEAPEST_ARC", seed=0):
    """Generic OR-Tools routing run. penalties[i] None => node i is mandatory (depot excluded)."""
    n = len(dist_int)
    mgr = pywrapcp.RoutingIndexManager(n, n_veh, 0)
    rt = pywrapcp.RoutingModel(mgr)
    cb = rt.RegisterTransitCallback(lambda i, j: int(dist_int[mgr.IndexToNode(i)][mgr.IndexToNode(j)]))
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    rt.AddDimensionWithVehicleCapacity(cb, 0, list(caps), True, "battery")
    for i in range(1, n):
        if penalties[i] is not None:
            rt.AddDisjunction([mgr.NodeToIndex(i)], int(penalties[i]))
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, first)
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(time_limit * 1000))
    sol = rt.SolveWithParameters(p)
    if sol is None:
        return None
    routes = []
    for v in range(n_veh):
        idx, r = rt.Start(v), []
        while True:
            r.append(mgr.IndexToNode(idx))
            if rt.IsEnd(idx):
                break
            idx = sol.Value(rt.NextVar(idx))
        routes.append(r)
    return routes


def orienteering(site: Site, scores: np.ndarray, budget: float, vehicles: int = 1,
                 time_limit: float = 2.0) -> tuple[float, list[list[int]]]:
    """Maximize collected score under a per-drone battery `budget` (travel + inspection + return).

    Returns (true severity found, one route per drone, each starting and ending at the depot).
    """
    s = np.clip(np.asarray(scores, float), 0, None)
    s = s / s.max() if s.max() > 0 else s
    # arc i->j costs travel + inspection of j, so the battery dimension accumulates the true consumption
    arc = np.rint((site.dist + site.cost[None, :]) * SCALE).astype(np.int64)
    pen = [None] + [int(round(v * 1e6)) + 1 for v in s[1:]]
    routes = _solve(arc, vehicles, [int(budget * SCALE)] * vehicles, pen, time_limit)
    if routes is None:
        return 0.0, [[0, 0]] * vehicles
    visited = {j for r in routes for j in r if j != 0}
    return float(sum(site.severity[j] for j in visited)), routes


def tsp(site: Site, nodes: list[int], time_limit: float = 1.0) -> tuple[float, list[int]]:
    """Shortest closed tour from the depot through `nodes` (OR-Tools, guided local search)."""
    ids = [0] + [int(v) for v in nodes if v != 0]
    d = np.rint(site.dist[np.ix_(ids, ids)] * SCALE).astype(np.int64)
    r = _solve(d, 1, [10**12], [None] * len(ids), time_limit)[0]
    tour = [ids[i] for i in r]
    return tour_length(site, tour), tour


def cover_vrp(site: Site, nodes: list[int], battery: float, max_drones: int = 4,
              time_limit: float = 2.0) -> tuple[float, list[list[int]]] | None:
    """Visit all `nodes` with <= max_drones drones, each limited to `battery`; minimize total length.
    Inspection cost counts toward the battery. Returns None if infeasible."""
    ids = [0] + [int(v) for v in nodes if v != 0]
    full = site.dist + site.cost[None, :]
    arc = np.rint(full[np.ix_(ids, ids)] * SCALE).astype(np.int64)
    routes = _solve(arc, max_drones, [int(battery * SCALE)] * max_drones, [None] * len(ids), time_limit)
    if routes is None:
        return None
    routes = [[ids[i] for i in r] for r in routes if len(r) > 2]
    return sum(tour_length(site, r) for r in routes), routes


def tour_length(site: Site, tour: list[int]) -> float:
    return float(sum(site.dist[a, b] for a, b in zip(tour, tour[1:])))


def greedy_nn_tour(site: Site, nodes: list[int], with_paths: bool = True):
    """Nearest-neighbour closed tour through `nodes`; each leg expanded with A* into a waypoint path.
    Returns (length along the A* paths, node tour, full path)."""
    todo = {int(v) for v in nodes if v != 0}
    cur, tour = 0, [0]
    while todo:
        nxt = min(todo, key=lambda j: site.dist[cur, j])
        todo.remove(nxt)
        tour.append(nxt)
        cur = nxt
    tour.append(0)
    length, path = 0.0, [0]
    for a, b in zip(tour, tour[1:]):
        g, p = astar(site, a, b)
        length += g
        path += p[1:]
    return length, tour, path


def route_cost(site: Site, route: list[int]) -> float:
    """Battery used by a route: travel + inspection of each visited waypoint."""
    return tour_length(site, route) + float(sum(site.cost[j] for j in route[1:-1]))


def brute_force_orienteering(site: Site, cand: list[int], values: np.ndarray, budget: float) -> float:
    """Exact optimum over subsets of `cand` (<= ~12 nodes) by Held-Karp style DP over (mask, last).

    The score is sum of `values[j]` of the visited candidates; used to validate the OR-Tools solver.
    """
    m = len(cand)
    d0 = site.dist[0, cand]
    dd = site.dist[np.ix_(cand, cand)]
    c = site.cost[cand]
    INF = np.inf
    best_cost = np.full((1 << m, m), INF)
    for j in range(m):
        best_cost[1 << j, j] = d0[j] + c[j]
    best = 0.0
    for mask in range(1, 1 << m):
        for j in range(m):
            cj = best_cost[mask, j]
            if cj == INF:
                continue
            if cj + d0[j] <= budget + 1e-12:
                best = max(best, float(sum(values[cand[k]] for k in range(m) if mask >> k & 1)))
            for k in range(m):
                if not mask >> k & 1:
                    nc = cj + dd[j, k] + c[k]
                    if nc < best_cost[mask | 1 << k, k]:
                        best_cost[mask | 1 << k, k] = nc
    return best


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t

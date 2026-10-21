import numpy as np
import pytest

from gnn_inspect.data import make_site
from gnn_inspect.planner import astar, full_tour_cost, plan
from gnn_inspect.priority import centrality_table, combined_priority
from gnn_inspect.routing import (brute_force_orienteering, cover_vrp, greedy_nn_tour, orienteering,
                                 route_cost, tour_length, tsp)


@pytest.fixture(scope="module")
def site():
    return make_site(np.random.default_rng(1))


def test_centralities_shape_and_range(site):
    t = centrality_table(site)
    assert set(t) == {"degree", "betweenness", "closeness", "pagerank"}
    assert all(v.shape == (site.n,) and np.isfinite(v).all() for v in t.values())
    assert abs(t["pagerank"].sum() - 1) < 1e-6


def test_combined_priority_is_unit_range(site):
    p = combined_priority(site, site.severity)
    assert p.shape == (site.n,) and p.min() >= 0 and p.max() <= 1 + 1e-9


def test_orienteering_respects_battery(site):
    budget = 0.2 * full_tour_cost(site)
    found, routes = orienteering(site, site.severity, budget, time_limit=1)
    r = routes[0]
    assert r[0] == 0 and r[-1] == 0
    assert route_cost(site, r) <= budget + 1e-3
    assert len(set(r[1:-1])) == len(r) - 2  # no node inspected twice
    assert found == pytest.approx(sum(site.severity[j] for j in r[1:-1]))


def test_orienteering_not_worse_than_greedy(site):
    budget = 0.2 * full_tour_cost(site)
    g, _ = plan(site, site.severity, budget)
    o, _ = orienteering(site, site.severity, budget, time_limit=2)
    assert o >= g * 0.98  # the local-search solver should at least match the greedy heuristic


def test_orienteering_matches_brute_force_on_small_candidate_set(site):
    rng = np.random.default_rng(0)
    cand = [int(v) for v in rng.choice(np.arange(1, site.n), size=9, replace=False)]
    values = np.zeros(site.n)
    values[cand] = rng.uniform(0.2, 1, size=9)  # only candidates carry value
    budget = 0.08 * full_tour_cost(site)
    exact = brute_force_orienteering(site, cand, values, budget)
    found, _ = orienteering_with_values(site, values, budget)
    assert found == pytest.approx(exact, rel=1e-3)


def orienteering_with_values(site, values, budget):
    from gnn_inspect import routing
    old = site.severity
    site.severity = values
    try:
        return routing.orienteering(site, values, budget, time_limit=2)
    finally:
        site.severity = old


def test_tsp_beats_or_matches_nearest_neighbour(site):
    nodes = list(range(1, site.n, 4))
    nn_len, nn_tour, path = greedy_nn_tour(site, nodes)
    opt_len, tour = tsp(site, nodes, time_limit=1)
    assert sorted(tour[1:-1]) == sorted(nodes) and tour[0] == tour[-1] == 0
    assert opt_len <= nn_len + 1e-6
    assert path[0] == path[-1] == 0


def test_greedy_leg_lengths_match_astar(site):
    nodes = [5, 9, 20]
    length, tour, _ = greedy_nn_tour(site, nodes)
    assert length == pytest.approx(sum(astar(site, a, b)[0] for a, b in zip(tour, tour[1:])))
    assert length == pytest.approx(tour_length(site, tour))


def test_cover_vrp_routes_respect_battery_and_cover_all(site):
    nodes = list(range(1, site.n, 5))
    battery = 0.5 * full_tour_cost(site)
    out = cover_vrp(site, nodes, battery, max_drones=4, time_limit=2)
    assert out is not None
    total, routes = out
    assert sorted(j for r in routes for j in r[1:-1]) == sorted(nodes)
    assert all(route_cost(site, r) <= battery + 1e-3 for r in routes)
    assert total == pytest.approx(sum(tour_length(site, r) for r in routes))


def test_cover_vrp_infeasible_when_battery_too_small(site):
    far = int(np.argmax(site.dist[0]))
    assert cover_vrp(site, [far], battery=0.01, max_drones=2, time_limit=1) is None

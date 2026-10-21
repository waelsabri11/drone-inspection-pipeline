import networkx as nx
import numpy as np
import pytest
import torch

from gnn_inspect import baselines
from gnn_inspect.data import FEATURES, make_dataset, make_site, split
from gnn_inspect.experiment import main
from gnn_inspect.metrics import ndcg_at_k, spearman
from gnn_inspect.model import NodeScorer
from gnn_inspect.planner import astar, full_tour_cost, plan


@pytest.fixture(scope="module")
def site():
    return make_site(np.random.default_rng(0))


def test_site_is_valid(site):
    assert nx.is_connected(site.graph)
    assert site.severity[0] == 0 and site.cost[0] == 0  # depot
    assert np.all(site.prior[site.observed == 0] == 0)
    x = site.features()
    assert x.shape == (site.n, len(FEATURES)) and np.isfinite(x).all()


def test_dataset_is_deterministic():
    a, b = make_dataset(3, 2), make_dataset(3, 2)
    assert all(np.array_equal(s.severity, t.severity) for s, t in zip(a, b))


def test_split_has_no_graph_leakage():
    sites = list(range(10))
    tr, va, te = split(sites, 6, 2)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 10


def test_astar_matches_dijkstra(site):
    for t in (1, site.n // 2, site.n - 1):
        length, path = astar(site, 0, t)
        assert length == pytest.approx(site.dist[0, t])
        assert path[0] == 0 and path[-1] == t


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.3])
def test_plan_respects_battery(site, frac):
    budget = frac * full_tour_cost(site)
    found, route = plan(site, baselines.pagerank(site), budget)
    used = sum(site.dist[a, b] for a, b in zip(route, route[1:])) + site.cost[route].sum()
    assert used <= budget + 1e-9
    assert route[0] == route[-1] == 0
    assert found == pytest.approx(site.severity[route].sum())
    if frac == 0.0:
        assert found == 0


def test_metrics():
    truth = np.array([3.0, 2.0, 1.0, 0.0])
    assert ndcg_at_k(truth, truth, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(-truth, truth, k=2) < 0.5
    assert spearman(truth, truth) == pytest.approx(1.0)


@pytest.mark.parametrize("name", list(baselines.CENTRALITY) + list(baselines.PRIOR_BASED))
def test_baselines_shape(site, name):
    scores = {**baselines.CENTRALITY, **baselines.PRIOR_BASED}[name](site)
    assert scores.shape == (site.n,) and np.isfinite(scores).all()


@pytest.mark.parametrize("kind", ["gcn", "gat", "mlp"])
def test_model_forward(site, kind):
    d = site.to_data()
    out = NodeScorer(len(FEATURES), kind)(d.x, d.edge_index)
    assert out.shape == (site.n,) and torch.isfinite(out).all()


def test_end_to_end_quick(tmp_path):
    agg = main(["--quick", "--out", str(tmp_path)])
    assert (tmp_path / "results.md").exists()
    assert {"gcn", "pagerank", "betweenness", "degree", "closeness"} <= set(agg)

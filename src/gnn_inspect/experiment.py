"""End-to-end experiment: generate sites, train GNNs, compare with baselines.

    python -m gnn_inspect.experiment            # full run (5 seeds)
    python -m gnn_inspect.experiment --quick    # smoke run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import baselines
from .data import FEATURES, make_dataset, split
from .metrics import ndcg_at_k, spearman
from .model import NodeScorer, predict, train
from .planner import full_tour_cost, plan

BUDGETS = (0.10, 0.20, 0.30)
MODELS = ("gcn", "gat", "mlp")


def evaluate(site, scores) -> dict:
    out = {"ndcg@10": ndcg_at_k(scores, site.severity), "spearman": spearman(scores, site.severity)}
    ref = full_tour_cost(site)
    for b in BUDGETS:
        found, _ = plan(site, scores, b * ref)
        oracle, _ = plan(site, site.severity, b * ref)
        out[f"found@{int(b * 100)}%"] = 100 * found / oracle
    return out


def run_seed(seed: int, n_train: int, n_val: int, n_test: int, epochs: int) -> dict:
    torch.manual_seed(seed)
    sites = make_dataset(seed, n_train + n_val + n_test)
    tr, va, te = split(sites, n_train, n_val)
    tr_d, va_d = [s.to_data() for s in tr], [s.to_data() for s in va]

    scorers = {**baselines.CENTRALITY, **baselines.PRIOR_BASED}
    rng = np.random.default_rng(seed + 10_000)
    scorers["random"] = lambda s: rng.random(s.n)
    for kind in MODELS:
        model = train(NodeScorer(len(FEATURES), kind), tr_d, va_d, epochs=epochs, seed=seed)
        scorers[kind] = lambda s, m=model: predict(m, s.to_data())

    rows = {name: [evaluate(s, f(s)) for s in te] for name, f in scorers.items()}
    return {name: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for name, rs in rows.items()}


def aggregate(per_seed: list[dict]) -> dict:
    out = {}
    for name in per_seed[0]:
        out[name] = {
            k: (float(np.mean([s[name][k] for s in per_seed])), float(np.std([s[name][k] for s in per_seed])))
            for k in per_seed[0][name]
        }
    return out


LABELS = {
    "gcn": "GCN (ours)", "gat": "GAT (ours)", "mlp": "MLP, no graph (ablation)",
    "degree": "Degree centrality", "betweenness": "Betweenness centrality",
    "closeness": "Closeness centrality", "pagerank": "PageRank",
    "prior": "Survey prior only", "smoothed_prior": "Neighbour-smoothed prior",
    "personalized_pagerank": "Personalized PageRank (prior)", "random": "Random",
}


def to_markdown(agg: dict) -> str:
    cols = list(next(iter(agg.values())))
    head = "| Method | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    order = sorted(agg, key=lambda m: -agg[m]["found@20%"][0])
    body = ""
    for m in order:
        fmt = lambda k: f"{agg[m][k][0]:.3f} ± {agg[m][k][1]:.3f}" if k in ("ndcg@10", "spearman") \
            else f"{agg[m][k][0]:.1f} ± {agg[m][k][1]:.1f}"
        body += f"| {LABELS.get(m, m)} | " + " | ".join(fmt(k) for k in cols) + " |\n"
    return head + body


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="tiny config for smoke tests")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    cfg = dict(n_train=20, n_val=5, n_test=5, epochs=3) if a.quick else dict(n_train=400, n_val=100, n_test=200, epochs=150)
    seeds = range(2 if a.quick else a.seeds)

    t0 = time.time()
    per_seed = []
    for s in seeds:
        per_seed.append(run_seed(s, **cfg))
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    agg = aggregate(per_seed)

    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    (out / "results.json").write_text(json.dumps({"config": {**cfg, "seeds": list(seeds)}, "per_seed": per_seed, "mean_std": agg}, indent=2))
    table = to_markdown(agg)
    (out / "results.md").write_text(table)
    print(table)
    return agg


if __name__ == "__main__":
    main()

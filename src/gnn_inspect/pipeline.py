"""End-to-end: crack detector -> defect evidence on the site graph -> priorities -> battery-limited route.

Simulation assumption (stated in the README): at each surveyed waypoint the drone takes PATCHES photos.
Each photo shows a crack with probability severity/max_severity of that site; the photo itself is a real
patch drawn from the held-out crack-dataset pool (val pool for train/val sites, TEST pool for test sites),
and the evidence is the fraction of those photos the trained detector flags as cracked. The coupling between
site severity and photo content is simulated; the photos and the detector's errors on them are real.

    python -m gnn_inspect.pipeline --arch resnet18 --seeds 3
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import baselines
from .crack.predict import pool_predictions
from .data import FEATURES, make_dataset, split
from .experiment import BUDGETS
from .metrics import ndcg_at_k, spearman
from .model import NodeScorer, predict, train
from .planner import full_tour_cost, plan
from .priority import combined_priority
from .routing import orienteering

PATCHES = 4


def observe(site, pool, rng):
    """Replace the synthetic survey prior with detector evidence on the surveyed nodes."""
    y, pred = pool
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    p = site.severity / site.severity.max()
    ev = np.zeros(site.n)
    for i in range(site.n):
        if site.observed[i]:
            is_crack = rng.random(PATCHES) < p[i]
            idx = np.where(is_crack, rng.choice(pos, PATCHES), rng.choice(neg, PATCHES))
            ev[i] = pred[idx].mean()
    return dataclasses.replace(site, prior=ev * site.observed), ev


def run_seed(seed, arch, n_train, n_val, n_test, epochs, ort_time):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 777)
    pools = {s: pool_predictions(arch, s) for s in ("val", "test")}
    sites = make_dataset(seed, n_train + n_val + n_test)
    tr, va, te = split(sites, n_train, n_val)
    tr = [observe(s, pools["val"], rng)[0] for s in tr]
    va = [observe(s, pools["val"], rng)[0] for s in va]
    te = [observe(s, pools["test"], rng)[0] for s in te]
    model = train(NodeScorer(len(FEATURES), "gcn"), [s.to_data() for s in tr], [s.to_data() for s in va],
                  epochs=epochs, seed=seed)

    rankers = {
        "Random": lambda s: np.random.default_rng(seed).random(s.n),
        "Betweenness centrality": baselines.betweenness,
        "Closeness centrality": baselines.closeness,
        "Detector evidence only": baselines.prior,
        "Detector evidence + betweenness": lambda s: combined_priority(s, baselines.prior(s)),
        "GCN on detector evidence": lambda s: predict(model, s.to_data()),
    }
    rows = {k: [] for k in rankers}
    for s in te:
        ref = full_tour_cost(s)
        opt_found = {}
        for b in BUDGETS:
            opt_found[b] = orienteering(s, s.severity, b * ref, time_limit=ort_time)[0]
        for name, f in rankers.items():
            sc = np.asarray(f(s), float)
            r = {"ndcg@10": ndcg_at_k(sc, s.severity), "spearman": spearman(sc, s.severity)}
            for b in BUDGETS:
                gf, _ = plan(s, sc, b * ref)
                r[f"greedy@{int(b*100)}%"] = 100 * gf / opt_found[b]
            of, _ = orienteering(s, sc, 0.2 * ref, time_limit=ort_time)
            r["ortools@20%"] = 100 * of / opt_found[0.2]
            rows[name].append(r)
    return {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "deit_tiny"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--ort-time", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    cfg = dict(n_train=20, n_val=5, n_test=3, epochs=3, ort_time=0.2) if a.quick else \
        dict(n_train=400, n_val=100, n_test=a.n_test, epochs=150, ort_time=a.ort_time)
    seeds = range(1 if a.quick else a.seeds)
    t0, per = time.time(), []
    for s in seeds:
        per.append(run_seed(s, a.arch, **cfg))
        print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    agg = {k: {m: (float(np.mean([p[k][m] for p in per])), float(np.std([p[k][m] for p in per]))) for m in per[0][k]}
           for k in per[0]}
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    (out / "pipeline.json").write_text(json.dumps({"arch": a.arch, "config": {**cfg, "seeds": list(seeds)}, "per_seed": per, "mean_std": agg}, indent=2))
    cols = list(next(iter(agg.values())))
    md = "| Priority source | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    for k in sorted(agg, key=lambda k: -agg[k]["ortools@20%"][0]):
        md += f"| {k} | " + " | ".join(f"{agg[k][c][0]:.3f} ± {agg[k][c][1]:.3f}" if c in ("ndcg@10", "spearman") else f"{agg[k][c][0]:.1f} ± {agg[k][c][1]:.1f}" for c in cols) + " |\n"
    (out / "pipeline.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

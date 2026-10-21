"""Greedy + A* vs OR-Tools on route length and runtime.

The solvers are given the TRUE severities here, so the table isolates the routing algorithm
(ranking quality is studied separately in experiment.py / pipeline.py).

    python -m gnn_inspect.bench_planning --sites 30
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import make_dataset
from .planner import full_tour_cost, plan
from .routing import (brute_force_orienteering, cover_vrp, greedy_nn_tour, orienteering, timed, tsp)

SEED = 123  # benchmark sites are independent of the GNN experiment seeds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=30)
    ap.add_argument("--k", type=int, default=15, help="priority nodes for TSP / VRP")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    sites = make_dataset(SEED, a.sites)
    rows = {"orient": [], "tsp": [], "vrp": [], "gap": []}
    for si, s in enumerate(sites):
        ref = full_tour_cost(s)
        for b in (0.1, 0.2, 0.3):
            (gf, _), gt = timed(plan, s, s.severity, b * ref)
            (of, _), ot = timed(orienteering, s, s.severity, b * ref, time_limit=2.0)
            rows["orient"].append(dict(budget=b, greedy=gf, ortools=of, t_greedy=gt, t_ortools=ot))
        top = [int(j) for j in np.argsort(-s.severity)[: a.k] if j != 0]
        (nl, _, _), nt = timed(greedy_nn_tour, s, top)
        (ol, _), ot = timed(tsp, s, top, time_limit=1.0)
        rows["tsp"].append(dict(nn=nl, ortools=ol, t_nn=nt, t_ortools=ot))
        # cover the top-k with battery-limited drones: how many drones / how much total flight?
        batt = 0.25 * ref
        out = cover_vrp(s, top, batt, max_drones=4, time_limit=2.0)
        if out is not None:
            rows["vrp"].append(dict(total=out[0], drones=len(out[1]), single_tsp=ol))
        # exact check on small subproblems (10 random candidates, budget 8% of reference tour)
        rng = np.random.default_rng(si)
        cand = [int(v) for v in rng.choice(np.arange(1, s.n), size=10, replace=False)]
        vals = np.zeros(s.n)
        vals[cand] = s.severity[cand]
        old, s.severity = s.severity, vals
        ex = brute_force_orienteering(s, cand, vals, 0.08 * ref)
        of, _ = orienteering(s, vals, 0.08 * ref, time_limit=1.0)
        gf, _ = plan(s, vals, 0.08 * ref)
        s.severity = old
        if ex > 0:
            rows["gap"].append(dict(ortools=of / ex, greedy=gf / ex))
        print(f"site {si+1}/{a.sites}", flush=True)

    m = lambda rs, k: float(np.mean([r[k] for r in rs]))
    sd = lambda rs, k: float(np.std([r[k] for r in rs]))
    res = {"n_sites": a.sites, "k": a.k, "seed": SEED, "orienteering": {}, "tsp": {}, "cover_vrp": {}, "exact_check": {}}
    for b in (0.1, 0.2, 0.3):
        r = [x for x in rows["orient"] if x["budget"] == b]
        res["orienteering"][f"{int(b*100)}%"] = {
            "greedy_found": m(r, "greedy"), "ortools_found": m(r, "ortools"),
            "ortools_gain_pct": 100 * (m(r, "ortools") / m(r, "greedy") - 1),
            "greedy_ms": 1000 * m(r, "t_greedy"), "ortools_ms": 1000 * m(r, "t_ortools"),
            "ortools_ge_greedy_in": int(sum(x["ortools"] >= x["greedy"] - 1e-9 for x in r)), "of": len(r)}
    t = rows["tsp"]
    res["tsp"] = {"nn_len": m(t, "nn"), "ortools_len": m(t, "ortools"),
                  "ortools_shorter_pct": 100 * (1 - m(t, "ortools") / m(t, "nn")),
                  "nn_ms": 1000 * m(t, "t_nn"), "ortools_ms": 1000 * m(t, "t_ortools")}
    v = rows["vrp"]
    res["cover_vrp"] = {"feasible_sites": len(v), "of": a.sites, "mean_drones": m(v, "drones"),
                        "total_len": m(v, "total"), "single_unlimited_tsp_len": m(v, "single_tsp")}
    g = rows["gap"]
    res["exact_check"] = {"instances": len(g), "ortools_vs_exact": m(g, "ortools"), "ortools_min": float(min(x["ortools"] for x in g)),
                          "greedy_vs_exact": m(g, "greedy")}
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    (out / "planning_bench.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

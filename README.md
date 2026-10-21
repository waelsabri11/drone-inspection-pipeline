# GNN for drone inspection planning (PyTorch Geometric)

This project trains a graph neural network to rank the inspection waypoints a drone should visit on a structure. It then compares that ranking with the hand-crafted centrality rankings used before: degree, betweenness, closeness and PageRank. Each ranking drives the same battery-limited itinerary planner.

It extends my 2022 ENSTA Paris / U2IS internship on drone crack inspection. There, the site was modelled as a graph, inspection points were prioritised with centrality measures, and paths were planned with A* under battery limits. This repository is a new, independent 2026 project. It runs on **synthetic** sites (see Limits).

## Results

The test set has 200 unseen sites per seed. Values are the mean ± std over 5 seeds (0–4); for each seed, metrics are first averaged over its test sites. The table is sorted by defects found at the 20% budget.

- **found@B%**: true defect severity collected by the greedy itinerary when the battery equals B% of a full-site tour. It is shown as a % of the same planner fed the *true* severities. That reference is a greedy one, not an optimal tour.
- **ndcg@10** and **spearman** measure how well each method ranks waypoints against the hidden true severity.

| Method | ndcg@10 | spearman | found@10% | found@20% | found@30% |
|---|---:|---:|---:|---:|---:|
| GCN (ours) | 0.752 ± 0.005 | 0.619 ± 0.007 | 86.5 ± 1.7 | 90.0 ± 1.0 | 89.6 ± 1.4 |
| GAT (ours) | 0.728 ± 0.010 | 0.583 ± 0.006 | 85.8 ± 2.8 | 88.3 ± 1.0 | 88.7 ± 1.7 |
| Neighbour-smoothed prior | 0.725 ± 0.009 | 0.485 ± 0.005 | 78.7 ± 2.2 | 81.0 ± 2.2 | 82.7 ± 2.0 |
| Personalized PageRank (prior) | 0.668 ± 0.008 | 0.468 ± 0.004 | 78.7 ± 2.0 | 80.7 ± 1.1 | 82.8 ± 1.0 |
| MLP, no graph (ablation) | 0.656 ± 0.008 | 0.501 ± 0.011 | 81.6 ± 1.9 | 79.9 ± 2.7 | 80.9 ± 1.3 |
| Closeness centrality | 0.481 ± 0.012 | 0.434 ± 0.012 | 73.8 ± 2.4 | 75.5 ± 1.9 | 77.4 ± 1.0 |
| Survey prior only | 0.642 ± 0.007 | 0.338 ± 0.005 | 75.4 ± 2.7 | 74.1 ± 2.2 | 74.3 ± 0.9 |
| Betweenness centrality | 0.555 ± 0.006 | 0.590 ± 0.008 | 78.6 ± 1.4 | 73.2 ± 1.1 | 74.8 ± 1.2 |
| Degree centrality | 0.390 ± 0.005 | 0.257 ± 0.008 | 72.1 ± 2.9 | 69.6 ± 1.2 | 72.0 ± 1.7 |
| PageRank | 0.351 ± 0.009 | 0.137 ± 0.008 | 71.3 ± 1.9 | 68.7 ± 1.2 | 70.4 ± 1.0 |
| Random | 0.308 ± 0.010 | -0.002 ± 0.008 | 66.4 ± 3.7 | 61.5 ± 2.7 | 59.5 ± 1.1 |

What the table shows:

- **GCN against the best centrality baseline.** At the 20% budget, the GCN finds 90.0% of the reference defect severity. The best centrality ranking (closeness) finds 75.5%, a gain of 14.5 points. GCN leads at all three budgets and on NDCG@10. Its Spearman (0.619) is only slightly above betweenness (0.590).
- **A fairer comparison.** Centrality measures cannot see the survey prior, so part of that gap comes from information, not from learning. The fairer comparison is against non-learned methods that use the prior. Against the best of them, the GCN gains about 9 points at 20% (90.0 vs 81.0 for the neighbour-smoothed prior). GCN was ahead on every one of the 5 seeds (per-seed values in `results/results.json`).
- **Message passing matters.** An MLP with the same features and no edges reaches 79.9% at 20%, against 90.0% for the GCN.
- **GAT did not beat GCN** in this setup.

## Method

**Sites.** Each site has 50–80 waypoints in the unit square. Each waypoint is linked to its 4 nearest neighbours, and the Euclidean minimum spanning tree is added so the graph is connected. An edge's length is the Euclidean distance times U(1, 1.3), standing in for obstacles. Node 0 (the corner closest to the origin) is the depot.

**Hidden ground truth.** Each waypoint's defect severity is the sum of three parts:
- Cracks: 2–4 clusters, each a Gaussian bump over shortest-path distance.
- Load: 0.5 × normalised betweenness, so centrality carries real signal.
- Small noise.

**Observed node features (GNN input):**
- position (x, y);
- a noisy survey prior (normalised severity + N(0, 0.35)), available for about 60% of nodes and 0 elsewhere;
- a flag saying whether the survey covered the node;
- inspection cost;
- distance to the depot.

The true severity is never given as input.

**Models.** The models are built with PyTorch Geometric. Each has 3 GCNConv or GATConv layers (hidden size 64; GAT uses 4 heads) and a 2-layer MLP scoring head. They are trained by MSE regression on the true severity with Adam (lr 3e-3, batch 32, 150 epochs). The checkpoint with the best validation loss is kept. The data is split by **whole graphs**: 400 train / 100 val / 200 test sites per seed, so no test node is ever seen in training. Seeds are fixed.

**Planner.** The drone starts at the depot. It repeatedly flies to the unvisited waypoint with the best *score / (travel + inspection cost)*, as long as it still has enough battery to return. Travel uses shortest paths. An A* implementation with a Euclidean heuristic is included; it gives the same lengths as Dijkstra (tested), and the planner uses all-pairs Dijkstra for speed. Every ranking method goes through this exact planner. Budgets are 10/20/30% of a reference full-site tour (2 × MST length + all inspection costs).

**Baselines:**
- degree, betweenness, closeness (distance-weighted) and PageRank (1/length weights);
- the raw survey prior (missing values filled with the mean of the observed ones);
- the neighbour-smoothed prior: 0.5 × own prior + 0.5 × mean prior of the neighbours;
- personalized PageRank seeded by the prior;
- random.

## Limits

- **Synthetic data.** I designed the defect generator, so the results show that the method works when defects are spatially clustered and partly observed. They are not evidence about real structures. The 0.5 betweenness weight and 0.35 survey noise are arbitrary choices, and different choices would change the size of the gaps.
- **Unequal information.** The centrality baselines do not use the prior. The prior-based baselines are there to separate "uses more information" from "learns better".
- **Greedy reference.** The reference planner is a greedy one fed perfect scores, not an optimal orienteering solution. Values above 100% would therefore be possible; none occurred in the means.
- **Narrow tuning.** Only GCN and GAT were tried, with no hyperparameter search.

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/gnn-inspect --seeds 5      # full run, ~6 min on a laptop CPU, writes results/
.venv/bin/python -m pytest -q        # 19 tests, includes a quick end-to-end run
```

Layout: `src/gnn_inspect/` contains `data.py` (site generator), `baselines.py`, `planner.py` (A*, greedy itinerary), `model.py` (GNN), `metrics.py` and `experiment.py` (end-to-end run). Tests are in `tests/`.

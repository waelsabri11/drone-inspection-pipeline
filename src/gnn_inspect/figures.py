"""Figures for reports/: confusion matrices, compression trade-off, route comparison on one site."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import make_dataset
from .planner import full_tour_cost, plan
from .priority import centrality_table
from .routing import orienteering

ROOT = Path(__file__).resolve().parents[2]
RES, REP = ROOT / "results", ROOT / "reports"
NAMES = {"resnet18": "ResNet-18", "deit_tiny": "DeiT-Tiny (ViT)"}


def confusion_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, arch in zip(axes, NAMES):
        r = json.loads((RES / "crack" / f"{arch}_test.json").read_text())
        m = np.array(r["confusion"])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1], ["no crack", "crack"])
        ax.set_yticks([0, 1], ["no crack", "crack"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{NAMES[arch]}  F1={r['f1']:.4f}", fontsize=9)
    fig.suptitle(f"Held-out test set (n={m.sum()})", fontsize=10)
    fig.tight_layout()
    fig.savefig(REP / "confusion_matrices.png", dpi=150)


def compression_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, arch in zip(axes, NAMES):
        rows = json.loads((RES / "crack" / f"{arch}_compression.json").read_text())["rows"]
        rows = [r for r in rows if r["lat_bs1_ms"] == r["lat_bs1_ms"]]  # drop NaN-latency rows
        for r in rows:
            ax.scatter(r["lat_bs1_ms"], r["f1"], s=20 + 12 * r["size_mb"], alpha=0.7)
            ax.annotate(r["variant"].replace("magnitude prune", "prune").replace(" + 1 ep fine-tune", "+ft"),
                        (r["lat_bs1_ms"], r["f1"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("CPU latency, batch 1 (ms, 4 threads)")
        ax.set_ylabel("test F1")
        ax.set_title(f"{NAMES[arch]}  (marker area ~ file size)", fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "compression_tradeoff.png", dpi=150)


def route_figure():
    s = make_dataset(123, 1)[0]
    budget = 0.2 * full_tour_cost(s)
    gf, groute = plan(s, s.severity, budget)
    of, oroutes = orienteering(s, s.severity, budget, time_limit=2.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bc = centrality_table(s)["betweenness"]
    for ax, title in zip(axes, ["Betweenness centrality", f"Greedy: severity {gf:.2f}", f"OR-Tools: severity {of:.2f}"]):
        for u, v in s.graph.edges():
            ax.plot(*zip(s.pos[u], s.pos[v]), color="0.85", lw=0.7, zorder=1)
        ax.scatter(*s.pos.T, c=bc if "Between" in title else s.severity, s=28, cmap="viridis" if "Between" in title else "magma_r", zorder=2)
        ax.scatter(*s.pos[0], marker="s", c="red", s=50, zorder=4)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax, route in zip(axes[1:], [groute, oroutes[0]]):
        ax.plot(*s.pos[route].T, color="tab:blue", lw=1.6, zorder=3)
    fig.suptitle("One synthetic site, battery = 20% of a full tour (colour = true severity; red square = depot)", fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "route_comparison.png", dpi=150)


def cross_figure():
    d = json.loads((RES / "crack" / "cross_dataset.json").read_text())["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, metric, title in zip(axes, ["f1", "pr_auc"], ["F1 at threshold 0.5", "PR-AUC (threshold-free)"]):
        for i, ds in enumerate(["deepcrack", "crackforest"]):
            for j, (arch, aug) in enumerate([(a, g) for a in NAMES for g in ("light", "moderate")]):
                v = d[f"{arch}|{aug}|{ds}"][metric]
                x = i * 5 + j
                ax.bar(x, v["mean"], color=("tab:blue", "tab:cyan", "tab:red", "tab:orange")[j],
                       yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], capsize=2,
                       label=f"{NAMES[arch]} / {aug} aug" if i == 0 else None)
        ax.set_xticks([1.5, 6.5], ["DeepCrack", "CrackForest"])
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.4 if metric == "f1" else 0.9, 1.0)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Trained on METU only, no fine-tuning; mean of 3 seeds, 95% bootstrap CI over images", fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "cross_dataset.png", dpi=150)


def main():
    REP.mkdir(exist_ok=True)
    cross_figure()
    confusion_figure()
    compression_figure()
    route_figure()
    print("figures written to", REP)


if __name__ == "__main__":
    main()

"""Plot candidate speedup over the PyTorch-eager reference from benchmark.csv.

One bar per seq_len at the median across recorded runs, with each individual
run drawn as a dot so the run-to-run spread stays visible.

Usage: python scripts/plot_speedup.py [candidate_name]
Writes kda-attention-sm89/docs/speedup.png.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "kda-attention-sm89", "benchmark.csv")
OUT_PATH = os.path.join(ROOT, "kda-attention-sm89", "docs", "speedup.png")

SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES_1 = "#2a78d6"


def load(candidate):
    by_len = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["mode"] == candidate:
                by_len[int(row["seq_len"])].append(float(row["speedup_vs_pytorch"]))
    return dict(sorted(by_len.items()))


def main():
    candidate = sys.argv[1] if len(sys.argv) > 1 else "candidate_001_fused_cuda"
    data = load(candidate)
    if not data:
        sys.exit(f"no rows for {candidate} in {CSV_PATH}")

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = range(len(data))
    medians = [statistics.median(v) for v in data.values()]
    ax.bar(xs, medians, width=0.32, color=SERIES_1, zorder=2)
    for x, runs in zip(xs, data.values()):
        ax.scatter([x] * len(runs), runs, s=26, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for x, m in zip(xs, medians):
        ax.annotate(f"{m:.1f}x", (x, 0), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", color=SURFACE, fontsize=9, fontweight="bold", zorder=4)

    ax.set_xticks(list(xs), [f"seq_len {n}" for n in data], color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("speedup over PyTorch eager (x)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    n_runs = max(len(v) for v in data.values())
    ax.set_title(f"{candidate}: bar = median, dots = each of {n_runs} runs",
                 color=INK, fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=SURFACE)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

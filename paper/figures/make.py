"""Figures for the paper (vector PDF, column width, colourblind-safe palette).

    python3 figures/make.py [--src BENCH_RESULTS_DIR] [--raw RAW_LATENCY_DIR]

fig1_paired.pdf     one-row speedup of ours (fold + direct traversal) over each arm, tuned models
fig2_bound.pdf      one-row speedup vs the split-feasibility ratio (mechanism: Lemma 2 bound)
fig3_latency.pdf    absolute one-row latency per arm, median with 10-90 % spread
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "results"))
from numlib import two_level  # noqa: E402

DS = [("concrete", "concrete"), ("yacht", "yacht"), ("credit-g", "credit-g")]
ARMS = [("pristine", "upstream ngboost"), ("fold", "fold only"),
        ("direct", "direct traversal only"), ("merged", "merged steps (inexact)")]
COL = {"pristine": "#2a78d6", "fold": "#eb6834", "direct": "#1baf7a", "merged": "#eda100",
       "fold_direct": "#444441"}
MRK = {"concrete": "o", "yacht": "s", "credit-g": "^", "friedman1": "D"}
INK, MUTED = "#222221", "#77766f"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.family": "serif", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 4.1  # figure width in inches, placed at 0.75 of the NeurIPS text width


def fig1(raw, out):
    fig, ax = plt.subplots(figsize=(W, 2.15))
    y, ticks, labels = 0, [], []
    for a, alab in ARMS:
        for key, dlab in DS:
            ds = [np.log(raw[key][f"tuned/seed{i}"]["1"]["fold_direct"]) -
                  np.log(raw[key][f"tuned/seed{i}"]["1"][a]) for i in range(3)]
            pt, se, (lo, hi) = two_level(ds)
            ax.plot([np.exp(-hi), np.exp(-lo)], [y, y], color=COL[a], lw=2, solid_capstyle="round")
            ax.plot(np.exp(-pt), y, MRK[key], color=COL[a], ms=6, mec="white", mew=0.8)
            for d in ds:
                ax.plot(np.exp(-np.median(d)), y + 0.28, "|", color=MUTED, ms=4)
            ticks.append(y)
            labels.append(f"{alab}, {dlab}")
            y -= 1
        y -= 0.6
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.85, 4.8)
    ax.set_xticks([0.9, 1, 1.5, 2, 3, 4])
    ax.set_xticklabels(["0.9", "1", "1.5", "2", "3", "4"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel("one-row speedup of ours over the arm")
    ax.grid(axis="x", color="#e5e4df", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig2(src, out):
    fig, ax = plt.subplots(figsize=(W, 2.15))
    for key, dlab in DS:
        g = json.load(open(src / f"{key}.json"))["grid"]
        for r in g:
            p = r["per_param"][0]
            ratio = p["batch_rows"] / p["min_rows"]
            x = r["latency"]["1"]["paired"]["pristine"]["speedup_median"]
            deg = r["constant_share"] == 1.0
            ax.plot(ratio, x, MRK[key], ms=5, mec=COL["pristine"] if not deg else COL["fold"],
                    mfc="none" if not deg else COL["fold"], mew=1)
    d = json.load(open(src / "variants" / "low_signal_friedman1.json"))
    for v in d["levels"].values():
        for fit in v["fits"]:
            p = fit["per_param"][0]
            ax.plot(p["batch_rows"] / p["min_rows"], fit["latency"]["1"]["paired"]["pristine"]["speedup_median"],
                    MRK["friedman1"], ms=5, mfc=COL["fold"], mec=COL["fold"])
    ax.annotate("upstream stopped\nafter 1 round", (0.35, 7.4), (0.12, 60), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.text(0.9, 6e4, "no split feasible", ha="right", va="center", color=INK, fontsize=7)
    ax.text(1.1, 6e4, "splits feasible", ha="left", va="center", color=INK, fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 1.5e5)
    ax.set_xlabel(r"rows per round / $(2\,\times$ min_samples_leaf$)$ (log scale)")
    ax.set_ylabel("speedup vs upstream (x)")
    h = [plt.Line2D([], [], ls="", marker=MRK[k], color=INK, mfc="none", label=k) for k, _ in DS]
    h.append(plt.Line2D([], [], ls="", marker=MRK["friedman1"], color=INK, mfc="none",
                        label="Friedman 1, low signal (tuned)"))
    h.append(plt.Line2D([], [], ls="", marker="o", color=COL["fold"], label="all learners constant"))
    h.append(plt.Line2D([], [], ls="", marker="o", mfc="none", color=COL["pristine"], label="no learner constant"))
    ax.legend(handles=h, loc="center right", frameon=False, ncol=1, bbox_to_anchor=(1.0, 0.45))
    ax.grid(color="#e5e4df", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig3(raw, out):
    arms = ARMS[:3] + [("fold_direct", "ours (fold + direct)")]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), sharey=True)
    for ax, (key, dlab) in zip(axes, DS):
        for j, (a, alab) in enumerate(arms):
            t = 1e3 * np.concatenate([raw[key][f"tuned/seed{i}"]["1"][a] for i in range(3)])
            q10, q50, q90 = np.quantile(t, [0.1, 0.5, 0.9])
            ax.barh(j, q50, color=COL[a], height=0.6)
            ax.plot([q10, q90], [j, j], color=INK, lw=1)
        ax.set_title(dlab, color=INK)
        ax.set_xlabel("one-row latency (ms)")
        ax.grid(axis="x", color="#e5e4df", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels([lab for _, lab in arms])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent.parent / "bench" / "results"))
    ap.add_argument("--raw", default=str(HERE.parent.parent.parent / "_data" / "ngbfast_raw"))
    args = ap.parse_args()
    src, rawd = Path(args.src), Path(args.raw)
    raw = {k: json.load(gzip.open(rawd / f"{k}_latency_raw.json.gz")) for k, _ in DS}
    fig1(raw, HERE / "fig1_paired.pdf")
    fig2(src, HERE / "fig2_bound.pdf")
    fig3(raw, HERE / "fig3_latency.pdf")
    print("wrote fig1_paired.pdf fig2_bound.pdf fig3_latency.pdf")


if __name__ == "__main__":
    main()

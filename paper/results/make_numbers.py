"""Write results/numbers.tex: one macro per number the paper may cite, each with its source.

Usage (from the paper directory):
    python3 results/make_numbers.py [--src BENCH_RESULTS_DIR] [--raw RAW_LATENCY_DIR]

--src defaults to ../bench/results of the ngbfast repository (the paper's only evidence).
--raw (optional) holds <dataset>_latency_raw.json.gz, the per-call times of the tuned models; with it
the paper's own two-level bootstrap SE is added next to the benchmark's reported SE.
"""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from numlib import Macros, f, fint, fx, two_level  # noqa: E402
from sections import grid_and_variants, provenance, setup  # noqa: E402

DS = {"concrete": "Concrete", "yacht": "Yacht", "credit-g": "CreditG"}
ARMS = {"pristine": "Pristine", "fold": "Fold", "direct": "Direct", "merged": "Merged"}
BATCH = {"1": "One", "100": "Hundred"}


def load(p):
    return json.load(open(p))


def per_dataset(m, key, S, D, tag, raw_dir):
    s = S[key]
    src = f"summary.json [{key}]"
    m.section(f"{key}: data and tuned configuration")
    m.add(f"n{tag}N", fint(s["n"]), f"{src} n (rows in the data set)")
    m.add(f"n{tag}D", fint(D["d"]), f"{key}.json d (features)")
    m.add(f"n{tag}Task", s["task"], f"{src} task (distribution)")
    for k, v in s["picked"].items():
        name = "".join(w.capitalize() for w in k.split("_"))
        val = (f(v, 4) if k == "learning_rate" else f(v, 1)) if isinstance(v, float) else fint(v)
        m.add(f"n{tag}Pick{name}", val, f"{src} picked.{k}")
    m.add(f"n{tag}SearchTrials", fint(D["search"]["n_trials"]), f"{key}.json search.n_trials")
    m.add(f"n{tag}TunedShareMax", f(100 * max(s["tuned_constant_share"]), 0),
          f"{src} tuned_constant_share, max over seeds, percent")
    m.add(f"n{tag}DefaultShareMax", f(100 * max(s["default_constant_share"]), 0),
          f"{src} default_constant_share, max over seeds, percent")
    m.add(f"n{tag}Seeds", fint(len(s["tuned_constant_share"])), f"{src} len(tuned_constant_share)")
    nll = s["tuned_test_nll"]
    m.add(f"n{tag}TunedNLL", f(float(np.mean(nll))), f"{src} tuned_test_nll, mean over seeds (nats)")

    m.section(f"{key}: output equality over every fitted model")
    ex = s["exact_all_models"]
    m.add(f"n{tag}EqRows", fint(ex["fold_direct"]["n_rows"]), f"{src} exact_all_models.fold_direct.n_rows")
    m.add(f"n{tag}EqModels", fint(ex["fold_direct"]["n_models"]), f"{src} exact_all_models.fold_direct.n_models")
    for a in ("fold", "direct", "fold_direct"):
        assert ex[a]["bitwise_share"] == 1.0 and ex[a]["max_abs_diff"] == 0.0, (key, a)
    m.add(f"n{tag}MergedBitwise", f(100 * ex["merged"]["bitwise_share"], 0),
          f"{src} exact_all_models.merged.bitwise_share, percent of rows")
    e = ex["merged"]["max_abs_diff"]
    mant, exp = f"{e:.1e}".split("e")
    m.add(f"n{tag}MergedMaxDiff", f"{mant}\\times 10^{{{int(exp)}}}",
          f"{src} exact_all_models.merged.max_abs_diff (math mode)")

    m.section(f"{key}: tuned one-row paired latency, ours (fold + direct) vs arm")
    tp = s["tuned_paired"]
    raw = None
    if raw_dir is not None:
        rp = Path(raw_dir) / f"{key}_latency_raw.json.gz"
        raw = json.load(gzip.open(rp)) if rp.exists() else None
    for b, bt in BATCH.items():
        if b not in tp:
            continue
        for a, at in ARMS.items():
            p = tp[b][a]
            pre = f"n{tag}{bt}{at}"
            c = f"{src} tuned_paired.{b}.{a}"
            m.add(f"{pre}Log", f(p["mean"]), f"{c}.mean: mean over seeds of median log(t_ours/t_arm)")
            m.add(f"{pre}SE", f(p["se"]), f"{c}.se: benchmark SE, sd/sqrt(3) over seed medians")
            m.add(f"{pre}X", fx(p["speedup_at_mean"]), f"{c}.speedup_at_mean = exp(-mean)")
            if raw is not None and b == "1":
                ds = [np.log(raw[f"tuned/seed{i}"][b]["fold_direct"]) - np.log(raw[f"tuned/seed{i}"][b][a])
                      for i in range(3)]
                pt, se, (lo, hi) = two_level(ds)
                rc = f"raw {key}_latency_raw.json.gz tuned/seed*/{b}: paper's two-level bootstrap"
                m.add(f"{pre}BootSE", f(se), f"{rc} SE (seeds, then 10-rep blocks; 2,000 resamples)")
                m.add(f"{pre}BootLo", fx(math.exp(-hi)), f"{rc} 95% CI lower bound of the speedup")
                m.add(f"{pre}BootHi", fx(math.exp(-lo)), f"{rc} 95% CI upper bound of the speedup")
        for a, ms in tp[b]["p50_ms"].items():
            at = ARMS.get(a, "Ours")
            m.add(f"n{tag}{bt}{at}Ms", f(ms, 1), f"{src} tuned_paired.{b}.p50_ms.{a}, mean over seeds (ms)")
    full = "full" if "full" in tp else max((b for b in tp if b != "1"), key=int)
    nfull = max(int(b) for b in D["tuned"][0]["latency"])
    m.add(f"n{tag}FullRows", fint(nfull), f"{key}.json tuned[0].latency keys: full test set size")
    m.add(f"n{tag}FullPristineX", fx(tp[full]["pristine"]["speedup_at_mean"]),
          f"{src} tuned_paired.{full}.pristine.speedup_at_mean")
    slow = math.exp(tp["1"]["pristine"]["mean"] - tp["1"]["fold"]["mean"])
    m.add(f"n{tag}FoldSlowerPct", f(100 * (slow - 1), 0),
          f"{src} tuned_paired.1: t_fold/t_pristine = exp(d_pristine - d_fold), percent slower")
    m.add(f"n{tag}PlanOverFit", f(100 * float(np.median(s["plan_over_fit"])), 0),
          f"{src} plan_over_fit, median over tuned + default fits, percent of fit time")
    return ex["fold_direct"]["n_rows"], ex["fold_direct"]["n_models"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent.parent / "bench" / "results"))
    ap.add_argument("--raw", default=str(HERE.parent.parent.parent / "_data" / "ngbfast_raw"))
    ap.add_argument("--out", default=str(HERE / "numbers.tex"))
    args = ap.parse_args()
    src = Path(args.src)
    raw = Path(args.raw) if Path(args.raw).is_dir() else None
    S = load(src / "summary.json")
    m = Macros()
    rows = models = 0
    for key, tag in DS.items():
        r, n = per_dataset(m, key, S, load(src / f"{key}.json"), tag, raw)
        rows, models = rows + r, models + n
    m.section("totals over the three data sets")
    m.add("nEqRowsTotal", fint(rows), "sum of exact_all_models.fold_direct.n_rows")
    m.add("nEqModelsTotal", fint(models), "sum of exact_all_models.fold_direct.n_models")
    grid_and_variants(m, S, src)
    setup(m, Path(args.src))
    provenance(m, Path(args.src))
    m.add("nRawBoot", "yes" if raw else "no", f"raw latency dir used: {bool(raw)}")
    Path(args.out).write_text(m.text())
    print(f"wrote {len(m.names)} macros to {args.out}")


if __name__ == "__main__":
    main()

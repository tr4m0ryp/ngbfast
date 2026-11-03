"""Grid, variant, environment and external-line macros for make_numbers.py."""

import json
from pathlib import Path

import numpy as np

from numlib import f, fint, fx

DS = {"concrete": "Concrete", "yacht": "Yacht", "credit-g": "CreditG"}


def _grid(m, S, src):
    m.section("grid (1 seed, split 0): where constant learners appear")
    all_deg = []
    for key, tag in DS.items():
        g = S[key]["grid"]
        rows = json.load(open(src / f"{key}.json"))["grid"]
        c = f"summary.json [{key}] grid"
        deg = [x for x, r in zip(g, rows) if x["constant_share"] == 1.0]
        full = [x for x, r in zip(g, rows) if x["constant_share"] == 1.0 and r["n_rounds"] > 1]
        short = [x for x, r in zip(g, rows) if x["constant_share"] == 1.0 and r["n_rounds"] == 1]
        partial = [x for x in g if 0 < x["constant_share"] < 1]
        assert not partial, key
        for x, r in zip(g, rows):
            feas = all(p["feasible"] for p in r["per_param"])
            assert (x["constant_share"] == 1.0) == (not feas), (key, x)
        m.add(f"n{tag}GridPoints", fint(len(g)), f"{c}: number of grid points")
        m.add(f"n{tag}GridDegenerate", fint(len(deg)), f"{c}: points with constant_share == 1")
        best = min(x["test_nll"] for x in g)
        m.add(f"n{tag}GridBestNLL", f(best, 2), f"{c}: min test_nll over the grid (nats)")
        m.add(f"n{tag}GridDegNLL", f(float(np.mean([x["test_nll"] for x in deg])), 2),
              f"{c}: mean test_nll over degenerate points (nats)")
        xs = [x["ours_vs_pristine"] for x in full]
        m.add(f"n{tag}GridDegMinX", fx(min(xs)), f"{c}: min ours_vs_pristine at share 1, > 1 round")
        m.add(f"n{tag}GridDegMaxX", fx(max(xs)), f"{c}: max ours_vs_pristine at share 1, > 1 round")
        nd = [x["fold_vs_pristine"] for x in g if x["constant_share"] == 0.0]
        m.add(f"n{tag}GridSplitFoldX", f(float(np.median(nd)), 2),
              f"{c}: median fold_vs_pristine at share 0")
        nd = [x["ours_vs_pristine"] for x in g if x["constant_share"] == 0.0]
        m.add(f"n{tag}GridSplitOursMinX", fx(min(nd)), f"{c}: min ours_vs_pristine at share 0")
        m.add(f"n{tag}GridSplitOursMaxX", fx(max(nd)), f"{c}: max ours_vs_pristine at share 0")
        if short:
            xs = [x["ours_vs_pristine"] for x in short]
            m.add(f"n{tag}GridOneRoundN", fint(len(short)), f"{c}: share-1 fits stopped after 1 round")
            m.add(f"n{tag}GridOneRoundMinX", fx(min(xs)), f"{c}: min ours_vs_pristine, 1-round fits")
            m.add(f"n{tag}GridOneRoundMaxX", fx(max(xs)), f"{c}: max ours_vs_pristine, 1-round fits")
        all_deg.append(len(deg))
    m.add("nGridDegenerateTotal", fint(sum(all_deg)), "sum of grid points with constant_share == 1")
    m.add("nGridPointsTotal", fint(3 * 24), "3 data sets x 24 grid points")


def _low_signal(m, src):
    p = src / "variants" / "low_signal_friedman1.json"
    d = json.load(open(p))
    m.section("low-signal variant (Friedman 1, signal scaled by a)")
    m.add("nLowTrials", fint(d["n_trials"]), "low_signal_friedman1.json n_trials")
    xs, pri, ours, fold, share = [], [], [], [], []
    for lv, v in d["levels"].items():
        for fit in v["fits"]:
            lat = fit["latency"]["1"]
            xs.append(lat["paired"]["pristine"]["speedup_median"])
            pri.append(lat["arms"]["pristine"]["p50_ms"])
            ours.append(lat["arms"]["fold_direct"]["p50_ms"])
            fold.append(lat["arms"]["fold"]["p50_ms"])
            share.append(fit["constant_share"])
            assert all(e["bitwise_share"] == 1.0 for a, e in fit["equality"].items() if a != "merged")
        pk = v["picked"]
    c = "low_signal_friedman1.json levels.*.fits[*].latency.1"
    m.add("nLowFits", fint(len(xs)), "low_signal_friedman1.json: fits over both levels x 3 seeds")
    m.add("nLowShareMin", f(100 * min(share), 0), "low_signal_friedman1.json fits constant_share min, %")
    m.add("nLowMinX", fx(min(xs)), f"{c}.paired.pristine.speedup_median, min over fits")
    m.add("nLowMaxX", fx(max(xs)), f"{c}.paired.pristine.speedup_median, max over fits")
    m.add("nLowPristineMinMs", fint(min(pri)), f"{c}.arms.pristine.p50_ms min (ms)")
    m.add("nLowPristineMaxMs", fint(max(pri)), f"{c}.arms.pristine.p50_ms max (ms)")
    m.add("nLowOursMs", f(float(np.median(ours)), 2), f"{c}.arms.fold_direct.p50_ms median (ms)")
    m.add("nLowFoldMs", f(float(np.median(fold)), 2), f"{c}.arms.fold.p50_ms median (ms)")
    m.add("nLowPickLeaf", fint(pk["min_samples_leaf"]), "low_signal_friedman1.json picked.min_samples_leaf")
    m.add("nLowPickMinibatch", f(pk["minibatch_frac"], 1), "low_signal_friedman1.json picked.minibatch_frac")
    se = [fit["latency"]["1"]["paired"]["pristine"]["se"] for v in d["levels"].values() for fit in v["fits"]]
    m.add("nLowSEMin", f(min(se), 2), f"{c}.paired.pristine.se min (block bootstrap, log ratio)")
    m.add("nLowSEMax", f(max(se), 2), f"{c}.paired.pristine.se max")


def _one_se_small(m, src):
    d = json.load(open(src / "variants" / "one_se.json"))
    m.section("one-SE selection variant: held-out NLL of the one-SE pick minus the best pick")
    for key, tag in (("credit-g_n100", "CG"), ("friedman1_a0.0", "FA"), ("friedman1_a0.1", "FB")):
        p = d[key]["paired_test_nll"]
        c = f"one_se.json [{key}] paired_test_nll"
        nd = 3 if tag == "CG" else 5
        m.add(f"nOneSE{tag}Diff", f(p["nll_one_se_minus_best"], nd, sign=True), f"{c}.nll_one_se_minus_best (nats)")
        m.add(f"nOneSE{tag}SE", f(p["se"], nd), f"{c}.se, per-example SD/sqrt(n)")
        m.add(f"nOneSE{tag}N", fint(p["n"]), f"{c}.n test rows")
        sh = [x["one_se_constant_share"] for x in d[key]["fits"]]
        m.add(f"nOneSE{tag}Share", f(100 * min(sh), 0), f"one_se.json [{key}] fits one_se_constant_share min, %")
    ms = [x["one_se_vs_best_1row_ms"] for x in d["credit-g_n100"]["fits"]]
    m.add("nOneSECGOneSEMs", f(float(np.mean([a for a, b in ms])), 2), "one_se.json credit-g_n100 one_se_vs_best_1row_ms[0] mean")
    m.add("nOneSECGBestMs", f(float(np.mean([b for a, b in ms])), 2), "one_se.json credit-g_n100 one_se_vs_best_1row_ms[1] mean")
    d = json.load(open(src / "variants" / "small_n_credit-g.json"))
    v = d["sizes"]["100"]
    m.section("small-n variant (credit-g, 100 training rows)")
    c = "small_n_credit-g.json sizes.100"
    m.add("nSmallPickLeaf", fint(v["picked"]["min_samples_leaf"]), f"{c}.picked.min_samples_leaf")
    m.add("nSmallPickDepth", fint(v["picked"]["max_depth"]), f"{c}.picked.max_depth")
    m.add("nSmallPickRounds", fint(v["picked"]["n_estimators"]), f"{c}.picked.n_estimators")
    m.add("nSmallShareMax", f(100 * max(x["constant_share"] for x in v["fits"]), 0), f"{c}.fits constant_share max, %")
    xs = [x["latency"]["1"]["paired"]["pristine"]["speedup_median"] for x in v["fits"]]
    m.add("nSmallX", fx(float(np.mean(xs))), f"{c}.fits latency.1.paired.pristine.speedup_median, mean")


def grid_and_variants(m, S, src):
    _grid(m, S, src)
    _low_signal(m, src)
    _one_se_small(m, src)
    m.section("environment")
    c = json.load(open(src / "concrete.json"))
    for k, v in c["versions"].items():
        m.add(f"nVer{k.capitalize()}", v, f"concrete.json versions.{k}")
    loads = [x for key in DS for pair in S[key]["load"] for x in pair]
    m.add("nLoadMin", fint(min(loads)), "summary.json [*] load (1-min load average), min")
    m.add("nLoadMax", fint(max(loads)), "summary.json [*] load, max")
    m.add("nWallTuned", fint(sum(S[k]["wall_s"] for k in DS)), "summary.json [*] wall_s summed (s)")
    m.section("one external deployment (the only permitted sentence)")
    line = (src / "PRIVATE_LINE.txt").read_text().strip()
    m.add("nExternalLine", line.replace("%", "\\%"), "PRIVATE_LINE.txt, verbatim; cite at most once")


def setup(m, src):
    """Design constants of the benchmark, read from the result files (Setup section macros)."""
    import re

    from numlib.boot import BLOCK
    m.section("setup: design constants of the benchmark")
    c = json.load(open(src / "concrete.json"))
    m.add("nDataSets", fint(len(DS)), "number of public data sets in summary.json")
    m.add("nCVFolds", fint(c["search"]["n_folds"]), "concrete.json search.n_folds")
    lat = c["tuned"][0]["latency"]
    m.add("nRepsOne", fint(lat["1"]["arms"]["pristine"]["n"]), "concrete.json tuned[0].latency.1 arms n")
    m.add("nRepsHundred", fint(lat["100"]["arms"]["pristine"]["n"]), "concrete.json tuned[0].latency.100 arms n")
    m.add("nHundredRows", fint(100), "concrete.json tuned[0].latency key '100' (batch rows)")
    g = c["grid"]
    m.add("nGridRepsOne", fint(g[0]["latency"]["1"]["arms"]["pristine"]["n"]), "concrete.json grid[0].latency.1 arms n")
    for k, name in (("min_samples_leaf", "Leaf"), ("minibatch_frac", "Minibatch"), ("max_depth", "Depth")):
        vals = sorted(set(r["config"][k] for r in g))
        m.add(f"nGrid{name}Values", ", ".join(fint(v) if isinstance(v, int) else f(v, 1) for v in vals),
              f"concrete.json grid config.{k} values")
    m.add("nGridRounds", fint(g[0]["config"]["n_estimators"]), "concrete.json grid config.n_estimators")
    m.add("nGridLR", f(g[0]["config"]["learning_rate"], 2), "concrete.json grid config.learning_rate")
    dc = c["default"][0]["config"]
    m.add("nDefaultDepth", fint(dc["max_depth"]), "concrete.json default[0].config.max_depth")
    m.add("nDefaultLeaf", fint(dc["min_samples_leaf"]), "concrete.json default[0].config.min_samples_leaf")
    m.add("nBlockLen", fint(BLOCK), "numlib/boot.py BLOCK (moving-block length, repetitions)")
    m.add("nBootResamples", fint(2000), "numlib/boot.py two_level resamples default")
    low = json.load(open(src / "variants" / "low_signal_friedman1.json"))
    fit = next(iter(low["levels"].values()))["fits"][0]
    m.add("nLowReps", fint(fit["latency"]["1"]["arms"]["pristine"]["n"]), "low_signal_friedman1.json fits[0].latency.1 arms n")
    p = fit["per_param"][0]
    m.add("nLowBatchRows", fint(p["batch_rows"]), "low_signal_friedman1.json fits[0].per_param[0].batch_rows")
    m.add("nLowMinRows", fint(p["min_rows"]), "low_signal_friedman1.json fits[0].per_param[0].min_rows")
    m.add("nLowLevels", fint(len(low["levels"])), "low_signal_friedman1.json number of signal levels")
    m.add("nArmsTried", fint(4), "arms besides upstream: fold, direct, fold_direct, merged (summary.json tuned_paired)")
    nvar = len(list((src / "variants").glob("*.json")))
    m.add("nVariantsTried", fint(nvar), "bench/results/variants/*.json count")
    cores = re.search(r"shared (\d+)-core host", (src / "SCOPE.md").read_text()).group(1)
    m.add("nCores", cores, "SCOPE.md 'shared N-core host'")


def provenance(m, src):
    """Training sizes of the variants, the repository commit and the upstream patch size."""
    import subprocess
    m.section("variants: training rows; provenance")
    low = json.load(open(src / "variants" / "low_signal_friedman1.json"))
    pk = low["picked"] if "picked" in low else next(iter(low["levels"].values()))["picked"]
    p = next(iter(low["levels"].values()))["fits"][0]["per_param"][0]
    m.add("nLowTrainRows", fint(round(p["batch_rows"] / pk["minibatch_frac"])),
          "low_signal_friedman1.json per_param[0].batch_rows / picked.minibatch_frac")
    sm = json.load(open(src / "variants" / "small_n_credit-g.json"))
    m.add("nSmallTrainRows", fint(int(next(iter(sm["sizes"])))), "small_n_credit-g.json sizes key (training rows)")
    repo = src.parent.parent
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    m.add("nCommit", sha, "git rev-parse --short HEAD of the ngbfast repository")
    patch = repo / "patches" / "ngboost-check-input-once.patch"
    m.add("nPatchLines", fint(len(patch.read_text().splitlines())), "wc -l patches/ngboost-check-input-once.patch")
    t = json.load(open(repo / "patches" / "ngboost-check-input-once.timings.json"))
    m.add("nPatchBase", t["base_commit"], "ngboost-check-input-once.timings.json base_commit (ngboost master)")
    dflt = [r["speedup"] for r in t["rows"] if r["default"]]
    m.add("nPatchSpeedLo", f"{min(dflt):.1f}", "timings.json min speedup over default-learner rows")
    m.add("nPatchSpeedHi", f"{max(dflt):.1f}", "timings.json max speedup over default-learner rows")
    const = [r["speedup"] for r in t["rows"] if not r["default"]]
    m.add("nPatchSpeedConst", fint(round(const[0])), "timings.json speedup, all trees constant")

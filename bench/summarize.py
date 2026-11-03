"""Collect bench/results/<dataset>.json into bench/results/summary.json and print the headline table.

``python -m bench.summarize``. Across seeds the per-seed median paired log difference
(ours - arm) is averaged with SE = sd / sqrt(n_seeds); within a seed the block-bootstrap SE is kept.
"""

import json

import numpy as np

from .run.driver import RESULTS
from .timing import across_seeds

DATASETS = ["concrete", "yacht", "credit-g", "energy"]


def _exact(fits):
    out = {}
    for arm in ("fold", "direct", "fold_direct", "merged", "floor"):
        rows = [f["equality"][arm] for f in fits if arm in f["equality"]]
        if rows:
            n = sum(r["n_rows"] for r in rows)
            out[arm] = {"bitwise_share": sum(r["bitwise_share"] * r["n_rows"] for r in rows) / n,
                        "max_abs_diff": max(r["max_abs_diff"] for r in rows), "n_rows": n,
                        "n_models": len(rows)}
    return out


def _paired(fits):
    out = {}
    for batch in fits[0]["latency"]:
        out[batch] = {}
        for arm in fits[0]["latency"][batch]["paired"]:
            meds = [f["latency"][batch]["paired"][arm]["median_log_diff"] for f in fits
                    if arm in f["latency"].get(batch, {}).get("paired", {})]
            if meds:
                out[batch][arm] = across_seeds(meds)
        out[batch]["p50_ms"] = {arm: float(np.mean([f["latency"][batch]["arms"][arm]["p50_ms"]
                                                    for f in fits if arm in f["latency"][batch]["arms"]]))
                                for arm in fits[0]["latency"][batch]["arms"]}
    return out


def _grid(rows):
    out = []
    for g in rows:
        lat = g["latency"]["1"]
        out.append({**{k: g["config"][k] for k in ("min_samples_leaf", "minibatch_frac", "max_depth")},
                    "constant_share": g["constant_share"], "per_param": g["constant_per_param"],
                    "speedup_fold": lat["paired"]["fold"]["speedup_median"] if "fold" in lat["paired"] else None,
                    "fold_vs_pristine": float(np.exp(lat["paired"]["pristine"]["median_log_diff"]
                                                     - lat["paired"]["fold"]["median_log_diff"])),
                    "ours_vs_pristine": lat["paired"]["pristine"]["speedup_median"],
                    "ours_vs_pristine_se_log": lat["paired"]["pristine"]["se"],
                    "test_nll": g["test_nll"]})
    return out


def summarize(name):
    d = json.loads((RESULTS / f"{name}.json").read_text())
    shares = [f["constant_share"] for f in d["tuned"]]
    return {
        "dataset": name, "n": d["n"], "task": d["task"], "picked": d["search"]["picked"],
        "search_failed_trials": sum(t.get("failed", False) for t in d["search"]["trials"]),
        "tuned_constant_share": shares,
        "tuned_constant_per_param": [f["constant_per_param"] for f in d["tuned"]],
        "tuned_test_nll": [f["test_nll"] for f in d["tuned"]],
        "default_constant_share": [f["constant_share"] for f in d["default"]],
        "plan_over_fit": [f["plan_over_fit"] for f in d["tuned"] + d["default"]],
        "plan_build_s": [f["plan_build_s"] for f in d["tuned"] + d["default"]],
        "exact_all_models": _exact(d["grid"] + d["tuned"] + d["default"]),
        "tuned_paired": _paired(d["tuned"]),
        "default_paired": _paired(d["default"]),
        "grid": _grid(d["grid"]),
        "load": [d["load_start"], d["load_end"]], "wall_s": d["wall_s"],
    }


def main():
    out = {}
    for name in DATASETS:
        if (RESULTS / f"{name}.json").exists():
            out[name] = s = summarize(name)
            p = s["tuned_paired"]["1"]
            print(f"{name}: picked {s['picked']} share {np.round(s['tuned_constant_share'], 3)}")
            for arm in ("pristine", "fold", "direct", "merged"):
                a = p.get(arm)
                if a:
                    print(f"  1-row ours vs {arm}: {a['speedup_at_mean']:.2f}x "
                          f"(log {a['mean']:.3f} +- {a['se']:.3f}, n={a['n']})")
            print(f"  exact: { {k: v['bitwise_share'] for k, v in s['exact_all_models'].items()} }")
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

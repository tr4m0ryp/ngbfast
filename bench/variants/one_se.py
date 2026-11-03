"""One-standard-error selection: when the most regularised configuration ties the best, pick it.

What would have to be true for folding to matter on a tuned model: a configuration with constant
learners must be as accurate as the best one. Small, low-signal data makes that plausible (the best
model is close to a fixed distribution). The one-SE rule (Breiman et al. 1984) picks, among trials
whose CV NLL is within one standard error of the best, the most regularised one (largest leaf size,
then the smallest learning_rate x rounds). The SE is that of the best trial's out-of-fold per-example
NLL. Both picks are refit on S subsamples and compared on held-out rows (paired per example) and in
1-row latency. Run after ``small_n`` and ``low_signal``: ``python -m bench.variants.one_se``.
"""

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

import ngbfast  # noqa: E402

from ..run.checks import measure  # noqa: E402
from ..run.driver import RESULTS  # noqa: E402
from ..tuning import fit_model, nll  # noqa: E402
from .low_signal import make_task  # noqa: E402
from .small_n import LOADERS, subsample  # noqa: E402

KEYS = ("max_depth", "min_samples_leaf", "minibatch_frac", "learning_rate")


def _config(trial):
    return {**{k: trial[k] for k in KEYS}, "n_estimators": trial["best_rounds"]}


def _oof_nll(kind, cfg, X, y):
    cv = (StratifiedKFold if kind == "bernoulli" else KFold)(3, shuffle=True, random_state=0)
    out = np.empty(y.size)
    for tr, va in cv.split(X, y):
        model, _ = fit_model(kind, cfg, 0, X[tr], y[tr])
        ngbfast.fold(model)
        out[va] = nll(kind, ngbfast.pred_param(model, X[va]), y[va])
    return out


def one_se_pick(kind, trials, X0, y0):
    ok = [t for t in trials if np.isfinite(t["cv_nll"])]
    best = min(ok, key=lambda t: t["cv_nll"])
    oof = _oof_nll(kind, _config(best), X0, y0)
    se = float(oof.std(ddof=1) / np.sqrt(oof.size))
    near = [t for t in ok if t["cv_nll"] <= best["cv_nll"] + se]
    pick = max(near, key=lambda t: (t["min_samples_leaf"], -t["learning_rate"] * t["best_rounds"]))
    return _config(best), _config(pick), se, len(near)


def compare(kind, best, pick, sets, reps=60):
    rows, diffs = [], []
    for s, (Xtr, ytr, Xte, yte) in enumerate(sets):
        per = {}
        for name, cfg in (("best", best), ("one_se", pick)):
            model, fit_s = fit_model(kind, cfg, s, Xtr, ytr)
            info, _ = measure(model, kind, Xte, yte, fit_s, [(1, reps)], s, len(ytr),
                              arm_subset=["pristine", "fold", "fold_direct"])
            per[name] = (model, info)
        nll_b = nll(kind, per["best"][0].pred_param(Xte), yte)
        nll_o = nll(kind, per["one_se"][0].pred_param(Xte), yte)
        diffs.append(nll_o - nll_b)
        rows.append({"seed": s, **{f"{k}_{f}": v[1][f] for k, v in per.items()
                                   for f in ("constant_share", "test_nll", "n_rounds")},
                     "one_se_ours_vs_pristine_1row": per["one_se"][1]["latency"]["1"]["paired"]["pristine"],
                     "best_ours_vs_pristine_1row": per["best"][1]["latency"]["1"]["paired"]["pristine"],
                     "one_se_vs_best_1row_ms": [per["one_se"][1]["latency"]["1"]["arms"]["fold_direct"]["p50_ms"],
                                                per["best"][1]["latency"]["1"]["arms"]["fold_direct"]["p50_ms"]]})
    d = np.mean(diffs, axis=0) if len({x.size for x in diffs}) == 1 else np.concatenate(diffs)
    return rows, {"nll_one_se_minus_best": float(d.mean()),
                  "se": float(d.std(ddof=1) / np.sqrt(d.size)), "n": int(d.size),
                  "se_method": "per-example difference (averaged over seeds when the test rows are "
                               "shared), test rows independent, SE = sd / sqrt(n)"}


def main():
    out = {}
    path = RESULTS / "variants" / "small_n_credit-g.json"
    if path.exists():
        d = json.loads(path.read_text())
        task = LOADERS["credit-g"]()
        for n, block in d["sizes"].items():
            tr0, _ = subsample(task, int(n), 0)
            best, pick, se, k = one_se_pick(task.kind, block["trials"], task.X[tr0], task.y[tr0])
            sets = [(task.X[tr], task.y[tr], task.X[te], task.y[te])
                    for tr, te in (subsample(task, int(n), s) for s in range(d["seeds"]))]
            rows, paired = compare(task.kind, best, pick, sets)
            out[f"credit-g_n{n}"] = {"best": best, "one_se": pick, "cv_se": se, "n_within": k,
                                     "fits": rows, "paired_test_nll": paired}
            print(f"[one_se credit-g n={n}] {pick} {paired}", flush=True)
    path = RESULTS / "variants" / "low_signal_friedman1.json"
    if path.exists():
        d = json.loads(path.read_text())
        for a, block in d["levels"].items():
            t0 = make_task(float(a), 0)
            tr0 = t0.splits[0][0]
            best, pick, se, k = one_se_pick("normal", block["trials"], t0.X[tr0], t0.y[tr0])
            sets = []
            for s in range(d["seeds"]):
                t = make_task(float(a), s)
                tr, te = t.splits[0]
                sets.append((t.X[tr], t.y[tr], t.X[te], t.y[te]))
            rows, paired = compare("normal", best, pick, sets)
            out[f"friedman1_a{a}"] = {"best": best, "one_se": pick, "cv_se": se, "n_within": k,
                                      "fits": rows, "paired_test_nll": paired}
            print(f"[one_se friedman1 a={a}] {pick} {paired}", flush=True)
    (RESULTS / "variants" / "one_se.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

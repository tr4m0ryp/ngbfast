"""Per-model measurements: bitwise equality of every arm against the upstream loop, and latency."""

import time

import numpy as np

import ngbfast

from ..arms import OURS, make_arms
from ..timing import interleaved, paired_log_ratio, summary
from ..tuning import nll


def equality(arms, X):
    ref = arms["pristine"](X)
    out = {}
    for name, f in arms.items():
        if name == "pristine":
            continue
        theta = f(X)
        same = theta.view(np.uint64) == ref.view(np.uint64)
        out[name] = {"bitwise_share": float(same.all(axis=1).mean()),
                     "max_abs_diff": float(np.abs(theta - ref).max()), "n_rows": int(X.shape[0])}
    return ref, out


def latency(arms, X, batches, seed):
    """{batch: {"arms": {arm: summary}, "paired": {arm: ours - arm}}, plus raw per-call seconds}."""
    result, raw = {}, {}
    for batch, reps in batches:
        times = interleaved(arms, X, batch, reps, seed=seed)
        key = str(min(batch, X.shape[0]))
        raw[key] = {k: v.tolist() for k, v in times.items()}
        result[key] = {
            "arms": {k: summary(v) for k, v in times.items()},
            "paired": {k: paired_log_ratio(times[OURS], v, seed=seed)
                       for k, v in times.items() if k != OURS},
        }
    return result, raw


def measure(model, kind, X_test, y_test, fit_s, batches, seed, n_train, arm_subset=None):
    t0 = time.perf_counter()
    arms, plan, info = make_arms(model)
    info["plan_build_s"] = time.perf_counter() - t0
    info["fit_s"] = fit_s
    info["plan_over_fit"] = info["plan_build_s"] / fit_s
    diag = ngbfast.diagnose(model, n_train=n_train)
    info["per_param"] = diag["per_param"]
    info["degenerate"] = bool(diag["degenerate"])
    ref, info["equality"] = equality(arms, X_test)
    info["test_nll"] = float(nll(kind, ref, y_test).mean())
    timed = arms if arm_subset is None else {k: arms[k] for k in arm_subset if k in arms}
    info["latency"], raw = latency(timed, X_test, batches, seed)
    return info, raw

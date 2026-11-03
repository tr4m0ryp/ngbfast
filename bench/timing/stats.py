"""Paired statistics for latency arms.

The unit is one repetition. The paired difference is ``d_r = log t_ours,r - log t_arm,r`` (negative:
ours is faster). Consecutive repetitions share machine load, so they are dependent: the standard
error and the 95 % interval of the median of d use a moving-block bootstrap (blocks of 10
repetitions, 2,000 resamples). Across seeds (independent fits) the per-seed medians are combined
with a plain mean and its standard error.
"""

import numpy as np

BLOCK = 10
RESAMPLES = 2000


def summary(seconds):
    ms = 1e3 * np.asarray(seconds)
    return {"p50_ms": float(np.median(ms)), "p90_ms": float(np.quantile(ms, 0.9)),
            "mean_ms": float(ms.mean()), "n": int(ms.size)}


def _block_bootstrap_median(d, rng):
    n = d.size
    block = min(BLOCK, n)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    meds = np.empty(RESAMPLES)
    for b in range(RESAMPLES):
        pick = rng.choice(starts, n_blocks)
        idx = (pick[:, None] + np.arange(block)[None, :]).ravel()[:n]
        meds[b] = np.median(d[idx])
    return meds


def paired_log_ratio(t_ours, t_arm, seed=0):
    """Median of log(t_ours / t_arm) per repetition, its block-bootstrap SE and 95 % interval."""
    d = np.log(np.asarray(t_ours)) - np.log(np.asarray(t_arm))
    meds = _block_bootstrap_median(d, np.random.default_rng(seed))
    ratio = np.asarray(t_arm) / np.asarray(t_ours)
    return {
        "median_log_diff": float(np.median(d)),
        "se": float(meds.std(ddof=1)),
        "ci95": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))],
        "speedup_median": float(np.exp(-np.median(d))),
        "speedup_q05_per_call": float(np.quantile(ratio, 0.05)),
        "n": int(d.size),
        "se_method": f"moving-block bootstrap over repetitions, block {BLOCK}, {RESAMPLES} resamples",
    }


def across_seeds(values):
    v = np.asarray(values, dtype=float)
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "n": int(v.size),
            "speedup_at_mean": float(np.exp(-v.mean())),
            "se_method": "independent seeds (separate split and fit), SE = sd / sqrt(n)"}

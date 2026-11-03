"""Two-level bootstrap of the across-seed mean of per-seed median paired log ratios.

Level 1 resamples seeds (independent splits and fits) with replacement; level 2 resamples moving
blocks of 10 consecutive repetitions within each drawn seed, because consecutive repetitions share
machine load. This is the paper's own SE; the benchmark's reported SE is sd / sqrt(n_seeds) of the
per-seed medians and ignores the within-seed noise.
"""

import numpy as np

BLOCK = 10


def _block_idx(n, rng):
    block = min(BLOCK, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def two_level(per_seed_d, resamples=2000, seed=0):
    """per_seed_d: list of 1-D arrays d_r = log t_ours - log t_arm. Returns mean, se, (lo, hi)."""
    rng = np.random.default_rng(seed)
    ds = [np.asarray(d, dtype=float) for d in per_seed_d]
    point = float(np.mean([np.median(d) for d in ds]))
    out = np.empty(resamples)
    for b in range(resamples):
        pick = rng.integers(0, len(ds), len(ds))
        out[b] = np.mean([np.median(ds[s][_block_idx(ds[s].size, rng)]) for s in pick])
    lo, hi = np.quantile(out, [0.025, 0.975])
    return point, float(out.std(ddof=1)), (float(lo), float(hi))

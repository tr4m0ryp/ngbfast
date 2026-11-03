"""Run every arm once per repetition on the same rows, in a rotating order (ABC, BCA, CAB, ...).

Rotating the order means no arm always runs first (cold caches) or last; arms in one repetition see
the same machine load, so per-repetition ratios cancel most of the load drift of a shared host.
"""

import time

import numpy as np


def interleaved(arms, X, batch, reps, seed=0):
    """Return {arm: array of per-call seconds}, length ``reps``; row blocks cycle through X."""
    names = list(arms)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    batch = min(batch, n)
    for f in arms.values():  # warm-up: imports, allocator, first-touch
        f(X[:batch])
    out = {name: np.empty(reps) for name in names}
    for r in range(reps):
        start = int(rng.integers(0, n - batch + 1))
        rows = np.ascontiguousarray(X[start:start + batch])
        shift = r % len(names)
        for name in names[shift:] + names[:shift]:
            t0 = time.perf_counter()
            arms[name](rows)
            out[name][r] = time.perf_counter() - t0
    return out

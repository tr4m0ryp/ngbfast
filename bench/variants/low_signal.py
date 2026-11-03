"""Low-signal variant: when the target carries little signal, does the search pick a constant model?

Friedman #1 inputs (10 uniform features, 5 informative) with the signal scaled by ``a`` and unit
Gaussian noise, n_train rows. ``a = 0`` is pure noise: the best model is a fixed distribution, and
the question is whether the search finds it through leaf regularisation (constant learners, which
fold) or through few rounds / small steps of split trees (which do not). Run:
``python -m bench.variants.low_signal 0 0.1``.
"""

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.datasets import make_friedman1  # noqa: E402

from ..data import Task  # noqa: E402
from ..run.checks import measure  # noqa: E402
from ..run.driver import RESULTS  # noqa: E402
from ..tuning import fit_model, random_search  # noqa: E402


def make_task(a, seed, n_train=400, n_test=200):
    X, signal = make_friedman1(n_samples=n_train + n_test, noise=0.0, random_state=seed)
    rng = np.random.default_rng(seed)
    y = a * (signal - signal.mean()) / signal.std() + rng.standard_normal(signal.size)
    idx = np.arange(n_train + n_test)
    return Task(name=f"friedman1-a{a}", X=X, y=y, kind="normal",
                splits=[(idx[:n_train], idx[n_train:])], source="sklearn make_friedman1")


def run(levels, n_seeds=3, n_trials=15, reps=60):
    out = {"n_trials": n_trials, "n_folds": 3, "seeds": n_seeds, "levels": {}}
    for a in levels:
        t0 = time.time()
        picked, trials = random_search(make_task(a, 0), n_trials=n_trials, n_folds=3, n_jobs=2)
        tasks = [make_task(a, s) for s in range(n_seeds)]
        fits = Parallel(n_jobs=2)(delayed(fit_model)("normal", picked, s, t.X[t.splits[0][0]],
                                                     t.y[t.splits[0][0]]) for s, t in enumerate(tasks))
        rows = []
        for s, (t, (model, fit_s)) in enumerate(zip(tasks, fits)):
            tr, te = t.splits[0]
            info, _ = measure(model, "normal", t.X[te], t.y[te], fit_s, [(1, reps)], s, len(tr),
                              arm_subset=["pristine", "fold", "direct", "fold_direct"])
            rows.append({"seed": s, **info})
        out["levels"][str(a)] = {"picked": picked, "trials": trials, "fits": rows,
                                 "wall_s": time.time() - t0}
        print(f"[low_signal a={a}] picked {picked} shares "
              f"{[round(r['constant_share'], 3) for r in rows]} {time.time() - t0:.0f}s", flush=True)
    path = RESULTS / "variants" / "low_signal_friedman1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run([float(a) for a in sys.argv[1:]] or [0.0, 0.1])

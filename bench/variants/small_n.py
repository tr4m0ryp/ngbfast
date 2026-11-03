"""Learning-curve variant: does tuning on a small training set pick a configuration with constant learners?

For each training size n, a random search (3-fold CV NLL) runs on a size-n subsample of split 0's
training rows; the picked configuration is refit on S independent size-n subsamples (seed s), and
each fit reports its constant share and the paired 1-row latency of ours vs the upstream loop and vs
the fold alone. Run: ``python -m bench.variants.small_n concrete 100 250``.
"""

import json
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from ..data import load_credit_g, load_uci  # noqa: E402
from ..run.checks import measure  # noqa: E402
from ..run.driver import RESULTS  # noqa: E402
from ..tuning import fit_model, random_search  # noqa: E402

LOADERS = {"concrete": lambda: load_uci("concrete"), "yacht": lambda: load_uci("yacht"),
           "energy": lambda: load_uci("energy"), "credit-g": load_credit_g}


def subsample(task, n, seed):
    train, test = task.splits[0]
    rng = np.random.default_rng(1000 + seed)
    return rng.choice(train, size=min(n, train.size), replace=False), test


def run(name, sizes, n_seeds=3, n_trials=20, reps=60):
    task = LOADERS[name]()
    out = {"dataset": name, "n_trials": n_trials, "n_folds": 3, "seeds": n_seeds, "sizes": {}}
    for n in sizes:
        t0 = time.time()
        sub0, test = subsample(task, n, 0)
        small = replace(task, splits=[(sub0, test)])
        picked, trials = random_search(small, n_trials=n_trials, n_folds=3, n_jobs=2)
        subs = [subsample(task, n, s) for s in range(n_seeds)]
        fits = Parallel(n_jobs=2)(delayed(fit_model)(task.kind, picked, s, task.X[tr], task.y[tr])
                                  for s, (tr, _) in enumerate(subs))
        rows = []
        for s, ((tr, te), (model, fit_s)) in enumerate(zip(subs, fits)):
            info, _ = measure(model, task.kind, task.X[te], task.y[te], fit_s, [(1, reps)], s,
                              len(tr), arm_subset=["pristine", "fold", "fold_direct"])
            rows.append({"seed": s, **info})
        out["sizes"][str(n)] = {"picked": picked, "trials": trials, "fits": rows,
                                "wall_s": time.time() - t0}
        print(f"[{name} n={n}] picked {picked} shares "
              f"{[round(r['constant_share'], 3) for r in rows]} {time.time() - t0:.0f}s", flush=True)
    path = RESULTS / "variants" / f"small_n_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run(sys.argv[1], [int(a) for a in sys.argv[2:]] or [100, 250])

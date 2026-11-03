"""Run one dataset end to end and write ``bench/results/<name>.json``.

Phases: (1) the fixed grid on split 0, one seed, 1-row latency of four arms; (2) a random search on
split 0's training rows; (3) the searched and the default configuration on splits 0..S-1 with seed s,
every arm, three batch sizes. Fits run in parallel (2 jobs); latency runs alone afterwards.
"""

import argparse
import gzip
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import ngboost
import sklearn

from ..tuning import DEFAULT, fit_model, grid_configs, random_search
from .checks import measure

RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(os.environ.get("NGBFAST_RAW", RESULTS.parents[2] / "_data" / "ngbfast_raw"))
GRID_ARMS = ["pristine", "fold", "fold_direct"]


def _fit_all(task, jobs, n_jobs):
    return Parallel(n_jobs=n_jobs)(
        delayed(fit_model)(task.kind, cfg, seed, task.X[task.splits[s][0]], task.y[task.splits[s][0]])
        for cfg, seed, s in jobs)


def run_dataset(task, n_seeds=3, n_trials=30, n_folds=3, grid_reps=15,
                batches=((1, 100), (100, 50), (10**9, 30)), default_batches=((1, 100),), n_jobs=2):
    t_start = time.time()
    out = {"dataset": task.name, "source": task.source, "n": task.n, "d": task.d,
           "task": task.kind, "seeds": n_seeds, "load_start": os.getloadavg()}
    raw = {}
    tr0, te0 = task.splits[0]
    grid = grid_configs()
    fits = _fit_all(task, [(cfg, 0, 0) for cfg in grid], n_jobs)
    out["grid"] = []
    for cfg, (model, fit_s) in zip(grid, fits):
        info, _ = measure(model, task.kind, task.X[te0], task.y[te0], fit_s, [(1, grid_reps)], 0,
                          len(tr0), arm_subset=GRID_ARMS)
        out["grid"].append({"config": cfg, **info})
    print(f"[{task.name}] grid done {time.time() - t_start:.0f}s", flush=True)

    picked, trials = random_search(task, n_trials=n_trials, n_folds=n_folds, n_jobs=n_jobs)
    out["search"] = {"n_trials": n_trials, "n_folds": n_folds, "picked": picked, "trials": trials}
    print(f"[{task.name}] search done {time.time() - t_start:.0f}s picked {picked}", flush=True)

    out["tuned"], out["default"] = [], []
    jobs = [(cfg, s, s) for s in range(n_seeds) for cfg in (picked, DEFAULT)]
    fits = _fit_all(task, jobs, n_jobs)
    for (cfg, seed, s), (model, fit_s) in zip(jobs, fits):
        tr, te = task.splits[s]
        which = "tuned" if cfg is picked else "default"
        info, r = measure(model, task.kind, task.X[te], task.y[te], fit_s,
                          list(batches if which == "tuned" else default_batches), seed, len(tr))
        out[which].append({"seed": seed, "split": s, "config": cfg, **info})
        raw[f"{which}/seed{seed}"] = r
    out["load_end"] = os.getloadavg()
    out["wall_s"] = time.time() - t_start
    out["versions"] = {"ngboost": ngboost.__version__, "sklearn": sklearn.__version__,
                       "numpy": np.__version__, "python": platform.python_version()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{task.name}.json"
    path.write_text(json.dumps(out, indent=1))
    RAW.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW / f"{task.name}_latency_raw.json.gz", "wt") as fh:
        json.dump(raw, fh)
    print(f"[{task.name}] wrote {path} in {out['wall_s']:.0f}s", flush=True)
    return out


def main(loader):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="smoke run: tiny search and reps")
    args = parser.parse_args()
    task = loader()
    if args.quick:
        return run_dataset(task, n_seeds=2, n_trials=2, grid_reps=3, batches=((1, 5), (10**9, 3)),
                           default_batches=((1, 3),))
    return run_dataset(task, n_seeds=args.seeds, n_trials=args.trials)

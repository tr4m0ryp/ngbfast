"""Random search by cross-validated NLL; the number of rounds is picked from the staged CV curve.

Only the training rows of split 0 are used, so the test rows of every split stay untouched by the
choice (splits 1 and 2 share training rows with split 0's search; the reported test metrics are for
model selection transparency, not a tuning-free generalisation estimate).
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold

import ngbfast

from .model import fit_model, nll

SPACE = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100, 200, 400],
         "minibatch_frac": [1.0, 0.8, 0.5, 0.3], "learning_rate": "log-uniform [0.005, 0.1]",
         "n_estimators": 500}


def draw(rng):
    return {"max_depth": int(rng.choice(SPACE["max_depth"])),
            "min_samples_leaf": int(rng.choice(SPACE["min_samples_leaf"])),
            "minibatch_frac": float(rng.choice(SPACE["minibatch_frac"])),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))),
            "n_estimators": SPACE["n_estimators"]}


def _curve(kind, cfg, seed, X, y, tr, va):
    try:
        model, _ = fit_model(kind, cfg, seed, X[tr], y[tr])
    except ValueError:  # upstream fit diverged (non-finite natural gradient): the trial fails
        return np.full(cfg["n_estimators"], np.inf)
    ngbfast.fold(model)
    stages = ngbfast.staged_pred_param(model, X[va])
    curve = np.array([nll(kind, p, y[va]).mean() for p in stages])
    curve = np.where(np.isfinite(curve), curve, np.inf)
    # upstream stops early when the projected gradient vanishes; later rounds would predict the same
    return np.concatenate([curve, np.full(cfg["n_estimators"] - curve.size, curve[-1])])


def random_search(task, n_trials=50, n_folds=3, seed=0, n_jobs=2):
    train = task.splits[0][0]
    X, y = task.X[train], task.y[train]
    rng = np.random.default_rng(seed)
    configs = [draw(rng) for _ in range(n_trials)]
    cv = (StratifiedKFold if task.kind == "bernoulli" else KFold)(n_folds, shuffle=True,
                                                                   random_state=seed)
    folds = list(cv.split(X, y))
    jobs = [(t, f) for t in range(n_trials) for f in range(n_folds)]
    curves = Parallel(n_jobs=n_jobs)(
        delayed(_curve)(task.kind, configs[t], seed, X, y, *folds[f]) for t, f in jobs)
    trials = []
    for t, cfg in enumerate(configs):
        mean = np.mean([curves[j] for j, (tt, _) in enumerate(jobs) if tt == t], axis=0)
        best = int(np.argmin(mean))
        trials.append({**cfg, "best_rounds": best + 1, "cv_nll": float(mean[best]),
                       "failed": bool(not np.isfinite(mean[best]))})
    pick = min(trials, key=lambda r: r["cv_nll"])
    picked = {k: pick[k] for k in ("max_depth", "min_samples_leaf", "minibatch_frac",
                                   "learning_rate")}
    picked["n_estimators"] = pick["best_rounds"]
    return picked, trials

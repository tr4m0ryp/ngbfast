"""Arms of the evidence plan (spec section 4).

* ``pristine``: upstream ``NGBoost.pred_param`` (ngboost 0.5.11 from PyPI, unchanged).
* ``fold``: the fold plan, every remaining learner called through its public ``predict``.
* ``direct``: direct tree traversal only; nothing folded (a plan in which no learner counts as constant).
* ``fold_direct``: the full method (ours).
* ``merged``: ablation, all constant steps of a parameter summed into one scalar; fast, not bitwise.
* ``floor``: one broadcast of the fixed parameters; exists only when every learner is constant.
"""

import copy
from unittest import mock

import numpy as np

import ngbfast
from ngbfast.fold import build as fold_build
from ngbfast.fold.inputs import Inputs

ARMS = ["pristine", "fold", "direct", "fold_direct", "merged", "floor"]
EXACT_ARMS = ["fold", "direct", "fold_direct"]
OURS = "fold_direct"


def _with_plan(model, plan):
    clone = copy.copy(model)  # shares the fitted learners; only fold_ differs
    clone.fold_ = plan
    return clone


def _direct_only_plan(model):
    with mock.patch.object(fold_build, "is_constant", lambda learner, fold_linear=False: False):
        return fold_build.build_plan(model)


def _merged(model, plan):
    """Start values with every constant step of a column summed at once (np.sum, pairwise order)."""
    lr, init = model.learning_rate, np.ones(plan["n_params"]) * model.init_params
    steps = [[] for _ in range(plan["n_params"])]
    evaluated = []
    probe_row = np.zeros((1, max(plan["n_features_needed"], 1)))
    for i, (learners, s, col_idx) in enumerate(
        zip(model.base_models, model.scalings, model.col_idxs)
    ):
        for k, learner in enumerate(learners):
            if ngbfast.is_constant(learner):
                v = float(learner.predict(probe_row[:, col_idx])[0])
                steps[k].append(lr * v * s)
            else:
                evaluated.append((i, k))
    start = init - np.array([np.sum(np.array(st, dtype=np.float64)) for st in steps])

    def predict(X):
        inputs = Inputs(X, plan, direct=True)
        params = np.ones((X.shape[0], plan["n_params"])) * start
        for i, k in evaluated:
            pred = inputs.predict(model.base_models[i][k], i, model.col_idxs[i], True)
            params[:, k] -= lr * np.asarray(pred) * model.scalings[i]
        return params

    return predict


def make_arms(model):
    """Return (arms, plan, info): arms maps name -> f(X); ``floor`` is present only when it applies."""
    plain = _with_plan(model, None)
    plan = ngbfast.build_plan(model)
    folded = _with_plan(model, plan)
    direct = _with_plan(model, _direct_only_plan(model))
    arms = {
        "pristine": plain.pred_param,
        "fold": lambda X: ngbfast.pred_param(folded, X, direct=False),
        "direct": lambda X: ngbfast.pred_param(direct, X, direct=True),
        "fold_direct": lambda X: ngbfast.pred_param(folded, X, direct=True),
        "merged": _merged(model, plan),
    }
    if plan.n_evaluated == 0:
        start = plan.start
        arms["floor"] = lambda X: np.ones((X.shape[0], plan["n_params"])) * start
    info = {
        "n_rounds": int(plan["n_rounds"]),
        "n_params": int(plan["n_params"]),
        "constant_share": float(plan.constant_share),
        "constant_per_param": [int(c) for c in plan["n_constant_per_param"]],
        "n_evaluated": int(plan.n_evaluated),
    }
    return arms, plan, info

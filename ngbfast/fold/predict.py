"""Prediction from a fold plan: full, truncated (``max_iter``) and staged, all bit-identical."""

import numpy as np
from sklearn.utils import check_array

from ..reference import loop_pred_param, loop_staged_pred_param
from .inputs import Inputs
from .plan import as_plan, plan_matches


def current_plan(model):
    """The model's plan if it matches the model's current rounds, else None."""
    plan = as_plan(getattr(model, "fold_", None))
    return plan if plan_matches(plan, model) else None


def _apply_round(params, model, inputs, i, entries, dtype):
    lr, s, col_idx = model.learning_rate, model.scalings[i], model.col_idxs[i]
    for k, step, direct in entries:
        if step is None:
            pred = inputs.predict(model.base_models[i][k], i, col_idx, direct)
            params[:, k] -= lr * np.asarray(pred).astype(dtype, copy=False) * s
        else:
            params[:, k] -= step


def pred_param(model, X, max_iter=None, direct=True):
    """theta(X) of shape (n_rows, n_params), bit-identical to ``NGBoost.pred_param``.

    Uses the model's plan (``model.fold_``) when it matches the model; otherwise falls back to the
    upstream loop. ``direct=False`` disables direct tree traversal (the fold alone).
    """
    plan = current_plan(model)
    if plan is None:
        return loop_pred_param(model, X, max_iter=max_iter)
    inputs = Inputs(X, plan, direct=direct)
    n_rounds = plan["n_rounds"]
    cut = min(max_iter, n_rounds) if max_iter else n_rounds
    start = plan.start if cut == n_rounds else plan.start_at(cut)
    params = np.ones((inputs.X.shape[0], plan["n_params"])) * start
    for i, entries, dtype in plan["rounds"]:
        if i >= cut:
            break
        _apply_round(params, model, inputs, i, entries, np.dtype(dtype))
    return params


def staged_pred_param(model, X, max_iter=None, direct=True):
    """List of theta(X) after each round, bit-identical to the upstream staged prediction."""
    plan = current_plan(model)
    if plan is None:
        return loop_staged_pred_param(model, X, max_iter=max_iter)
    inputs = Inputs(X, plan, direct=direct)
    work = {i: (entries, np.dtype(dtype)) for i, entries, dtype in plan["rounds"]}
    chains, first_split = plan["chains"], plan["first_split"]
    params = np.ones((inputs.X.shape[0], plan["n_params"])) * model.init_params
    out = []
    for i in range(plan["n_rounds"]):
        for k in range(plan["n_params"]):
            if i < first_split[k]:
                params[:, k] = chains[k][i + 1]
        if i in work:
            _apply_round(params, model, inputs, i, *work[i])
        out.append(np.copy(params))
        if max_iter and i + 1 == max_iter:
            break
    return out


def _check_X(X):
    return check_array(X, accept_sparse=True, ensure_all_finite="allow-nan")


def pred_dist(model, X, max_iter=None, direct=True):
    """The predicted distribution, as ``NGBoost.pred_dist`` returns it."""
    params = np.asarray(pred_param(model, _check_X(X), max_iter=max_iter, direct=direct))
    return model.Dist(params.T)


def staged_pred_dist(model, X, max_iter=None, direct=True):
    """One predicted distribution per round, as ``NGBoost.staged_pred_dist`` returns them."""
    stages = staged_pred_param(model, _check_X(X), max_iter=max_iter, direct=direct)
    return [model.Dist(p.T) for p in stages]

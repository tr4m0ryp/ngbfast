"""The upstream NGBoost prediction loop, reproduced as the reference for exactness checks.

These two functions repeat what ``NGBoost.pred_param`` and ``NGBoost.staged_pred_dist`` do in
ngboost 0.5.11: call every base learner of every boosting round and apply its update in round order.
They are kept here so that any model, patched or not, can be checked against the loop bit for bit.
"""

import numpy as np


def loop_pred_param(model, X, max_iter=None):
    """Parameters theta(X), shape (n_rows, n_params), by calling every learner in round order."""
    params = np.ones((X.shape[0], model.Manifold.n_params)) * model.init_params
    for i, (models, s, col_idx) in enumerate(
        zip(model.base_models, model.scalings, model.col_idxs)
    ):
        if max_iter and i == max_iter:
            break
        resids = np.array([learner.predict(X[:, col_idx]) for learner in models]).T
        params -= model.learning_rate * resids * s
    return params


def loop_staged_pred_param(model, X, max_iter=None):
    """List of theta(X) after each boosting round, as the upstream staged prediction builds them."""
    out = []
    params = np.ones((X.shape[0], model.Dist.n_params)) * model.init_params
    for i, (models, s, col_idx) in enumerate(
        zip(model.base_models, model.scalings, model.col_idxs), start=1
    ):
        resids = np.array([learner.predict(X[:, col_idx]) for learner in models]).T
        params -= model.learning_rate * resids * s
        out.append(np.copy(params))
        if max_iter and i == max_iter:
            break
    return out


def compare(model, X, fast_params, max_iter=None):
    """Exactness of ``fast_params`` against the loop.

    Returns the share of entries whose float64 bit patterns are identical (so -0.0 and 0.0 count as
    different, and NaNs count as equal only with the same payload) and the largest absolute difference.
    """
    ref = np.ascontiguousarray(loop_pred_param(model, X, max_iter=max_iter), dtype=np.float64)
    fast = np.ascontiguousarray(fast_params, dtype=np.float64)
    if ref.shape != fast.shape:
        raise ValueError(f"shape mismatch: loop {ref.shape}, fast {fast.shape}")
    same = ref.view(np.uint64) == fast.view(np.uint64)
    with np.errstate(invalid="ignore"):
        diff = np.abs(np.where(same, 0.0, ref - fast))
    return {
        "bitwise_share": float(same.mean()) if same.size else 1.0,
        "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        "shape": ref.shape,
    }

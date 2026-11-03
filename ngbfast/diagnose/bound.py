"""When can a boosting round's tree split at all? (Lemma 2 of the method.)

A CART tree needs at least ``2 * min_samples_leaf`` rows (both children need ``min_samples_leaf``)
and at least ``min_samples_split`` rows to split its root. NGBoost fits every round's trees on a
minibatch of ``int(minibatch_frac * n)`` rows, where ``n`` is the training size after any early
stopping hold-out. If the minibatch is below that bound, no round can split, every learner is a
one-node tree and the model predicts one fixed distribution for every input. Under k-fold
cross-validation the bound applies to the training-fold size, so a tuning search can pick such a
configuration without any visible error.
"""

import math

from sklearn.tree import BaseDecisionTree


def batch_rows(n_rows, minibatch_frac=1.0, early_stopping=False, validation_fraction=0.1):
    """Rows each round's learners are fitted on, as NGBoost samples them."""
    n = int(n_rows)
    if early_stopping:
        n -= math.ceil(validation_fraction * n)
    if minibatch_frac != 1.0:
        n = int(minibatch_frac * n)
    return n


def _resolve(value, n_rows, floor):
    if isinstance(value, float):
        return max(floor, math.ceil(value * n_rows))
    return int(value)


def split_bound(base, n_rows):
    """Smallest row count at which ``base`` (an unfitted tree) can split, or None if unknown.

    Returns a dict ``{"min_rows": int, "min_samples_leaf": int, "min_samples_split": int}``; the
    float (fraction) forms of the two parameters are resolved against ``n_rows``.
    """
    if not isinstance(base, BaseDecisionTree):
        return None
    params = base.get_params()
    leaf = _resolve(params["min_samples_leaf"], n_rows, 1)
    split = _resolve(params["min_samples_split"], n_rows, 2)
    if params.get("max_depth") == 0 or params.get("max_leaf_nodes") == 1:
        return {"min_rows": math.inf, "min_samples_leaf": leaf, "min_samples_split": split}
    return {
        "min_rows": max(split, 2 * leaf),
        "min_samples_leaf": leaf,
        "min_samples_split": split,
    }


def feasibility(model, n_rows, holdout=None):
    """Per-parameter split feasibility of an NGBoost model's base learners on ``n_rows`` rows.

    ``holdout`` says whether NGBoost carved an early-stopping validation set out of the ``n_rows``
    (it does when ``early_stopping_rounds`` is set and no validation data is passed); the default
    assumes it did whenever ``early_stopping_rounds`` is set. Returns a list with one dict per distribution parameter: ``batch_rows``, ``min_rows``
    (None when the base learner is not a scikit-learn tree) and ``feasible`` (None if unknown).
    """
    rows = batch_rows(
        n_rows,
        model.minibatch_frac,
        model.early_stopping_rounds is not None if holdout is None else holdout,
        model.validation_fraction,
    )
    bases = model.Base if isinstance(model.Base, (list, tuple)) else [model.Base] * model.Manifold.n_params
    out = []
    for base in bases:
        bound = split_bound(base, rows)
        min_rows = None if bound is None else bound["min_rows"]
        out.append({
            "batch_rows": rows,
            "min_rows": min_rows,
            "feasible": None if bound is None else rows >= min_rows,
        })
    return out

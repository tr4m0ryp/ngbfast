"""What ngbfast adds around NGBoost's own fit and predict methods, shared by the patch and the mixin."""

import math

from sklearn.utils.validation import _num_samples

from ..diagnose import warn_if_degenerate
from ..fold import current_plan, fold
from ..fold import pred_param as folded_pred_param
from ..fold import staged_pred_dist as folded_staged_pred_dist


def training_rows(model, X, X_val, Y_val):
    """Rows the rounds' minibatches are drawn from (after NGBoost's own early-stopping hold-out)."""
    n = _num_samples(X)
    if model.early_stopping_rounds is not None and X_val is None and Y_val is None:
        n -= math.ceil(model.validation_fraction * n)
    return n


def run_partial_fit(model, original, X, Y, args, kwargs):
    """Reset the plan, fit with ``original``, then fold and warn if every learner is constant.

    The reset matters: ``partial_fit`` predicts with the existing rounds to seed the new ones, and a
    plan built from the previous fit would carry stale initial parameters into that prediction.
    """
    model.fold_ = None
    X_val = kwargs.get("X_val", args[0] if len(args) > 0 else None)
    Y_val = kwargs.get("Y_val", args[1] if len(args) > 1 else None)
    n_rows = training_rows(model, X, X_val, Y_val)
    result = original(model, X, Y, *args, **kwargs)
    model.n_train_rows_ = n_rows
    fold(model, fold_linear=getattr(model, "fold_linear", False))
    warn_if_degenerate(model, stacklevel=5)
    return result


def run_pred_param(model, original, X, max_iter=None):
    if current_plan(model) is None:
        return original(model, X, max_iter)
    return folded_pred_param(model, X, max_iter=max_iter)


def run_staged_pred_dist(model, original, X, max_iter=None):
    if current_plan(model) is None:
        return original(model, X, max_iter)
    return folded_staged_pred_dist(model, X, max_iter=max_iter)

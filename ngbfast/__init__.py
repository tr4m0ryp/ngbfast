"""ngbfast: bit-identical faster prediction for NGBoost.

After training, a base learner that returns the same value for every input (a one-node tree, a tree
whose leaves agree, a DummyRegressor) contributes a precomputed scalar. ``fold`` builds a plan that
keeps every floating-point operation of NGBoost's prediction loop in its original order, so
``pred_param`` returns exactly the same numbers while calling only the learners that depend on the
input, and calling scikit-learn trees without repeated input validation.

    import ngbfast
    ngbfast.fold(model)                  # attach the plan as model.fold_
    theta = ngbfast.pred_param(model, X) # bit-identical to model.pred_param(X)
    ngbfast.diagnose(model, n_train=n)   # constant-learner share, split-feasibility bound
    ngbfast.patch()                      # opt-in: every NGBoost fit folds and predicts via the plan
"""

from .diagnose import (
    NGBoostDegenerateWarning,
    batch_rows,
    degenerate_message,
    diagnose,
    feasibility,
    split_bound,
    warn_if_degenerate,
)
from .fold import (
    FeatureCountError,
    FoldPlan,
    build_plan,
    current_plan,
    fold,
    pred_dist,
    pred_param,
    staged_pred_dist,
    staged_pred_param,
    unfold,
)
from .integrate import (
    FastNGBClassifier,
    FastNGBoost,
    FastNGBRegressor,
    FastNGBSurvival,
    FoldMixin,
    is_patched,
    patch,
    patched,
    unpatch,
)
from .learners import is_constant, is_direct_tree
from .reference import compare, loop_pred_param, loop_staged_pred_param

__version__ = "0.1.0"

__all__ = [
    "FastNGBClassifier",
    "FastNGBRegressor",
    "FastNGBSurvival",
    "FastNGBoost",
    "FeatureCountError",
    "FoldMixin",
    "FoldPlan",
    "NGBoostDegenerateWarning",
    "batch_rows",
    "build_plan",
    "compare",
    "current_plan",
    "degenerate_message",
    "diagnose",
    "feasibility",
    "fold",
    "is_constant",
    "is_direct_tree",
    "is_patched",
    "loop_pred_param",
    "loop_staged_pred_param",
    "patch",
    "patched",
    "pred_dist",
    "pred_param",
    "split_bound",
    "staged_pred_dist",
    "staged_pred_param",
    "unfold",
    "unpatch",
    "warn_if_degenerate",
]

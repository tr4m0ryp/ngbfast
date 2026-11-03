"""Constant folding of NGBoost base learners and the folded prediction paths."""

from .build import build_plan, fold, unfold
from .inputs import FeatureCountError
from .plan import FoldPlan, as_plan
from .predict import current_plan, pred_dist, pred_param, staged_pred_dist, staged_pred_param

__all__ = [
    "FeatureCountError",
    "FoldPlan",
    "as_plan",
    "build_plan",
    "current_plan",
    "fold",
    "pred_dist",
    "pred_param",
    "staged_pred_dist",
    "staged_pred_param",
    "unfold",
]

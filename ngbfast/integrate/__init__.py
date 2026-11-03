"""Ways to use folding with NGBoost: subclasses, or an opt-in patch of ngboost.NGBoost."""

from .estimators import FastNGBClassifier, FastNGBoost, FastNGBRegressor, FastNGBSurvival, FoldMixin
from .patch import is_patched, patch, patched, unpatch

__all__ = [
    "FastNGBClassifier",
    "FastNGBRegressor",
    "FastNGBSurvival",
    "FastNGBoost",
    "FoldMixin",
    "is_patched",
    "patch",
    "patched",
    "unpatch",
]

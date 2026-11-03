"""Drop-in NGBoost estimators that fold at fit end and predict through the plan, without patching."""

from ngboost import NGBClassifier, NGBoost, NGBRegressor, NGBSurvival

from ..diagnose import diagnose
from ..fold import fold
from . import hooks


class FoldMixin:
    """Adds folding to an NGBoost class. Put it first in the bases: ``class X(FoldMixin, NGBRegressor)``.

    Every fitted model carries ``fold_`` (the plan), ``n_train_rows_`` and warns with
    ``NGBoostDegenerateWarning`` when every base learner is constant.
    """

    _ngbfast_managed = True

    def partial_fit(self, X, Y, *args, **kwargs):
        return hooks.run_partial_fit(
            self, lambda m, *a, **k: super(FoldMixin, m).partial_fit(*a, **k), X, Y, args, kwargs
        )

    def pred_param(self, X, max_iter=None):
        return hooks.run_pred_param(
            self, lambda m, X_, it: super(FoldMixin, m).pred_param(X_, it), X, max_iter
        )

    def staged_pred_dist(self, X, max_iter=None):
        return hooks.run_staged_pred_dist(
            self, lambda m, X_, it: super(FoldMixin, m).staged_pred_dist(X_, it), X, max_iter
        )

    def fold_constant_learners(self, fold_linear=False):
        """Rebuild the plan, for example after editing base_models, scalings or init_params."""
        fold(self, fold_linear=fold_linear)
        return self

    def diagnose(self, n_train=None):
        """See ``ngbfast.diagnose``."""
        return diagnose(self, n_train=n_train)


class FastNGBoost(FoldMixin, NGBoost):
    """``ngboost.NGBoost`` with constant folding."""


class FastNGBRegressor(FoldMixin, NGBRegressor):
    """``ngboost.NGBRegressor`` with constant folding."""


class FastNGBClassifier(FoldMixin, NGBClassifier):
    """``ngboost.NGBClassifier`` with constant folding."""


class FastNGBSurvival(FoldMixin, NGBSurvival):
    """``ngboost.NGBSurvival`` with constant folding."""

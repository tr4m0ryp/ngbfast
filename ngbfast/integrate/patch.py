"""Opt-in monkeypatch of ``ngboost.NGBoost``: fold at the end of every fit, predict through the plan.

After ``ngbfast.patch()`` every NGBoost model (NGBRegressor, NGBClassifier, NGBSurvival and plain
NGBoost) folds its constant learners when ``fit`` / ``partial_fit`` returns, warns with
``NGBoostDegenerateWarning`` when all of them are constant, and predicts through the plan. Outputs are
bit-identical to the unpatched methods. ``ngbfast.unpatch()`` restores the original methods.
"""

from contextlib import contextmanager

from ngboost import NGBoost

from . import hooks

_ORIGINAL = {}


def _make_partial_fit(original):
    def partial_fit(self, X, Y, *args, **kwargs):
        if getattr(type(self), "_ngbfast_managed", False):
            return original(self, X, Y, *args, **kwargs)
        return hooks.run_partial_fit(self, original, X, Y, args, kwargs)

    partial_fit.__doc__ = original.__doc__
    return partial_fit


def _make_pred_param(original):
    def pred_param(self, X, max_iter=None):
        return hooks.run_pred_param(self, original, X, max_iter)

    return pred_param


def _make_staged_pred_dist(original):
    def staged_pred_dist(self, X, max_iter=None):
        return hooks.run_staged_pred_dist(self, original, X, max_iter)

    staged_pred_dist.__doc__ = original.__doc__
    return staged_pred_dist


_FACTORIES = {
    "partial_fit": _make_partial_fit,
    "pred_param": _make_pred_param,
    "staged_pred_dist": _make_staged_pred_dist,
}


def is_patched():
    return bool(_ORIGINAL)


def patch():
    """Patch ``ngboost.NGBoost`` in place. Idempotent."""
    if _ORIGINAL:
        return
    for name, factory in _FACTORIES.items():
        original = NGBoost.__dict__[name]
        _ORIGINAL[name] = original
        setattr(NGBoost, name, factory(original))


def unpatch():
    """Restore the original ``ngboost.NGBoost`` methods. Fitted plans stay on the models, unused."""
    for name, original in _ORIGINAL.items():
        setattr(NGBoost, name, original)
    _ORIGINAL.clear()


@contextmanager
def patched():
    """Context manager form of ``patch()`` / ``unpatch()``."""
    already = is_patched()
    patch()
    try:
        yield
    finally:
        if not already:
            unpatch()

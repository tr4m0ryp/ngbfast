"""Shared toy data and model builders for the tests (synthetic data only)."""

import numpy as np
from ngboost import NGBoost
from ngboost.distns import Normal
from ngboost.scores import LogScore
from numpy.testing import assert_array_equal
from sklearn.tree import DecisionTreeRegressor


def toy(n=300, seed=0, p=4):
    """Heteroscedastic regression: a step in the mean on x0, a step in the noise on x1."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    Y = 1.5 * (X[:, 0] > 0.3) + rng.normal(scale=0.5 + 0.8 * (X[:, 1] > 0), size=n)
    return X, Y


class NormalFixedInit(Normal):
    """Normal started at loc 0, log scale 0 instead of the marginal fit.

    With a fixed initialisation, constant learners take nonzero steps (from the marginal fit their
    mean gradient is about zero and boosting stops at once), so folding has real work to do.
    """

    @staticmethod
    def fit(Y):
        return np.array([0.0, 0.0])


def const_tree():
    return DecisionTreeRegressor(max_depth=2, min_samples_leaf=10_000, random_state=0)


def split_tree():
    return DecisionTreeRegressor(max_depth=2, min_samples_leaf=10, random_state=0)


def ngb(base, n=40, cls=NGBoost, **kw):
    kw.setdefault("Dist", NormalFixedInit)
    kw.setdefault("Score", LogScore)
    return cls(Base=base, n_estimators=n, learning_rate=0.05, verbose=False, random_state=0, **kw)


def constant_flags(model):
    return [t.tree_.node_count == 1 for ms in model.base_models for t in ms]


def assert_bitwise(a, b):
    """Equal float64 bit patterns, so -0.0 vs 0.0 or a last-bit difference fails."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    assert a.shape == b.shape
    assert_array_equal(a.view(np.uint64), b.view(np.uint64))

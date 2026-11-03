"""Direct traversal (G1): the same float32 values through the same trees, so the same outputs."""

import numpy as np
import scipy.sparse as sp
from helpers import assert_bitwise, ngb, split_tree, toy
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

import ngbfast
from ngbfast import is_direct_tree, loop_pred_param


def test_tree_kinds():
    X, Y = toy()
    assert is_direct_tree(DecisionTreeRegressor().fit(X, Y))
    assert is_direct_tree(ExtraTreeRegressor().fit(X, Y))
    assert not is_direct_tree(DecisionTreeRegressor())  # unfitted
    assert not is_direct_tree(DecisionTreeRegressor().fit(X, np.c_[Y, Y]))  # two outputs


def test_direct_equals_validated_predict_on_awkward_values():
    """Values that round when cast to float32, tiny and huge (finite in float32) entries."""
    rng = np.random.RandomState(7)
    X = rng.normal(size=(500, 4)) * np.array([1e-30, 1.0, 1e30, 1.0 / 3.0])
    Y = (X[:, 0] > 0) + 0.1 * rng.normal(size=500)
    m = ngb(DecisionTreeRegressor(max_depth=4, min_samples_leaf=5), n=30, col_sample=0.75).fit(X, Y)
    ngbfast.fold(m)
    X2 = rng.normal(size=(300, 4)) * np.array([1e-30, 1.0, 1e30, 1.0 / 3.0])
    assert_bitwise(ngbfast.pred_param(m, X2), loop_pred_param(m, X2))
    assert_bitwise(ngbfast.pred_param(m, X2, direct=False), loop_pred_param(m, X2))


def test_nan_routing_identical():
    X, Y = toy()
    X = X.copy()
    X[::7, 0] = np.nan
    m = ngb(split_tree(), n=20).fit(X, Y)
    ngbfast.fold(m)
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))


def test_sparse_input_takes_the_validated_path():
    X, Y = toy()
    X[np.abs(X) < 0.8] = 0.0
    Xs = sp.csr_matrix(X)
    m = ngb(split_tree(), n=15).fit(Xs, Y)
    ngbfast.fold(m)
    assert_bitwise(ngbfast.pred_param(m, Xs), loop_pred_param(m, Xs))

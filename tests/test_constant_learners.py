"""Which learners fold (G2) and that folding each kind keeps the output bit-identical."""

import numpy as np
from helpers import assert_bitwise, const_tree, ngb, split_tree, toy
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import ngbfast
from ngbfast import is_constant, loop_pred_param


def test_one_node_tree_and_equal_leaves():
    X, _ = toy()
    assert is_constant(const_tree().fit(X, X[:, 0]))
    t = DecisionTreeRegressor(max_depth=1).fit(X, (X[:, 0] > 0).astype(float))
    assert not is_constant(t)
    t.tree_.value[1:] = 0.25  # two leaves, same value: no input can change the output
    assert is_constant(t)
    assert np.all(t.predict(X) == 0.25)


def test_signed_zero_leaves_are_not_equal():
    X, _ = toy()
    t = DecisionTreeRegressor(max_depth=1).fit(X, (X[:, 0] > 0).astype(float))
    t.tree_.value[1], t.tree_.value[2] = 0.0, -0.0
    assert not is_constant(t)


def test_dummy_regressor_folds():
    X, _ = toy()
    assert is_constant(DummyRegressor().fit(X, X[:, 0]))


def test_zero_coefficient_linear_is_opt_in():
    X, _ = toy()
    lin = Ridge(alpha=1e12).fit(X, X[:, 0])
    lin.coef_[:] = 0.0
    assert not is_constant(lin)
    assert is_constant(lin, fold_linear=True)
    assert not is_constant(Ridge().fit(X, X[:, 0]), fold_linear=True)


def test_dummy_base_model_bit_identical():
    X, Y = toy()
    m = ngb(DummyRegressor(), n=25).fit(X, Y)
    plan = ngbfast.fold(m)
    assert plan.n_constant == plan.n_learners == 50 and plan["rounds"] == []
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))


def test_equal_leaf_trees_inside_a_split_model():
    X, Y = toy()
    m = ngb([split_tree(), split_tree()], n=12).fit(X, Y)
    for ms in m.base_models[::3]:
        t = ms[1]
        t.tree_.value[t.tree_.children_left == -1] = 0.125
    plan = ngbfast.fold(m)
    assert plan["n_constant_per_param"][1] == 4
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))


def test_linear_base_opt_in_fold_exact_on_finite_input():
    X, Y = toy()
    m = ngb(Ridge(alpha=1e15), n=10).fit(X, Y)
    for ms in m.base_models:
        for lin in ms:
            lin.coef_[:] = 0.0
    plan = ngbfast.fold(m, fold_linear=True)
    assert plan.n_constant == plan.n_learners
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))
    assert ngbfast.fold(m).n_constant == 0  # default: linear learners are evaluated
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))

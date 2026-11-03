"""Lemma 1: the folded prediction is bit-identical to the upstream loop on every fitted model."""

import numpy as np
import pytest
from helpers import assert_bitwise, const_tree, constant_flags, ngb, split_tree, toy
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, LogNormal, TFixedDf, k_categorical
from sklearn.tree import DecisionTreeRegressor

import ngbfast
from ngbfast import loop_pred_param


@pytest.mark.parametrize("direct", [True, False])
def test_all_constant_model_needs_no_learner_call(direct):
    X, Y = toy()
    m = ngb(const_tree(), minibatch_frac=0.7, col_sample=0.5).fit(X, Y)
    assert all(constant_flags(m)) and len(m.base_models) == 40
    plan = ngbfast.fold(m)
    assert np.abs(loop_pred_param(m, X)[0] - m.init_params).max() > 0.1  # the constants carry the fit
    assert plan["rounds"] == [] and plan.n_evaluated == 0
    assert_bitwise(ngbfast.pred_param(m, X, direct=direct), loop_pred_param(m, X))
    assert_bitwise(np.ones((1, 2)) * plan.start, loop_pred_param(m, X[:1]))


def test_constant_in_one_parameter_only():
    """The mean splits, the scale never does: the scale column folds, the mean keeps its trees."""
    X, Y = toy()
    m = ngb([split_tree(), const_tree()]).fit(X, Y)
    flags = np.array(constant_flags(m)).reshape(-1, 2)
    assert not flags[:, 0].all() and flags[:, 1].all()
    plan = ngbfast.fold(m)
    assert {k for _, entries, _ in plan["rounds"] for k, step, _ in entries if step is None} == {0}
    assert int(plan["first_split"][1]) == plan["n_rounds"]
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))


def test_trailing_constants_stay_as_ordered_steps():
    """Constants after a column's first split cannot join the start without reordering additions."""
    X, Y = toy()
    m = ngb([split_tree(), split_tree()], n=30).fit(X, Y)
    for ms in m.base_models[10:]:
        for k in range(2):
            ms[k] = const_tree().fit(X, np.full(len(X), 0.01 * (k + 1)))
    plan = ngbfast.fold(m)
    scalar_steps = sum(step is not None for _, e, _ in plan["rounds"] for _, step, _ in e)
    assert scalar_steps == 40
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))


def test_merging_scalar_steps_would_break_exactness():
    """The ablation the order-preserving plan avoids: (a - s1) - s2 differs from a - (s1 + s2)."""
    a, s1, s2 = np.float64(1.0), np.float64(2.0**-54), np.float64(2.0**-54)
    assert (a - s1) - s2 == 1.0  # each half-ulp step rounds back to 1.0
    assert a - (s1 + s2) == 1.0 - 2.0**-53  # the merged step does not


@pytest.mark.parametrize(
    "make",
    [
        lambda: NGBRegressor(Dist=TFixedDf, n_estimators=40, verbose=False, random_state=1),
        lambda: NGBRegressor(Dist=LogNormal, n_estimators=40, verbose=False, random_state=1,
                             minibatch_frac=0.5, col_sample=0.5),
        lambda: NGBRegressor(n_estimators=40, verbose=False, random_state=1,
                             Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=70)),
    ],
)
def test_regressors(make):
    X, Y = toy(n=250, seed=3)
    m = make().fit(X, np.abs(Y) + 0.1)
    ngbfast.fold(m)
    assert_bitwise(ngbfast.pred_param(m, X), loop_pred_param(m, X))
    assert ngbfast.compare(m, X, ngbfast.pred_param(m, X)) ["bitwise_share"] == 1.0


@pytest.mark.parametrize("dist,K", [(Bernoulli, 2), (k_categorical(3), 3)])
def test_classifiers(dist, K):
    X, Y = toy(n=300, seed=4)
    y = np.digitize(Y, np.quantile(Y, np.linspace(0, 1, K + 1)[1:-1]))
    c = NGBClassifier(Dist=dist, n_estimators=30, verbose=False, random_state=0,
                      Base=DecisionTreeRegressor(max_depth=2, min_samples_leaf=40)).fit(X, y)
    ngbfast.fold(c)
    assert_bitwise(ngbfast.pred_param(c, X), loop_pred_param(c, X))
    assert_bitwise(ngbfast.pred_dist(c, X).class_probs(), c.pred_dist(X).class_probs())


def test_early_stopping_and_sample_weight():
    X, Y = toy(n=400, seed=5)
    w = np.random.RandomState(0).uniform(0.5, 2, size=len(Y))
    r = NGBRegressor(n_estimators=60, verbose=False, random_state=0, early_stopping_rounds=3,
                     Base=DecisionTreeRegressor(max_depth=2, min_samples_leaf=150)).fit(X, Y, sample_weight=w)
    ngbfast.fold(r)
    assert_bitwise(ngbfast.pred_param(r, X), loop_pred_param(r, X))


def test_wider_input_is_fine_like_upstream():
    X, Y = toy()
    m = ngb(split_tree(), n=10).fit(X, Y)
    ngbfast.fold(m)
    Xw = np.hstack([X, X])
    assert_bitwise(ngbfast.pred_param(m, Xw), loop_pred_param(m, Xw))


def test_pred_dist_matches_method():
    X, Y = toy()
    m = ngb([split_tree(), const_tree()]).fit(X, Y)
    ngbfast.fold(m)
    assert_bitwise(ngbfast.pred_dist(m, X).params["loc"], m.pred_dist(X).params["loc"])
    assert_bitwise(ngbfast.pred_dist(m, X).params["scale"], m.pred_dist(X).params["scale"])

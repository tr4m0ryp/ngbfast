"""Fold at fit end: the opt-in patch of ngboost.NGBoost and the Fast* subclasses."""

import warnings

import pytest
from helpers import assert_bitwise, const_tree, ngb, split_tree, toy
from ngboost import NGBoost, NGBRegressor, NGBSurvival
from sklearn.base import clone

import ngbfast
from ngbfast import NGBoostDegenerateWarning, loop_pred_param, loop_staged_pred_param


@pytest.fixture
def patched():
    ngbfast.patch()
    yield
    ngbfast.unpatch()


def test_patch_folds_at_fit_end_and_restores(patched):
    X, Y = toy()
    m = ngb([split_tree(), const_tree()], n=20).fit(X, Y)
    assert ngbfast.current_plan(m) is not None
    assert_bitwise(m.pred_param(X), loop_pred_param(m, X))
    assert_bitwise(m.pred_param(X, max_iter=7), loop_pred_param(m, X, 7))
    for a, b in zip(m.staged_pred_dist(X), loop_staged_pred_param(m, X)):
        assert_bitwise(a.params["loc"], b[:, 0])
    ngbfast.unpatch()
    assert not ngbfast.is_patched()
    assert NGBoost.pred_param.__module__ == "ngboost.ngboost"
    ngbfast.patch()


def test_patch_warns_when_every_learner_is_constant(patched):
    X, Y = toy()
    with pytest.warns(NGBoostDegenerateWarning):
        ngb(const_tree(), n=5).fit(X, Y)


def test_refit_and_partial_fit_rebuild_the_plan(patched):
    X, Y = toy()
    X2, Y2 = toy(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NGBoostDegenerateWarning)
        m = ngb(const_tree(), n=15).fit(X, Y)
        m.fit(X2, Y2 + 3.0)  # a stale plan would carry the old initial parameters
        assert_bitwise(m.pred_param(X2), loop_pred_param(m, X2))
        m.partial_fit(X, Y)
    assert len(m.base_models) == 30 and ngbfast.current_plan(m)["n_rounds"] == 30
    assert_bitwise(m.pred_param(X), loop_pred_param(m, X))


def test_patched_fit_trains_the_same_model(patched):
    """Folding happens after training, so the fitted rounds are those of an unpatched fit."""
    X, Y = toy()
    a = ngb(split_tree(), n=15, minibatch_frac=0.6).fit(X, Y)
    ngbfast.unpatch()
    b = ngb(split_tree(), n=15, minibatch_frac=0.6).fit(X, Y)
    ngbfast.patch()
    assert a.scalings == b.scalings
    assert_bitwise(loop_pred_param(a, X), loop_pred_param(b, X))


@pytest.mark.parametrize("cls", [ngbfast.FastNGBRegressor, ngbfast.FastNGBoost])
def test_fast_estimators(cls):
    X, Y = toy()
    m = cls(Base=split_tree(), n_estimators=15, verbose=False, random_state=0).fit(X, Y)
    assert ngbfast.current_plan(m) is not None
    assert_bitwise(m.pred_param(X), loop_pred_param(m, X))
    assert_bitwise(m.predict(X), NGBRegressor.predict(m, X))
    c = clone(m)
    assert type(c) is cls and c.get_params()["n_estimators"] == 15


def test_fast_classifier_and_survival():
    X, Y = toy()
    c = ngbfast.FastNGBClassifier(Base=split_tree(), n_estimators=10, verbose=False).fit(X, (Y > 1).astype(int))
    assert_bitwise(c.predict_proba(X)[:, 1], c.pred_dist(X).class_probs()[:, 1])
    assert_bitwise(c.pred_param(X), loop_pred_param(c, X))
    T, E = 1.0 + (Y - Y.min()), (Y > 0.5).astype(int)
    s = ngbfast.FastNGBSurvival(Base=split_tree(), n_estimators=10, verbose=False).fit(X, T, E)
    assert ngbfast.current_plan(s) is not None
    assert_bitwise(s.pred_param(X), loop_pred_param(s, X))
    assert isinstance(s, NGBSurvival)


def test_fast_estimator_together_with_patch_folds_once(patched):
    X, Y = toy()
    with pytest.warns(NGBoostDegenerateWarning) as rec:
        ngbfast.FastNGBRegressor(Base=const_tree(), n_estimators=5, verbose=False).fit(X, Y)
    assert sum(issubclass(w.category, NGBoostDegenerateWarning) for w in rec) == 1

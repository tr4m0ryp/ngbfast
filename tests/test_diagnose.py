"""Lemma 2 (split feasibility) and the degeneracy diagnostic (G4)."""

import math
import warnings

import numpy as np
import pytest
from helpers import ngb, split_tree, toy
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor

import ngbfast
from ngbfast import NGBoostDegenerateWarning, batch_rows, split_bound


def test_batch_rows_follows_ngboost_sampling():
    assert batch_rows(1000) == 1000
    assert batch_rows(1000, minibatch_frac=0.37) == int(0.37 * 1000)
    assert batch_rows(1000, early_stopping=True, validation_fraction=0.1) == 900
    assert batch_rows(1001, 0.5, True, 0.1) == int(0.5 * (1001 - math.ceil(100.1)))


def test_split_bound_values():
    assert split_bound(DecisionTreeRegressor(min_samples_leaf=50), 1000)["min_rows"] == 100
    assert split_bound(DecisionTreeRegressor(min_samples_split=300, min_samples_leaf=50), 1000)["min_rows"] == 300
    assert split_bound(DecisionTreeRegressor(min_samples_leaf=0.1), 1000)["min_rows"] == 200
    assert split_bound(object(), 1000) is None


@pytest.mark.parametrize("leaf", [10, 60, 74, 75, 76, 150])
def test_bound_is_exact_on_real_fits(leaf):
    """Below the bound no tree splits (every learner constant); at or above it trees can split."""
    X, Y = toy(n=300)
    m = NGBRegressor(Base=DecisionTreeRegressor(max_depth=2, min_samples_leaf=leaf), n_estimators=8,
                     minibatch_frac=0.5, verbose=False, random_state=0).fit(X, Y)
    rep = ngbfast.diagnose(m, n_train=300)
    feasible = rep["per_param"][0]["feasible"]
    assert feasible == (150 >= 2 * leaf)
    if not feasible:
        assert rep["degenerate"] and rep["constant_share"] == 1.0
    else:
        assert rep["n_constant"] < rep["n_learners"]


def test_warning_names_the_bound():
    X, Y = toy(n=300)
    m = ngbfast.FastNGBRegressor(Base=DecisionTreeRegressor(min_samples_leaf=100), n_estimators=5,
                                 minibatch_frac=0.5, verbose=False, random_state=0)
    with pytest.warns(NGBoostDegenerateWarning, match="150 rows, but a split needs at least 200"):
        m.fit(X, Y)
    assert m.n_train_rows_ == 300
    assert m.diagnose()["per_param"][0]["feasible"] is False


def test_no_warning_when_trees_split():
    X, Y = toy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", NGBoostDegenerateWarning)
        ngbfast.FastNGBRegressor(Base=split_tree(), n_estimators=5, verbose=False).fit(X, Y)


def test_report_fields():
    X, Y = toy()
    m = ngb([split_tree(), DecisionTreeRegressor(min_samples_leaf=1000)], n=10).fit(X, Y)
    rep = ngbfast.diagnose(m)
    assert rep["n_learners"] == 20 and rep["n_constant"] == 10 and rep["constant_share"] == 0.5
    assert rep["n_evaluated"] == 10 and not rep["degenerate"]
    assert rep["per_param"][0]["first_input_dependent_round"] == 0
    assert rep["per_param"][1]["first_input_dependent_round"] is None
    assert "feasible" not in rep["per_param"][0]  # n_train unknown
    assert np.isclose(rep["constant_share"], 0.5)

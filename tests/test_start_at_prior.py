"""The plan starts from the model's initial parameters, and an empty model predicts exactly them."""

import numpy as np
from helpers import assert_bitwise, const_tree, ngb, split_tree, toy

import ngbfast
from ngbfast import loop_pred_param


def test_chains_start_at_initial_parameters():
    X, Y = toy()
    m = ngb([split_tree(), const_tree()], n=15).fit(X, Y)
    plan = ngbfast.fold(m)
    for k, chain in enumerate(plan["chains"]):
        assert_bitwise(chain[:1], (np.ones(2) * m.init_params)[k : k + 1])
    assert_bitwise(plan.start_at(0), np.ones(2) * m.init_params)


def test_zero_rounds_predicts_the_initial_parameters():
    X, Y = toy()
    m = ngb(split_tree(), n=5).fit(X, Y)
    m.base_models, m.scalings, m.col_idxs = [], [], []
    plan = ngbfast.fold(m)
    assert plan["n_rounds"] == 0 and plan.n_learners == 0
    out = ngbfast.pred_param(m, X)
    assert_bitwise(out, loop_pred_param(m, X))
    assert_bitwise(out, np.ones((len(X), 2)) * m.init_params)


def test_rows_are_independent():
    """No information flows between rows: predicting one row alone equals its row in the batch."""
    X, Y = toy()
    m = ngb([split_tree(), split_tree()], n=20, minibatch_frac=0.5).fit(X, Y)
    ngbfast.fold(m)
    batch = ngbfast.pred_param(m, X)
    for r in (0, 17, len(X) - 1):
        assert_bitwise(ngbfast.pred_param(m, X[r : r + 1]), batch[r : r + 1])

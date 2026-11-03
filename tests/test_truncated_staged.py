"""Truncated (max_iter) and staged prediction from the plan, bit-identical to the loop."""

import numpy as np
import pytest
from helpers import assert_bitwise, const_tree, ngb, split_tree, toy

import ngbfast
from ngbfast import loop_pred_param, loop_staged_pred_param


def _mixed_model():
    """Column 0 constant for 8 rounds, then splitting; column 1 always constant; trailing constants."""
    X, Y = toy()
    m = ngb([split_tree(), split_tree()], n=25).fit(X, Y)
    for i, ms in enumerate(m.base_models):
        if i < 8 or i >= 20:
            ms[0] = const_tree().fit(X, np.full(len(X), 0.03 * (i + 1)))
        ms[1] = const_tree().fit(X, np.full(len(X), -0.02 * (i + 1)))
    ngbfast.fold(m)
    return m, X


@pytest.mark.parametrize("max_iter", [None, 0, 1, 5, 8, 9, 19, 20, 24, 25, 100])
def test_max_iter(max_iter):
    m, X = _mixed_model()
    assert_bitwise(ngbfast.pred_param(m, X, max_iter=max_iter), loop_pred_param(m, X, max_iter))


@pytest.mark.parametrize("max_iter", [None, 1, 8, 13, 25])
def test_staged(max_iter):
    m, X = _mixed_model()
    fast = ngbfast.staged_pred_param(m, X, max_iter=max_iter)
    ref = loop_staged_pred_param(m, X, max_iter=max_iter)
    assert len(fast) == len(ref)
    for a, b in zip(fast, ref):
        assert_bitwise(a, b)


def test_staged_dist_matches_upstream_method():
    m, X = _mixed_model()
    fast = ngbfast.staged_pred_dist(m, X)
    ref = m.staged_pred_dist(X)  # unpatched: the upstream loop
    assert len(fast) == len(ref) == 25
    for a, b in zip(fast, ref):
        assert_bitwise(a.params["loc"], b.params["loc"])


def test_all_constant_staged():
    X, Y = toy()
    m = ngb(const_tree(), n=20).fit(X, Y)
    ngbfast.fold(m)
    for a, b in zip(ngbfast.staged_pred_param(m, X), loop_staged_pred_param(m, X)):
        assert_bitwise(a, b)
    assert len(m.base_models) == 20  # rounds stay stored


@pytest.mark.parametrize("cut", [1, 8, 15, 24])
def test_no_lookahead(cut):
    """A prediction truncated at round r uses nothing from later rounds: it equals the plan of the
    same model with every round after r deleted."""
    m, X = _mixed_model()
    full = ngbfast.pred_param(m, X, max_iter=cut)
    staged = ngbfast.staged_pred_param(m, X)[cut - 1]
    m.base_models, m.scalings, m.col_idxs = m.base_models[:cut], m.scalings[:cut], m.col_idxs[:cut]
    ngbfast.fold(m)
    alone = ngbfast.pred_param(m, X)
    assert_bitwise(full, alone)
    assert_bitwise(staged, alone)

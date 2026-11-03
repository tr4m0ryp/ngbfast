"""Speed: an all-constant model predicts without learner calls; direct traversal helps split models."""

import time

import numpy as np
import pytest
from helpers import const_tree, ngb, split_tree, toy

import ngbfast
from ngbfast import loop_pred_param


def _median_ratio(slow, fast, reps=15):
    a, b = [], []
    for _ in range(reps):  # interleaved, so load changes hit both arms alike
        t = time.perf_counter()
        slow()
        a.append(time.perf_counter() - t)
        t = time.perf_counter()
        fast()
        b.append(time.perf_counter() - t)
    return np.median(a) / np.median(b)


@pytest.mark.slow
def test_all_constant_one_row_at_least_20x():
    X, Y = toy()
    m = ngb(const_tree(), n=300).fit(X, Y)
    ngbfast.fold(m)
    x = X[:1]
    assert _median_ratio(lambda: loop_pred_param(m, x), lambda: ngbfast.pred_param(m, x)) >= 20


@pytest.mark.slow
def test_direct_traversal_faster_on_split_model():
    X, Y = toy()
    m = ngb(split_tree(), n=100).fit(X, Y)
    ngbfast.fold(m)
    x = X[:1]
    ratio = _median_ratio(lambda: ngbfast.pred_param(m, x, direct=False), lambda: ngbfast.pred_param(m, x))
    assert ratio >= 1.5

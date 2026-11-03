"""The folded path rejects what the skipped learners would have rejected."""

import numpy as np
import pytest
from helpers import const_tree, ngb, split_tree, toy

import ngbfast
from ngbfast import FeatureCountError, loop_pred_param


@pytest.fixture(params=["constant", "split"])
def model(request):
    X, Y = toy()
    base = const_tree() if request.param == "constant" else split_tree()
    m = ngb(base, n=10).fit(X, Y)
    ngbfast.fold(m)
    return m, X


def test_narrow_input_rejected(model):
    m, X = model
    with pytest.raises((ValueError, IndexError)):
        loop_pred_param(m, X[:, :2])
    with pytest.raises(FeatureCountError):
        ngbfast.pred_param(m, X[:, :2])


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 1e39])
def test_infinite_or_float32_overflow_rejected(model, bad):
    m, X = model
    X = X.copy()
    X[3, 1] = bad
    with pytest.raises(ValueError, match="infinity"):
        loop_pred_param(m, X)
    with pytest.raises(ValueError, match="infinity"):
        ngbfast.pred_param(m, X)


def test_one_dimensional_input_rejected(model):
    m, X = model
    with pytest.raises(ValueError):
        ngbfast.pred_param(m, X[0])


def test_stale_plan_is_ignored():
    X, Y = toy()
    m = ngb(const_tree(), n=10).fit(X, Y)
    ngbfast.fold(m)
    m.base_models.append(m.base_models[-1])
    m.scalings.append(m.scalings[-1])
    m.col_idxs.append(m.col_idxs[-1])
    assert ngbfast.current_plan(m) is None
    np.testing.assert_array_equal(ngbfast.pred_param(m, X), loop_pred_param(m, X))
    m.init_params = m.init_params + 1.0
    ngbfast.fold(m)
    m.init_params = m.init_params - 1.0
    assert ngbfast.current_plan(m) is None

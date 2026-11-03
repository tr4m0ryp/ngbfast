"""Which fitted base learners provably return the same value for every input row.

A learner is folded only when its output cannot depend on the input. Three kinds qualify:

* a scikit-learn regression tree with one output whose leaves all hold the same value, bit for bit
  (this covers the common one-node tree, where no split was feasible);
* a fitted ``DummyRegressor`` with one output;
* opt-in (``fold_linear=True``): a linear model whose coefficients are all zero. Its prediction
  ``X @ 0 + b`` equals ``b`` only for finite inputs, so this is not the default.
"""

import numpy as np
from sklearn.base import is_classifier
from sklearn.dummy import DummyRegressor
from sklearn.tree import BaseDecisionTree


def _tree_is_constant(learner):
    tree = getattr(learner, "tree_", None)
    if tree is None or is_classifier(learner) or getattr(learner, "n_outputs_", 1) != 1:
        return False
    if tree.node_count == 1:
        return True
    leaves = tree.children_left == -1
    values = np.ascontiguousarray(tree.value[leaves].reshape(int(leaves.sum()), -1)[:, 0])
    if values.dtype != np.float64:
        values = values.astype(np.float64)
    bits = values.view(np.uint64)
    return bool((bits == bits[0]).all())


def _dummy_is_constant(learner):
    constant = getattr(learner, "constant_", None)
    return constant is not None and np.size(constant) == 1


def _linear_is_constant(learner):
    coef = getattr(learner, "coef_", None)
    intercept = getattr(learner, "intercept_", None)
    if coef is None or intercept is None or np.size(intercept) != 1:
        return False
    coef = np.asarray(coef)
    if coef.ndim > 1 and coef.shape[0] != 1:
        return False
    return bool(np.all(coef == 0))


def is_constant(learner, fold_linear=False):
    """True when ``learner`` (fitted) returns one value for every input row.

    Parameters
    ----------
    learner : fitted estimator
        A base learner taken from ``model.base_models``.
    fold_linear : bool, default False
        Also treat linear models with all-zero coefficients as constant. Exact only for finite inputs.
    """
    if isinstance(learner, BaseDecisionTree):
        return _tree_is_constant(learner)
    if isinstance(learner, DummyRegressor):
        return _dummy_is_constant(learner)
    if fold_linear:
        return _linear_is_constant(learner)
    return False

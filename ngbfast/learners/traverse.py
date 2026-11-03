"""Direct traversal of scikit-learn regression trees.

``DecisionTreeRegressor.predict`` validates its input on every call (dtype conversion to float32,
finiteness, feature count). NGBoost calls it once per learner per boosting round, so for small
batches the validation, not the traversal, dominates prediction time. Here the input is converted to
float32 and validated once per prediction, and each tree is then called with ``check_input=False``.
The tree walks the same float32 values through the same nodes, so the output is identical.
"""

from sklearn.base import is_classifier
from sklearn.tree import BaseDecisionTree


def is_direct_tree(learner):
    """True for a fitted single-output scikit-learn regression tree."""
    return (
        isinstance(learner, BaseDecisionTree)
        and not is_classifier(learner)
        and getattr(learner, "n_outputs_", 1) == 1
        and getattr(learner, "tree_", None) is not None
    )


def predict_direct(learner, X32):
    """Predict with a regression tree on an already validated float32 matrix."""
    return learner.predict(X32, check_input=False)

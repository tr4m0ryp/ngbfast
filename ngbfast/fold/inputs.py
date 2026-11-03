"""Input handling for the folded path: one validation per prediction instead of one per learner.

The checks reproduce what the skipped ``predict`` calls would have done, restricted to the columns
those learners read: NaN where a learner rejects NaN, infinity (after the float32 cast for trees,
which also catches values too large for float32) where a learner rejects it, and the column count.
"""

import numpy as np
import scipy.sparse as sp


class FeatureCountError(ValueError, IndexError):
    """X has fewer columns than the model's learners read (upstream raises IndexError here)."""


def _any(block, test):
    if sp.issparse(block):
        return bool(test(block.data).any())
    return bool(test(block).any())


class Inputs:
    """The prediction matrix, its float32 view (built on first use) and per-round column slices."""

    def __init__(self, X, plan, direct=True):
        self.sparse = sp.issparse(X)
        self.X = X if self.sparse else np.asarray(X)
        if self.X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got an array of shape {self.X.shape}")
        self.direct = direct and not self.sparse
        self._X32 = None
        self._slice = (None, None, None)
        self._check(plan)

    @property
    def X32(self):
        if self._X32 is None:
            with np.errstate(over="ignore"):
                self._X32 = np.asarray(self.X, dtype=np.float32)
        return self._X32

    def _check(self, plan):
        need = plan["n_features_needed"]
        if self.X.shape[1] < need:
            raise FeatureCountError(
                f"X has {self.X.shape[1]} features, but the model's learners read column {need - 1}"
            )
        if plan["nan_cols"].size and _any(self.X[:, plan["nan_cols"]], np.isnan):
            raise ValueError("Input X contains NaN.")
        if plan["inf64_cols"].size and _any(self.X[:, plan["inf64_cols"]], np.isinf):
            raise ValueError("Input X contains infinity or a value too large for dtype('float64').")
        if plan["inf32_cols"].size:
            block = self.X[:, plan["inf32_cols"]]
            if self.sparse:
                with np.errstate(over="ignore"):
                    bad = _any(block.astype(np.float32), np.isinf)
            else:
                bad = _any(self.X32[:, plan["inf32_cols"]], np.isinf)
            if bad:
                raise ValueError(
                    "Input X contains infinity or a value too large for dtype('float32')."
                )

    def predict(self, learner, round_index, col_idx, direct):
        """Learner output for every row, through direct traversal when allowed."""
        use_direct = direct and self.direct
        key = (round_index, use_direct)
        if self._slice[:2] != key:
            source = self.X32 if use_direct else self.X
            self._slice = (round_index, use_direct, source[:, col_idx])
        X_i = self._slice[2]
        if use_direct:
            return learner.predict(X_i, check_input=False)
        return learner.predict(X_i)

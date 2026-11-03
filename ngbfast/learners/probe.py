"""One-row probes of a fitted learner: output dtype and shape, and which inputs it rejects.

The folded path skips the per-call ``predict`` of some learners. To keep the same contract it must
know what those learners would have returned (dtype) and which inputs they would have refused (NaN,
infinity). Both are read from the learner's own behaviour on a single probe row, not assumed.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .traverse import is_direct_tree


@dataclass(frozen=True)
class Probe:
    """What one learner returned on a zero row and whether it accepts NaN / infinite inputs."""

    value: np.ndarray  # shape (1,), the learner's output for a row of zeros
    accepts_nan: bool
    accepts_inf: bool
    float32_input: bool  # the learner casts its input to float32 before checking it


def _accepts(learner, n_features, fill):
    row = np.full((1, n_features), fill)
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            learner.predict(row)
    except ValueError:
        return False
    return True


def probe(learner, n_features, checks=True):
    """Probe ``learner`` on one row with ``n_features`` columns.

    The zero-row output must be one-dimensional with one entry, as NGBoost assumes. With
    ``checks=False`` only the output is probed (for learners the folded path still calls).
    """
    value = np.asarray(learner.predict(np.zeros((1, n_features))))
    if value.shape != (1,):
        raise ValueError(
            f"base learner {type(learner).__name__} returned shape {value.shape} for one row; "
            "NGBoost expects one value per row"
        )
    if not checks or n_features == 0:
        return Probe(value, True, True, False)
    return Probe(
        value,
        _accepts(learner, n_features, np.nan),
        _accepts(learner, n_features, np.inf),
        is_direct_tree(learner),
    )

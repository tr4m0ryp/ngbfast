"""Degeneracy diagnostics: constant-learner share and the split-feasibility bound."""

from .bound import batch_rows, feasibility, split_bound
from .report import NGBoostDegenerateWarning, degenerate_message, diagnose, warn_if_degenerate

__all__ = [
    "NGBoostDegenerateWarning",
    "batch_rows",
    "degenerate_message",
    "diagnose",
    "feasibility",
    "split_bound",
    "warn_if_degenerate",
]

"""The benchmark arms: callables ``f(X) -> theta`` built from one fitted NGBoost model."""

from .build import ARMS, EXACT_ARMS, OURS, make_arms

__all__ = ["ARMS", "EXACT_ARMS", "OURS", "make_arms"]

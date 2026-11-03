"""Facts about individual base learners: which are constant, which can be traversed directly."""

from .constant import is_constant
from .probe import Probe, probe
from .traverse import is_direct_tree, predict_direct

__all__ = ["Probe", "is_constant", "is_direct_tree", "predict_direct", "probe"]

"""Dataset loaders: each returns a Task with X, y, the distribution kind and its train/test splits."""

from .openml import load_credit_g
from .task import DATA_ROOT, Task
from .uci import load_uci

__all__ = ["DATA_ROOT", "Task", "load_credit_g", "load_uci"]

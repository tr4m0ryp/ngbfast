"""The task record shared by every dataset loader."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("NGBFAST_DATA", Path(__file__).resolve().parents[3] / "_data"))


@dataclass
class Task:
    name: str
    X: np.ndarray
    y: np.ndarray
    kind: str  # "normal" (NGBRegressor, Normal) or "bernoulli" (NGBClassifier)
    splits: list = field(default_factory=list)  # list of (train_idx, test_idx)
    source: str = ""

    @property
    def n(self):
        return int(self.X.shape[0])

    @property
    def d(self):
        return int(self.X.shape[1])

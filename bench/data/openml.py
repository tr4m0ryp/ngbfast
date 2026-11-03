"""credit-g (OpenML 31): 1,000 credit applicants, binary good/bad, one-hot encoded categoricals."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit

from .task import DATA_ROOT, Task


def load_credit_g(n_splits=3, test_size=0.1):
    frame = fetch_openml(data_id=31, as_frame=True, data_home=str(DATA_ROOT / "openml"))
    X = pd.get_dummies(frame.data, drop_first=False).to_numpy(dtype=np.float64)
    y = (frame.target.astype(str) == "bad").to_numpy().astype(np.int64)
    splits = []
    for seed in range(n_splits):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        splits.append(next(sss.split(X, y)))
    return Task(name="credit-g", X=X, y=y, kind="bernoulli", splits=splits,
                source="https://www.openml.org/d/31")

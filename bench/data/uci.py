"""UCI regression sets with the fixed test masks of github.com/treforevans/uci_datasets."""

import urllib.request

import numpy as np
import pandas as pd

from .task import DATA_ROOT, Task

URL = "https://raw.githubusercontent.com/treforevans/uci_datasets/master/uci_datasets/{name}/{file}"


def _fetch(name, file):
    path = DATA_ROOT / "uci" / name / file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(name=name, file=file), path)
    return path


def load_uci(name, n_splits=3):
    """Features are every column but the last, the target is the last; split j uses mask column j."""
    data = pd.read_csv(_fetch(name, "data.csv.gz"), header=None).to_numpy(dtype=np.float64)
    mask = pd.read_csv(_fetch(name, "test_mask.csv.gz"), header=None).to_numpy().astype(bool)
    splits = []
    for j in range(n_splits):
        test = np.flatnonzero(mask[:, j])
        train = np.flatnonzero(~mask[:, j])
        splits.append((train, test))
    return Task(name=name, X=data[:, :-1], y=data[:, -1], kind="normal", splits=splits,
                source=URL.format(name=name, file="data.csv.gz"))

"""The fixed configuration grid of the evidence plan: leaf size x minibatch share x depth."""

import itertools

GRID = {"min_samples_leaf": [1, 20, 100, 400], "minibatch_frac": [1.0, 0.5, 0.2],
        "max_depth": [1, 3]}


def grid_configs(n_estimators=500, learning_rate=0.01):
    out = []
    for leaf, frac, depth in itertools.product(*GRID.values()):
        out.append({"max_depth": depth, "min_samples_leaf": leaf, "minibatch_frac": frac,
                    "learning_rate": learning_rate, "n_estimators": n_estimators})
    return out

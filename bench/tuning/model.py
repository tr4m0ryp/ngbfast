"""NGBoost models for the two task kinds, and the per-example negative log-likelihood."""

import time
import warnings

import numpy as np
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Normal
from sklearn.tree import DecisionTreeRegressor

DEFAULT = {"max_depth": 3, "min_samples_leaf": 1, "minibatch_frac": 1.0, "learning_rate": 0.01,
           "n_estimators": 500}


def make_model(kind, cfg, seed):
    base = DecisionTreeRegressor(criterion="squared_error", max_depth=cfg["max_depth"],
                                 min_samples_leaf=cfg["min_samples_leaf"], random_state=seed)
    common = dict(Base=base, n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                  minibatch_frac=cfg["minibatch_frac"], verbose=False, random_state=seed)
    if kind == "normal":
        return NGBRegressor(Dist=Normal, **common)
    if kind == "bernoulli":
        return NGBClassifier(Dist=Bernoulli, **common)
    raise ValueError(kind)


def fit_model(kind, cfg, seed, X, y):
    """Fit and return (model, seconds)."""
    model = make_model(kind, cfg, seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model, time.perf_counter() - t0


def nll(kind, theta, y):
    """Per-example negative log-likelihood from the parameter matrix (n, n_params)."""
    theta = np.asarray(theta)
    if kind == "normal":
        loc, log_scale = theta[:, 0], theta[:, 1]
        z = (y - loc) / np.exp(log_scale)
        return 0.5 * np.log(2 * np.pi) + log_scale + 0.5 * z * z
    logit = theta[:, 0]
    return np.logaddexp(0.0, logit) - y * logit

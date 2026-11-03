"""ngbfast quickstart: fold an NGBoost model, check it is bit-identical, time it, diagnose it.

Runs on synthetic data (scikit-learn's Friedman #1 regression problem); nothing is downloaded.
"""

import time
import warnings

import numpy as np
from ngboost import NGBRegressor
from sklearn.datasets import make_friedman1
from sklearn.tree import DecisionTreeRegressor

import ngbfast


def per_call_ms(fn, reps=30):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times))


def show(name, model, X_test):
    plan = ngbfast.fold(model)
    exact = ngbfast.compare(model, X_test, ngbfast.pred_param(model, X_test))
    row = X_test[:1]
    loop = per_call_ms(lambda: ngbfast.loop_pred_param(model, row))
    fast = per_call_ms(lambda: ngbfast.pred_param(model, row))
    print(f"\n{name}: {plan}")
    print(f"  bitwise share {exact['bitwise_share']:.3f}, max |diff| {exact['max_abs_diff']:.1e}")
    print(f"  one-row pred_param: loop {loop:.2f} ms, folded {fast:.2f} ms ({loop / fast:.1f}x)")


X, y = make_friedman1(n_samples=3000, noise=1.0, random_state=0)
X_train, y_train, X_test = X[:2000], y[:2000], X[2000:]

# A model whose trees split: folding finds little, direct traversal still removes per-call overhead.
split = NGBRegressor(Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=20), n_estimators=300,
                     verbose=False, random_state=0).fit(X_train, y_train)
show("trees split", split, X_test)

# A strongly regularised, minibatched model: each round sees int(0.2 * 2000) = 400 rows, but a split
# needs 2 * 250 = 500, so every learner is a one-node tree. The fit "works", yet the model predicts
# one fixed distribution for every input. ngbfast folds it to a single broadcast and says so.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    degenerate = ngbfast.FastNGBRegressor(
        Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=250), n_estimators=300,
        minibatch_frac=0.2, verbose=False, random_state=0).fit(X_train, y_train)
for w in caught:
    if issubclass(w.category, ngbfast.NGBoostDegenerateWarning):
        print(f"\nwarning: {w.message}")
show("every learner constant", degenerate, X_test)
report = ngbfast.diagnose(degenerate)
print(f"  constant share {report['constant_share']:.2f}; per parameter:")
for row in report["per_param"]:
    print(f"    param {row['param']}: {row['constant']}/{row['learners']} constant, "
          f"batch {row['batch_rows']} rows, split needs {row['min_rows']}")

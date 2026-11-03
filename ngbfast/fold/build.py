"""Build the fold plan of a fitted NGBoost model (Lemma 1 of the method).

Upstream NGBoost evaluates, for every round i and parameter k,
``theta_k <- theta_k - (learning_rate * f_ik(x)) * scale_i``. When ``f_ik`` is constant (value v), the
update is the scalar ``(learning_rate * v) * scale_i`` for every row. Before the first round in which
column k has an input-dependent learner, all of k's updates are such scalars, so the subtraction chain
is computed once (``chains``). After it, scalar updates stay in place between the evaluated learners;
they are never merged, because merging reorders floating-point subtractions. Each column therefore
sees the same rounded operations in the same order as the loop, and the result is bit-identical.
"""

import numpy as np

from ..learners import is_constant, is_direct_tree, probe
from .plan import PLAN_VERSION, FoldPlan


def _ordered_union(groups):
    if not groups:
        return np.zeros(0, dtype=np.intp)
    return np.unique(np.concatenate(groups)).astype(np.intp)


def _column(values, dtype):
    """A one-element array, so every scalar operation runs through numpy's array arithmetic."""
    return np.asarray(values, dtype=dtype).reshape(1)


def build_plan(model, fold_linear=False):
    """Return the FoldPlan of a fitted NGBoost ``model`` without attaching it.

    Parameters
    ----------
    model : fitted ngboost.NGBoost (or a subclass)
    fold_linear : bool, default False
        Also fold linear learners with all-zero coefficients (exact for finite inputs only).
    """
    if model.init_params is None:
        raise ValueError("model is not fitted: init_params is None")
    n_params = model.Manifold.n_params
    n_rounds = len(model.base_models)
    lr = model.learning_rate
    init = np.ones(n_params) * model.init_params
    chains = [[_column(init[k], np.float64)] for k in range(n_params)]
    first_split = [n_rounds] * n_params
    n_constant = np.zeros(n_params, dtype=np.int64)
    rounds, need = [], 0
    nan_cols, inf32_cols, inf64_cols = [], [], []

    for i, (models, s, col_idx) in enumerate(
        zip(model.base_models, model.scalings, model.col_idxs)
    ):
        col_idx = np.asarray(col_idx, dtype=np.intp)
        if col_idx.size:
            need = max(need, int(col_idx.max()) + 1)
        flags = [is_constant(learner, fold_linear) for learner in models]
        probes = [
            probe(learner, col_idx.size, checks=const or is_direct_tree(learner))
            for learner, const in zip(models, flags)
        ]
        dtype = np.result_type(*[p.value.dtype for p in probes])
        entries = []
        for k, (learner, const, pr) in enumerate(zip(models, flags, probes)):
            if const or is_direct_tree(learner):
                if not pr.accepts_nan:
                    nan_cols.append(col_idx)
                if not pr.accepts_inf:
                    (inf32_cols if pr.float32_input else inf64_cols).append(col_idx)
            if not const:
                if first_split[k] == n_rounds:
                    first_split[k] = i
                entries.append((k, None, is_direct_tree(learner)))
                continue
            n_constant[k] += 1
            step = lr * _column(pr.value, dtype) * s
            if first_split[k] == n_rounds:
                chains[k].append(chains[k][-1] - step)
            else:
                entries.append((k, step[0], False))
        if entries:
            rounds.append((i, entries, np.dtype(dtype).str))

    return FoldPlan(
        version=PLAN_VERSION,
        n_params=n_params,
        n_rounds=n_rounds,
        init_params=np.array(init, dtype=np.float64),
        chains=[np.concatenate(c).astype(np.float64) for c in chains],
        first_split=np.array(first_split, dtype=np.int64),
        n_constant_per_param=n_constant,
        rounds=rounds,
        n_features_needed=need,
        nan_cols=_ordered_union(nan_cols),
        inf32_cols=_ordered_union(inf32_cols),
        inf64_cols=_ordered_union(inf64_cols),
        fold_linear=bool(fold_linear),
    )


def fold(model, fold_linear=False):
    """Build the plan, attach it as ``model.fold_`` and return it.

    Call again after editing ``base_models``, ``scalings`` or ``init_params`` by hand; a plan that no
    longer matches the model's rounds is ignored by ``ngbfast.pred_param``.
    """
    plan = build_plan(model, fold_linear=fold_linear)
    model.fold_ = plan
    return plan


def unfold(model):
    """Remove the plan; prediction falls back to the upstream loop."""
    model.fold_ = None
    return model

"""Diagnostics of a fitted NGBoost model: constant-learner share and the split-feasibility bound."""

import warnings

from ..fold import build_plan, current_plan
from .bound import feasibility


class NGBoostDegenerateWarning(UserWarning):
    """Every base learner of a fitted NGBoost model is constant: it predicts one fixed distribution."""


def diagnose(model, n_train=None):
    """Summarise how much of a fitted model's ensemble depends on the input.

    Parameters
    ----------
    model : fitted ngboost.NGBoost
    n_train : int, optional
        Rows passed to ``fit``. When omitted, the training rows ngbfast recorded at fit time
        (``model.n_train_rows_``, after any early-stopping hold-out) are used if present.

    Returns
    -------
    dict with ``n_learners``, ``n_constant``, ``constant_share``, ``degenerate`` (every learner
    constant), ``per_param`` (constant count, first input-dependent round, and the split bound when
    ``n_train`` is known) and ``n_evaluated`` (learners a folded prediction still calls).
    """
    plan = current_plan(model) or build_plan(model)
    if n_train is not None:
        bounds = feasibility(model, n_train)
    elif getattr(model, "n_train_rows_", None) is not None:
        bounds = feasibility(model, model.n_train_rows_, holdout=False)
    else:
        bounds = [None] * plan["n_params"]
    per_param = []
    for k in range(plan["n_params"]):
        first = int(plan["first_split"][k])
        row = {
            "param": k,
            "constant": int(plan["n_constant_per_param"][k]),
            "learners": int(plan["n_rounds"]),
            "first_input_dependent_round": None if first == plan["n_rounds"] else first,
        }
        if bounds[k] is not None:
            row.update(bounds[k])
        per_param.append(row)
    return {
        "n_rounds": int(plan["n_rounds"]),
        "n_learners": plan.n_learners,
        "n_constant": plan.n_constant,
        "constant_share": plan.constant_share,
        "n_evaluated": plan.n_evaluated,
        "degenerate": plan.n_learners > 0 and plan.n_constant == plan.n_learners,
        "per_param": per_param,
    }


def degenerate_message(report):
    """Plain-English explanation of a degenerate fit, including the bound when it is known."""
    msg = (
        f"All {report['n_learners']} base learners of this NGBoost model are constant: it predicts "
        "the same distribution for every input."
    )
    infeasible = [r for r in report["per_param"] if r.get("feasible") is False]
    if infeasible:
        r = infeasible[0]
        msg += (
            f" Each round's learners were fitted on {r['batch_rows']} rows, but a split needs at "
            f"least {r['min_rows']} (max(min_samples_split, 2 * min_samples_leaf)). Lower "
            "min_samples_leaf or raise minibatch_frac."
        )
    else:
        msg += " No split improved the fit; the input carried no signal the base learner could use."
    return msg


def warn_if_degenerate(model, n_train=None, stacklevel=3):
    """Issue NGBoostDegenerateWarning when every learner is constant; return the report."""
    report = diagnose(model, n_train=n_train)
    if report["degenerate"]:
        warnings.warn(degenerate_message(report), NGBoostDegenerateWarning, stacklevel=stacklevel)
    return report

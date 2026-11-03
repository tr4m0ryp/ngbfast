"""The fold plan: a precomputed, order-preserving description of an NGBoost model's prediction.

For each distribution parameter (column) ``k`` the plan stores

* ``chains[k]``: the parameter value after 0, 1, ..., f_k leading rounds, where f_k is the first round
  whose learner for ``k`` depends on the input (f_k = number of rounds if none does). Every value is
  the upstream subtraction chain computed once on a scalar instead of once per row.
* ``rounds``: for every round with work left, the entries ``(k, step, direct)``. ``step`` is the
  scalar ``(learning_rate * v) * scale`` of a constant learner after f_k, or ``None`` for a learner
  that must be evaluated (``direct`` says whether it is a tree that can skip input validation).

The plan is a ``dict`` holding only numpy arrays and Python builtins, and it pickles as a plain
``dict``. A model carrying a plan can therefore be unpickled where ngbfast is not installed; upstream
ngboost ignores the extra attribute and predicts with its loop.
"""

import numpy as np

PLAN_VERSION = 1


class FoldPlan(dict):
    """Dictionary with attribute access to the plan fields; see the module docstring."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __reduce__(self):
        return (dict, (dict(self),))

    @property
    def start(self):
        """Parameter values when every leading constant round has been applied, shape (n_params,)."""
        return np.array([chain[-1] for chain in self["chains"]], dtype=np.float64)

    def start_at(self, n_rounds):
        """Parameter values after the first ``n_rounds`` rounds, for columns still in their prefix."""
        return np.array(
            [chain[min(n_rounds, f)] for chain, f in zip(self["chains"], self["first_split"])],
            dtype=np.float64,
        )

    @property
    def n_learners(self):
        return int(self["n_rounds"] * self["n_params"])

    @property
    def n_constant(self):
        return int(np.sum(self["n_constant_per_param"]))

    @property
    def constant_share(self):
        return self.n_constant / self.n_learners if self.n_learners else 0.0

    @property
    def n_evaluated(self):
        """Learners the folded path still calls per prediction."""
        return sum(step is None for _, entries, _ in self["rounds"] for _, step, _ in entries)

    def __repr__(self):
        return (
            f"FoldPlan(n_rounds={self['n_rounds']}, n_params={self['n_params']}, "
            f"n_constant={self.n_constant}/{self.n_learners}, n_evaluated={self.n_evaluated})"
        )


def as_plan(obj):
    """Wrap a plain dict (for example an unpickled plan) as a FoldPlan; None stays None."""
    if obj is None or isinstance(obj, FoldPlan):
        return obj
    if isinstance(obj, dict) and obj.get("version") == PLAN_VERSION:
        return FoldPlan(obj)
    return None


def plan_matches(plan, model):
    """True when ``plan`` was built from the model's current rounds and initial parameters."""
    return (
        plan is not None
        and plan["n_rounds"] == len(model.base_models)
        and plan["n_rounds"] == len(model.scalings)
        and model.init_params is not None
        and np.array_equal(plan["init_params"], np.asarray(model.init_params, dtype=np.float64))
    )

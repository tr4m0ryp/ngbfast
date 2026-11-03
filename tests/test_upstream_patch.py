"""The upstream-ready patch applies to the installed ngboost release and keeps outputs bit-identical."""

import shutil
import subprocess
import sys
from pathlib import Path

import ngboost
import pytest

PATCH = Path(__file__).resolve().parents[1] / "patches" / "ngboost-fold-constant-trees.patch"

CHECK = """
import warnings, numpy as np, ngboost
from ngboost import NGBRegressor
from ngboost.ngboost import NGBoostDegenerateWarning
from sklearn.tree import DecisionTreeRegressor
import ngbfast
assert ngboost.__file__.startswith({root!r}), ngboost.__file__
rng = np.random.RandomState(0)
X = rng.normal(size=(300, 4)); Y = X[:, 0] + rng.normal(size=300)
for leaf, frac in ((10, 1.0), (60, 0.5), (100, 0.5)):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = NGBRegressor(Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=leaf), n_estimators=60,
                         minibatch_frac=frac, verbose=False, random_state=0).fit(X, Y)
    assert m.fold_ is not None
    a = np.asarray(m.pred_param(X)).view(np.uint64)
    b = ngbfast.loop_pred_param(m, X).view(np.uint64)
    assert (a == b).all()
    degenerate = any(issubclass(x.category, NGBoostDegenerateWarning) for x in w)
    assert degenerate == ngbfast.diagnose(m)["degenerate"], (leaf, frac)
print("ok")
"""


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_patch_applies_and_is_exact(tmp_path):
    assert ngboost.__version__ == "0.5.11", "the patch targets ngboost 0.5.11"
    src = Path(ngboost.__file__).parent
    shutil.copytree(src, tmp_path / "ngboost", ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(
        ["git", "apply", "--exclude=tests/*", str(PATCH)], cwd=tmp_path, check=True,
        capture_output=True,
    )
    code = f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + CHECK.format(root=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"

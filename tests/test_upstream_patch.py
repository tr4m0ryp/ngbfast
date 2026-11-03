"""The upstream-ready patch applies to ngboost master 662aae3 and keeps outputs bit-identical.

The patch targets upstream master, not the 0.5.11 release installed for the benchmarks (the two
differ in ``ngboost/ngboost.py``). The test needs a local ngboost git clone that holds the base
commit, given by ``NGBFAST_NGBOOST_SRC`` (default ``../_upstream/ngboost``); without one it skips.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "ngboost-check-input-once.patch"
BASE = "662aae3"
SRC = Path(os.environ.get("NGBFAST_NGBOOST_SRC", ROOT.parent / "_upstream" / "ngboost"))

CHECK = """
import sys, warnings, numpy as np
sys.path.insert(0, {root!r})
import ngboost
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
assert ngboost.__file__.startswith({root!r}), ngboost.__file__
rng = np.random.RandomState(0)
X = rng.normal(size=(300, 4)); Y = X[:, 0] + rng.normal(size=300)
for leaf, frac in ((10, 1.0), (60, 0.5), (100, 0.5)):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = NGBRegressor(Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=leaf), n_estimators=60,
                         minibatch_frac=frac, verbose=False, random_state=0).fit(X, Y)
    bits = np.asarray(m.pred_param(X)).view(np.uint64)
    warned = any(x.category.__name__ == "NGBoostDegenerateWarning" for x in w)
    print(leaf, frac, int(warned), bits.tobytes().hex())
"""


def _has_base():
    if shutil.which("git") is None or not (SRC / ".git").exists():
        return False
    ok = subprocess.run(["git", "-C", str(SRC), "cat-file", "-e", BASE + "^{commit}"], capture_output=True)
    return ok.returncode == 0


def _export(dest):
    dest.mkdir()
    archive = subprocess.run(["git", "-C", str(SRC), "archive", BASE, "ngboost"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


def _run(root):
    out = subprocess.run([sys.executable, "-c", CHECK.format(root=str(root))], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [line.split() for line in out.stdout.strip().splitlines()]


@pytest.mark.skipif(not _has_base(), reason=f"the patch targets ngboost master {BASE} and does not apply to the "
                    "0.5.11 release; set NGBFAST_NGBOOST_SRC to an ngboost git clone holding that commit")
def test_patch_applies_and_is_exact(tmp_path):
    pristine, patched = tmp_path / "pristine", tmp_path / "patched"
    _export(pristine)
    _export(patched)
    subprocess.run(["git", "apply", "--exclude=tests/*", str(PATCH)], cwd=patched, check=True, capture_output=True)
    before, after = _run(pristine), _run(patched)
    for (leaf, frac, warned_before, bits_before), (_, _, warned_after, bits_after) in zip(before, after):
        assert bits_before == bits_after, (leaf, frac)
        assert warned_before == "0"
        assert warned_after == ("1" if leaf == "100" else "0"), (leaf, frac)

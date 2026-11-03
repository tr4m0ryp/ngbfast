"""Plans pickle as plain dicts: models load and predict where ngbfast is not installed."""

import pickle
import subprocess
import sys

from helpers import assert_bitwise, const_tree, ngb, split_tree, toy

import ngbfast
from ngbfast import FoldPlan, loop_pred_param


def test_roundtrip_and_upstream_pickle():
    X, Y = toy()
    m = ngb([split_tree(), const_tree()], n=15).fit(X, Y)
    ngbfast.fold(m)
    m2 = pickle.loads(pickle.dumps(m))
    assert type(m2.fold_) is dict  # no ngbfast class inside the pickle
    assert isinstance(ngbfast.current_plan(m2), FoldPlan)
    assert_bitwise(ngbfast.pred_param(m2, X), loop_pred_param(m, X))
    del m2.fold_  # a model pickled by upstream ngboost: no plan, the loop runs
    assert ngbfast.current_plan(m2) is None
    assert_bitwise(ngbfast.pred_param(m2, X), loop_pred_param(m, X))


def test_loads_without_ngbfast(tmp_path):
    X, Y = toy()
    m = ngb(const_tree(), n=10, Dist=__import__("ngboost").distns.Normal).fit(X, Y)
    ngbfast.fold(m)
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps(m))
    code = (
        "import sys, pickle, numpy as np\n"
        "sys.modules['ngbfast'] = None\n"
        f"m = pickle.loads(open({str(path)!r}, 'rb').read())\n"
        "print(type(m.fold_).__name__, m.pred_param(np.zeros((2, 4))).shape)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["dict", "(2,", "2)"]

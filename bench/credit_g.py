"""OpenML credit-g (Bernoulli classification, stratified 90/10 splits): `python -m bench.credit_g`."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from bench.data import load_credit_g  # noqa: E402
from bench.run import main  # noqa: E402

if __name__ == "__main__":
    main(load_credit_g)

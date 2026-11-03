"""UCI concrete (Normal regression, test masks of the uci_datasets collection): `python -m bench.concrete`."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from bench.data import load_uci  # noqa: E402
from bench.run import main  # noqa: E402

if __name__ == "__main__":
    main(lambda: load_uci("concrete"))

import os
from pathlib import Path

import pytest

DATA_ROOT = Path(os.environ.get("QUAML_DATA_ROOT", "C:/repos/conetworks/data/raw"))


@pytest.fixture(scope="session")
def data_root() -> Path:
    if not DATA_ROOT.exists():
        pytest.skip(f"market data not found at {DATA_ROOT} (set QUAML_DATA_ROOT)")
    return DATA_ROOT


def first_file(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern))
    if not files:
        pytest.skip(f"no files match {root / pattern}")
    return files[0]

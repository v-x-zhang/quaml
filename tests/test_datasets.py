import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("quaml._core")

from quaml.datasets import WINDOW_CHANNELS, EventWindowDataset, Sessions, load_sessions
from quaml.windows import INPUT_NAMES

from conftest import first_file


def tiny_sessions():
    # Two sessions of the same contract; windows must not cross between them.
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    ts = np.array([0, 1, 2, 0, 1, 2]) * 10**9
    group = np.array([0, 0, 0, 1 << 32, 1 << 32, 1 << 32], dtype=np.int64)
    y = np.array([0.1, 0.2, np.nan, 0.3, 0.4, np.nan])
    return Sessions(X, ts, group, np.array([0, 0, 0, 1, 1, 1]), y, [])


def test_batch_layout_and_padding():
    ds = EventWindowDataset(tiny_sessions(), anchors=np.array([1, 3]), length=3)
    b = ds.__getitems__([0, 1])
    assert b["x"].shape == (2, 3, 2 + 2)
    np.testing.assert_array_equal(b["mask"], [[0, 1, 1], [0, 0, 1]])
    np.testing.assert_allclose(b["x"][0, 1], [0, 1, np.log1p(1.0), 1])  # row 0, 1s before anchor
    np.testing.assert_allclose(b["x"][0, 2], [2, 3, 0, 1])              # the anchor itself
    assert torch.all(b["x"][1, :2] == 0)  # row 3 is the first of session 1: no crossover
    np.testing.assert_allclose(b["y"], [0.2, 0.3])


def test_rejects_unlabelled_anchors():
    with pytest.raises(ValueError):
        EventWindowDataset(tiny_sessions(), anchors=np.array([2]), length=3)


def test_loader_on_real_session(data_root):
    path = first_file(data_root, "options/DIA/tcbbo_clean/*.tcbbo.dbn.zst")
    data = load_sessions([path], window_ns=10**9, horizon_ns=30 * 10**9)
    assert data.X.shape == (len(data.ts), len(INPUT_NAMES))
    anchors = np.flatnonzero(np.isfinite(data.y))
    ds = EventWindowDataset(data, anchors, length=32)
    batch = next(iter(ds.loader(batch_size=64, shuffle=True)))
    assert batch["x"].shape == (64, 32, len(WINDOW_CHANNELS))
    assert torch.isfinite(batch["x"]).all()
    assert batch["mask"][:, -1].all()

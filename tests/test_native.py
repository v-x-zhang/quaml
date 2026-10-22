"""Differential tests: the C++ engine must reproduce the NumPy reference."""

from dataclasses import fields

import numpy as np
import pytest

pytest.importorskip("quaml._core")

import quaml
from quaml import native
from quaml.reference import features, rolling

from conftest import first_file

ONE_SECOND = 1_000_000_000


@pytest.mark.parametrize("seed", range(10))
def test_rolling_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 2_000
    ts = np.sort(rng.integers(0, 10_000, n))
    v = rng.normal(size=n)
    v[rng.random(n) < 0.1] = np.nan
    g = rng.integers(0, 20, n) * 1_000_003  # sparse ids
    w = int(rng.integers(1, 2_000))
    s, c = native.rolling_sum_count(ts, v, g, w)
    rs, rc = rolling.rolling_sum_count(ts, v, g, w)
    np.testing.assert_allclose(s, rs, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(c, rc)


def test_rolling_rejects_bad_input():
    with pytest.raises(ValueError):
        native.rolling_sum_count([2, 1], [1.0, 1.0], [0, 0], 5)
    with pytest.raises(ValueError):
        native.rolling_sum_count([1, 2], [1.0, 1.0], [0, 0], 0)


def test_empty_input():
    s, c = native.rolling_sum_count(np.array([], np.int64), np.array([]), np.array([], np.uint32), 5)
    assert len(s) == 0 and len(c) == 0


@pytest.fixture(scope="module", params=[
    "options/DIA/tcbbo_clean/*.tcbbo.dbn.zst",
    "equities/SPY/tbbo/*.tbbo.dbn.zst",
])
def dbn_path(request, data_root):
    return first_file(data_root, request.param)


def test_reader_matches_databento(dbn_path):
    ref = quaml.read_dbn(dbn_path)
    nat = native.read_dbn(dbn_path)
    for f in fields(ref):
        a, b = getattr(ref, f.name), getattr(nat, f.name)
        assert a.dtype == b.dtype, f.name
        np.testing.assert_array_equal(a, b, err_msg=f.name)


def test_features_match_reference(dbn_path):
    ev = quaml.read_dbn(dbn_path)
    ref = features.trade_quote_features(ev, ONE_SECOND)
    nat = native.trade_quote_features(ev, ONE_SECOND)
    assert list(nat) == list(ref)
    for name in ref:
        np.testing.assert_array_equal(nat[name], ref[name], err_msg=name)

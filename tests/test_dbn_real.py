"""Checks against real Databento files. Skipped when the data is not present."""

import numpy as np
import pytest

import quaml
from quaml.reference import features

from conftest import first_file

pytest.importorskip("databento")

ONE_SECOND = 1_000_000_000


@pytest.fixture(scope="module", params=[
    "options/DIA/tcbbo_clean/*.tcbbo.dbn.zst",
    "equities/SPY/tbbo/*.tbbo.dbn.zst",
])
def events(request, data_root):
    return quaml.read_dbn(first_file(data_root, request.param))


def test_events_are_well_formed(events):
    assert len(events) > 0
    assert np.all(np.diff(events.ts_recv) >= 0)
    assert np.all(events.price > 0)
    # Capture happens after the venue event.
    assert np.all(events.ts_recv >= events.ts_event)


def test_features_respect_quote_bounds(events):
    f = features.trade_quote_features(events, ONE_SECOND)
    ok = np.isfinite(f["microprice"])
    assert ok.any()
    assert np.all(f["microprice"][ok] >= events.bid_px[ok] - 1e-9)
    assert np.all(f["microprice"][ok] <= events.ask_px[ok] + 1e-9)
    assert np.all(f["trade_count_w"] >= 1)  # the current trade is always in its window
    fi = f["flow_imbalance_w"][np.isfinite(f["flow_imbalance_w"])]
    assert np.all(np.abs(fi) <= 1 + 1e-12)


def test_rolling_volume_spot_check(events):
    """Recompute the 1s windowed volume directly for a sample of rows."""
    f = features.trade_quote_features(events, ONE_SECOND)
    rng = np.random.default_rng(0)
    ts, inst, size = events.ts_recv, events.instrument_id, events.size.astype(float)
    for i in rng.choice(len(events), size=min(200, len(events)), replace=False):
        j = np.arange(i + 1)
        mask = (inst[j] == inst[i]) & (ts[j] > ts[i] - ONE_SECOND)
        assert f["volume_w"][i] == pytest.approx(size[j][mask].sum())

import numpy as np
import pytest

import quaml
from quaml.windows import GroupIndex, forward_mid_return, model_inputs


def test_windows_are_left_padded_and_end_at_anchor():
    group = np.array([7, 9, 7, 7, 9, 7])
    gi = GroupIndex(group)
    idx, mask = gi.windows(np.array([5, 4, 0]), length=3)
    # group 7 rows: 0, 2, 3, 5; group 9 rows: 1, 4
    np.testing.assert_array_equal(idx, [[2, 3, 5], [4, 1, 4], [0, 0, 0]])
    np.testing.assert_array_equal(mask, [[1, 1, 1], [0, 1, 1], [0, 0, 1]])


def test_forward_mid_return():
    ts = np.array([0, 10, 20, 35, 100])
    mid = np.array([1.0, 2.0, 4.0, 8.0, 1.0])
    group = np.array([0, 0, 1, 0, 0])
    y = forward_mid_return(ts, mid, GroupIndex(group), horizon_ns=30)
    # row 0: last group-0 row in (0, 30] is row 1. row 1: (10, 40] -> row 3.
    # row 2: no other group-1 rows. row 3: (35, 65] -> none. row 4: none.
    np.testing.assert_allclose(y, [np.log(2), np.log(4), np.nan, np.nan, np.nan])
    # From the next quote: row 0 -> log(mid1/mid1) = 0; row 1 -> log(mid3/mid3) = 0.
    y_next = forward_mid_return(ts, mid, GroupIndex(group), horizon_ns=30, start="next")
    np.testing.assert_allclose(y_next, [0, 0, np.nan, np.nan, np.nan])
    y_next = forward_mid_return(ts, mid, GroupIndex(group), horizon_ns=40, start="next")
    np.testing.assert_allclose(y_next[0], np.log(8 / 2))  # row 0: next is row 1, horizon row 3


def random_events(n, seed):
    rng = np.random.default_rng(seed)
    bid = np.round(rng.uniform(1, 5, n), 2)
    ask = bid + rng.choice([0.01, 0.02, 0.05], n)
    return quaml.TradeQuoteEvents(
        instrument_id=rng.integers(0, 5, n).astype(np.uint32),
        ts_event=np.sort(rng.integers(0, 10**10, n)),
        ts_recv=np.sort(rng.integers(0, 10**10, n)),
        price=np.where(rng.random(n) < 0.5, bid, ask),
        size=rng.integers(1, 50, n).astype(np.uint32),
        side=np.zeros(n, dtype=np.int8),
        bid_px=bid, ask_px=ask,
        bid_sz=rng.integers(0, 100, n).astype(np.uint32),
        ask_sz=rng.integers(0, 100, n).astype(np.uint32),
    )


def test_no_lookahead_in_inputs_or_windows():
    """Rewriting every row after a cut must not change any window up to the cut."""
    pytest.importorskip("quaml._core")
    from quaml import native

    ev = random_events(5_000, seed=1)
    cut = 3_000
    rng = np.random.default_rng(2)
    later = slice(cut + 1, None)
    tampered = quaml.TradeQuoteEvents(**{
        **{f: getattr(ev, f) for f in ev.__dataclass_fields__},
        "price": np.r_[ev.price[:cut + 1], rng.uniform(1, 5, len(ev) - cut - 1)],
        "size": np.r_[ev.size[:cut + 1], ev.size[later][::-1]],
        "bid_sz": np.r_[ev.bid_sz[:cut + 1], ev.ask_sz[later]],
    })
    window_ns = 10**8
    X1 = model_inputs(ev, native.trade_quote_features(ev, window_ns))
    X2 = model_inputs(tampered, native.trade_quote_features(tampered, window_ns))
    gi = GroupIndex(ev.instrument_id)
    anchors = np.arange(cut + 1)
    idx, mask = gi.windows(anchors, length=16)
    assert idx.max() <= cut
    np.testing.assert_array_equal(X1[idx], X2[idx])
    assert not np.array_equal(X1[cut + 1:], X2[cut + 1:])  # the tamper did something

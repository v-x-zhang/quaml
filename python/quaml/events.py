"""Columnar event store for trade-with-quote records (Databento TBBO / TCBBO).

Time semantics
--------------
Every event carries two timestamps, both integer nanoseconds since the epoch:

* ``ts_event`` -- when the venue says the event happened.
* ``ts_recv``  -- when the capture server received it. This is the
  *availability* time: a strategy could not have known about the event before
  ``ts_recv``. Events are stored sorted by ``ts_recv`` and every feature in
  :mod:`quaml.reference` is computed on that clock.

``ts_event`` is not monotonic in OPRA data (records arrive out of venue order),
so it must never be used to order or window events.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

# Databento fixed-point prices: 1 unit = 1e-9.
PRICE_SCALE = 1e-9
UNDEF_PRICE = np.iinfo(np.int64).max

SIDE_BUY = 1
SIDE_SELL = -1
SIDE_NONE = 0


@dataclass(frozen=True)
class TradeQuoteEvents:
    """One trade per row, with the best bid/offer just before the trade.

    Undefined quote prices are NaN. ``side`` is the aggressor side as reported
    by the feed (+1 buy, -1 sell, 0 unknown); see
    :func:`quaml.reference.signing.sign_trades` to fill in unknowns.
    """

    instrument_id: np.ndarray  # uint32
    ts_event: np.ndarray       # int64 ns
    ts_recv: np.ndarray        # int64 ns, non-decreasing
    price: np.ndarray          # float64
    size: np.ndarray           # uint32
    side: np.ndarray           # int8
    bid_px: np.ndarray         # float64, NaN if undefined
    ask_px: np.ndarray         # float64, NaN if undefined
    bid_sz: np.ndarray         # uint32
    ask_sz: np.ndarray         # uint32

    def __post_init__(self):
        n = len(self.ts_recv)
        for f in fields(self):
            if len(getattr(self, f.name)) != n:
                raise ValueError(f"column {f.name} has length "
                                 f"{len(getattr(self, f.name))}, expected {n}")
        if n > 1 and np.any(np.diff(self.ts_recv) < 0):
            raise ValueError("events must be sorted by ts_recv")

    def __len__(self) -> int:
        return len(self.ts_recv)


def _fixed_to_float(px: np.ndarray) -> np.ndarray:
    out = px.astype(np.float64) * PRICE_SCALE
    out[px == UNDEF_PRICE] = np.nan
    return out


def _side_to_int(side: np.ndarray) -> np.ndarray:
    out = np.zeros(len(side), dtype=np.int8)
    out[side == b"B"] = SIDE_BUY
    out[side == b"A"] = SIDE_SELL
    return out


def from_dbn_records(rec: np.ndarray) -> TradeQuoteEvents:
    """Convert a structured array of TBBO/TCBBO records (``DBNStore.to_ndarray()``)."""
    order = np.argsort(rec["ts_recv"], kind="stable")
    rec = rec[order]
    return TradeQuoteEvents(
        instrument_id=rec["instrument_id"].astype(np.uint32),
        ts_event=rec["ts_event"].astype(np.int64),
        ts_recv=rec["ts_recv"].astype(np.int64),
        price=_fixed_to_float(rec["price"]),
        size=rec["size"].astype(np.uint32),
        side=_side_to_int(rec["side"]),
        bid_px=_fixed_to_float(rec["bid_px_00"]),
        ask_px=_fixed_to_float(rec["ask_px_00"]),
        bid_sz=rec["bid_sz_00"].astype(np.uint32),
        ask_sz=rec["ask_sz_00"].astype(np.uint32),
    )


def read_dbn(path: str | Path) -> TradeQuoteEvents:
    """Read a ``.tbbo.dbn.zst`` or ``.tcbbo.dbn.zst`` file via the databento package."""
    import databento as db

    store = db.DBNStore.from_file(str(path))
    if store.schema not in (db.Schema.TBBO, db.Schema.TCBBO):
        raise ValueError(f"{path}: expected tbbo or tcbbo, got {store.schema}")
    return from_dbn_records(store.to_ndarray())

"""C++ implementations with the same signatures as their reference counterparts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _core
from .events import TradeQuoteEvents

_FEATURE_INPUTS = ("instrument_id", "ts_recv", "price", "size", "side",
                   "bid_px", "ask_px", "bid_sz", "ask_sz")


def read_dbn(path: str | Path) -> TradeQuoteEvents:
    """Native equivalent of :func:`quaml.events.read_dbn`."""
    return TradeQuoteEvents(**_core.read_trade_quotes(str(path)))


def trade_quote_features(ev: TradeQuoteEvents, window_ns: int) -> dict[str, np.ndarray]:
    """Native equivalent of :func:`quaml.reference.features.trade_quote_features`."""
    return _core.trade_quote_features(*(getattr(ev, c) for c in _FEATURE_INPUTS), int(window_ns))


def rolling_sum_count(ts, values, group, window_ns: int):
    """Native equivalent of :func:`quaml.reference.rolling.rolling_sum_count`."""
    _, dense = np.unique(np.asarray(group), return_inverse=True)
    return _core.rolling_sum_count(np.ascontiguousarray(ts, dtype=np.int64),
                                   np.ascontiguousarray(values, dtype=np.float64),
                                   np.ascontiguousarray(dense, dtype=np.uint32),
                                   int(window_ns))

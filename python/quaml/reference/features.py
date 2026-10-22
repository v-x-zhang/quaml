"""Composite per-event features built from the reference primitives."""

from __future__ import annotations

import numpy as np

from ..events import TradeQuoteEvents
from . import quote, rolling, signing


def trade_quote_features(ev: TradeQuoteEvents, window_ns: int) -> dict[str, np.ndarray]:
    """Per-event features, each available at ``ev.ts_recv[i]``.

    Rolling features are per instrument over ``(t - window_ns, t]``.
    """
    sign = signing.sign_trades(ev.side, ev.price, ev.bid_px, ev.ask_px)
    size = ev.size.astype(np.float64)
    signed_vol = sign * size

    vol_sum, n_trades = rolling.rolling_sum_count(ev.ts_recv, size, ev.instrument_id, window_ns)
    signed_sum = rolling.rolling_sum(ev.ts_recv, signed_vol, ev.instrument_id, window_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_imbalance = np.where(vol_sum > 0, signed_sum / vol_sum, np.nan)

    return {
        "mid": quote.mid(ev.bid_px, ev.ask_px),
        "spread": quote.spread(ev.bid_px, ev.ask_px),
        "relative_spread": quote.relative_spread(ev.bid_px, ev.ask_px),
        "microprice": quote.microprice(ev.bid_px, ev.ask_px, ev.bid_sz, ev.ask_sz),
        "book_imbalance": quote.book_imbalance(ev.bid_sz, ev.ask_sz),
        "sign": sign.astype(np.float64),
        "signed_volume_w": signed_sum,
        "volume_w": vol_sum,
        "trade_count_w": n_trades.astype(np.float64),
        "flow_imbalance_w": flow_imbalance,
        "time_since_prev": rolling.time_since_prev(ev.ts_recv, ev.instrument_id),
    }

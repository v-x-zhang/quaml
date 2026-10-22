"""Trade signing: +1 buyer-initiated, -1 seller-initiated, 0 unknown."""

from __future__ import annotations

import numpy as np

from .quote import mid

# Prices sit on Databento's 1e-9 fixed-point grid, so anything closer than this
# to the mid is float rounding, not a real price difference.
MID_TOLERANCE = 1e-10


def quote_rule(price, bid_px, ask_px) -> np.ndarray:
    """+1 if the trade printed above the mid, -1 below, 0 at the mid or with no
    valid quote."""
    d = price - mid(bid_px, ask_px)
    out = np.zeros(len(price), dtype=np.int8)
    out[d > MID_TOLERANCE] = 1
    out[d < -MID_TOLERANCE] = -1
    return out


def sign_trades(side, price, bid_px, ask_px) -> np.ndarray:
    """Use the feed's aggressor side where it is known; fall back to the quote
    rule where it is not (all of OPRA, and part of equity TBBO)."""
    return np.where(side != 0, side, quote_rule(price, bid_px, ask_px)).astype(np.int8)

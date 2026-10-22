"""Top-of-book quote features. Each returns one value per event; NaN where the
quote is undefined, crossed (bid > ask), or has no displayed size."""

from __future__ import annotations

import numpy as np


def _valid(bid_px, ask_px):
    return np.isfinite(bid_px) & np.isfinite(ask_px) & (bid_px <= ask_px)


def mid(bid_px: np.ndarray, ask_px: np.ndarray) -> np.ndarray:
    return np.where(_valid(bid_px, ask_px), 0.5 * (bid_px + ask_px), np.nan)


def spread(bid_px: np.ndarray, ask_px: np.ndarray) -> np.ndarray:
    return np.where(_valid(bid_px, ask_px), ask_px - bid_px, np.nan)


def relative_spread(bid_px: np.ndarray, ask_px: np.ndarray) -> np.ndarray:
    """Spread divided by mid. NaN when mid is not positive."""
    m = mid(bid_px, ask_px)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, spread(bid_px, ask_px) / m, np.nan)


def microprice(bid_px, ask_px, bid_sz, ask_sz) -> np.ndarray:
    """Size-weighted mid: (ask * bid_sz + bid * ask_sz) / (bid_sz + ask_sz)."""
    bid_sz = bid_sz.astype(np.float64)
    ask_sz = ask_sz.astype(np.float64)
    total = bid_sz + ask_sz
    ok = _valid(bid_px, ask_px) & (total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mp = (ask_px * bid_sz + bid_px * ask_sz) / total
    return np.where(ok, mp, np.nan)


def book_imbalance(bid_sz, ask_sz) -> np.ndarray:
    """(bid_sz - ask_sz) / (bid_sz + ask_sz), in [-1, 1]. NaN when both are zero."""
    bid_sz = bid_sz.astype(np.float64)
    ask_sz = ask_sz.astype(np.float64)
    total = bid_sz + ask_sz
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (bid_sz - ask_sz) / total, np.nan)

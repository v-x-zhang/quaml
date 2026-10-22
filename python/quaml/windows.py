"""Event-window sampling and forward labels for sequence models.

A *window* for anchor row ``i`` is the last ``length`` rows of the same group
(contract) ending at ``i`` itself, in ``ts_recv`` order. Rows before the start
of the group are padding (``mask == False``). Because every per-row feature
only depends on rows ``<= j``, a window never contains information from after
the anchor.

Labels look forward on the same clock: see :func:`forward_mid_return`.
"""

from __future__ import annotations

import numpy as np

from .events import TradeQuoteEvents


class GroupIndex:
    """Per-row position of each event within its group, for O(1) window lookup."""

    def __init__(self, group: np.ndarray):
        group = np.asarray(group)
        n = len(group)
        self.order = np.argsort(group, kind="stable")  # rows grouped, time order kept
        sorted_group = group[self.order]
        self.starts = np.flatnonzero(np.r_[True, sorted_group[1:] != sorted_group[:-1]]) if n else \
            np.zeros(0, dtype=np.int64)
        self.ends = np.r_[self.starts[1:], n].astype(np.int64)

        segment = np.zeros(n, dtype=np.int64)
        segment[self.starts[1:]] = 1
        segment = np.cumsum(segment)
        start_sorted = self.starts[segment]
        self.start = np.empty(n, dtype=np.int64)
        self.start[self.order] = start_sorted
        self.rank = np.empty(n, dtype=np.int64)
        self.rank[self.order] = np.arange(n) - start_sorted

    def windows(self, anchors: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Row indices ``[B, length]`` of each anchor's window, left-padded.

        Padding slots hold the anchor's own row index (so gathers stay in
        bounds) and are ``False`` in the returned mask. The anchor is always
        the last slot.
        """
        anchors = np.asarray(anchors, dtype=np.int64)
        pos = self.rank[anchors][:, None] + np.arange(1 - length, 1)
        mask = pos >= 0
        idx = self.order[self.start[anchors][:, None] + np.maximum(pos, 0)]
        return np.where(mask, idx, anchors[:, None]), mask


def forward_mid_return(ts: np.ndarray, mid: np.ndarray, gi: GroupIndex,
                       horizon_ns: int, start: str = "anchor") -> np.ndarray:
    """``log(mid[j] / mid[s])`` where ``j`` is the last row of the same group
    with ``ts[j]`` in ``(ts[i], ts[i] + horizon_ns]``.

    ``mid[j]`` is the quote standing just before trade ``j``, so it was
    observable by ``ts[i] + horizon_ns``. NaN when the group has no later row
    inside the horizon or either mid is undefined.

    ``start`` picks the reference quote ``s``:

    * ``"anchor"`` -- ``s = i``. TBBO/TCBBO quotes are *pre-trade*, so this
      includes the quote's reaction to trade ``i`` itself, which has already
      happened at decision time but is not yet visible in the data.
    * ``"next"`` -- ``s = i + 1`` in the group (the first quote observed after
      the anchor), which excludes that reaction. If that next row is also the
      last one inside the horizon, the label is 0.
    """
    if start not in ("anchor", "next"):
        raise ValueError(f"start must be 'anchor' or 'next', got {start!r}")
    out = np.full(len(ts), np.nan)
    for s, e in zip(gi.starts, gi.ends):
        rows = gi.order[s:e]
        t = ts[rows]
        k = np.arange(len(rows))
        j = np.searchsorted(t, t + horizon_ns, side="right") - 1
        ref = k if start == "anchor" else np.minimum(k + 1, len(rows) - 1)
        m0 = mid[rows[ref]]
        m1 = mid[rows[j]]
        ok = (j > k) & (m0 > 0) & (m1 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[rows[ok]] = np.log(m1[ok] / m0[ok])
    return out


INPUT_NAMES = (
    "relative_spread",
    "book_imbalance",     # (microprice - mid) / spread is exactly half this

    "sign",
    "trade_vs_mid",       # (price - mid) / mid
    "flow_imbalance_w",
    "log_volume_w",
    "log_trade_count_w",
    "log_size",
    "log_gap_s",          # log1p(seconds since previous trade in the contract)
)


def model_inputs(ev: TradeQuoteEvents, feats: dict[str, np.ndarray]) -> np.ndarray:
    """Scale-free per-row inputs ``[N, len(INPUT_NAMES)]`` as float32.

    Raw prices are left out so contracts at different price levels are
    comparable. Undefined values become 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = {
            "relative_spread": feats["relative_spread"],
            "book_imbalance": feats["book_imbalance"],
            "sign": feats["sign"],
            "trade_vs_mid": (ev.price - feats["mid"]) / feats["mid"],
            "flow_imbalance_w": feats["flow_imbalance_w"],
            "log_volume_w": np.log1p(feats["volume_w"]),
            "log_trade_count_w": np.log1p(feats["trade_count_w"]),
            "log_size": np.log1p(ev.size.astype(np.float64)),
            "log_gap_s": np.log1p(feats["time_since_prev"] * 1e-9),
        }
    X = np.stack([cols[name] for name in INPUT_NAMES], axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

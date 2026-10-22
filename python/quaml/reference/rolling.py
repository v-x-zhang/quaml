"""Time-windowed rolling statistics over irregular event streams.

Window semantics (the C++ engine must match these exactly)
----------------------------------------------------------
Events are ordered by ``ts`` (non-decreasing) and grouped by ``group``
(typically ``instrument_id``). For event ``i`` the window holds every event
``j`` in the same group with

    j <= i  and  ts[j] > ts[i] - window_ns

i.e. a half-open interval ``(t - w, t]`` in time, truncated at the current
event in sequence. Events sharing a timestamp with ``i`` but stored after it
are excluded, so a feature at row ``i`` never depends on a later row.

NaN values are treated as missing: they add nothing to sums and are not
counted. A window with no valid values yields NaN for means.
"""

from __future__ import annotations

import numpy as np


def _group_slices(group: np.ndarray):
    """Row indices of each group, in their original (time) order."""
    order = np.argsort(group, kind="stable")
    sorted_groups = group[order]
    bounds = np.flatnonzero(np.diff(sorted_groups)) + 1
    return np.split(order, bounds)


def _window_left(ts: np.ndarray, window_ns: int) -> np.ndarray:
    """For sorted ts, index of the first element inside (ts[i] - w, ts[i]]."""
    return np.searchsorted(ts, ts - window_ns, side="right")


def _check(ts, values, group, window_ns):
    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if len(ts) != len(values) or len(ts) != len(group):
        raise ValueError("ts, values and group must have equal length")
    if len(ts) > 1 and np.any(np.diff(ts) < 0):
        raise ValueError("ts must be non-decreasing")


def rolling_sum_count(ts, values, group, window_ns: int):
    """Return (sum, count) of non-NaN values in each event's window."""
    ts = np.asarray(ts, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    group = np.asarray(group)
    _check(ts, values, group, window_ns)

    out_sum = np.empty(len(ts), dtype=np.float64)
    out_cnt = np.empty(len(ts), dtype=np.int64)
    for idx in _group_slices(group):
        t = ts[idx]
        v = values[idx]
        valid = ~np.isnan(v)
        csum = np.concatenate(([0.0], np.cumsum(np.where(valid, v, 0.0))))
        ccnt = np.concatenate(([0], np.cumsum(valid)))
        left = _window_left(t, window_ns)
        right = np.arange(1, len(t) + 1)
        out_sum[idx] = csum[right] - csum[left]
        out_cnt[idx] = ccnt[right] - ccnt[left]
    return out_sum, out_cnt


def rolling_sum(ts, values, group, window_ns: int) -> np.ndarray:
    return rolling_sum_count(ts, values, group, window_ns)[0]


def rolling_count(ts, values, group, window_ns: int) -> np.ndarray:
    return rolling_sum_count(ts, values, group, window_ns)[1]


def rolling_mean(ts, values, group, window_ns: int) -> np.ndarray:
    s, c = rolling_sum_count(ts, values, group, window_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0, s / c, np.nan)


def time_since_prev(ts, group) -> np.ndarray:
    """Nanoseconds since the previous event in the same group; NaN for the first."""
    ts = np.asarray(ts, dtype=np.int64)
    group = np.asarray(group)
    out = np.full(len(ts), np.nan)
    for idx in _group_slices(group):
        out[idx[1:]] = np.diff(ts[idx]).astype(np.float64)
    return out

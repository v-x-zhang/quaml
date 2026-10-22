"""Walk-forward folds, block-bootstrap model comparison, and a locked holdout.

Folds
-----
:func:`walk_forward` splits the calendar into expanding folds. In each fold
every training row's label window ends before validation starts, and every
validation row's label window ends before the test period starts, so nothing a
model is fitted or early-stopped on has seen a test-period return.

Comparison
----------
Errors of overlapping forward labels are strongly autocorrelated, so a plain
standard error is far too narrow. :func:`compare` averages the squared-error
difference per day and bootstraps the days in blocks of about the label
horizon, which keeps that dependence intact.

Holdout
-------
:class:`HoldoutLock` cuts the panel at a date. Development code only ever gets
the truncated panel, in which rows past the cut do not exist (so labels near
the cut are undefined rather than peeking across). Opening the holdout is an
explicit, logged act, refused the second time unless forced.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .panel import Panel


@dataclass
class Fold:
    k: int
    train: np.ndarray   # row indices
    val: np.ndarray
    test: np.ndarray
    test_start: np.datetime64
    test_end: np.datetime64  # exclusive


def walk_forward(panel: Panel, horizon: int, n_folds: int = 4, first_test: float = 0.5,
                 val_days: int = 252) -> list[Fold]:
    """Expanding walk-forward folds over the panel's calendar.

    The span from ``first_test`` (fraction of the calendar) to the end is cut
    into ``n_folds`` equal test periods. For a test period starting at day
    ``s``: validation rows have ``day`` in ``[s - horizon - val_days, s - horizon)``
    and training rows have ``day + horizon`` before validation starts.
    Filter the returned rows for usable features and labels yourself.
    """
    n = len(panel.calendar)
    bounds = np.linspace(int(first_test * n), n, n_folds + 1).astype(int)
    folds = []
    for k, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        v = s - horizon - val_days
        if v - horizon <= 0:
            raise ValueError(f"fold {k}: no room for training before day {s}; lower val_days, "
                             "n_folds, or raise first_test")
        d = panel.day
        folds.append(Fold(
            k,
            train=np.flatnonzero(d + horizon < v),
            val=np.flatnonzero((d >= v) & (d < s - horizon)),
            test=np.flatnonzero((d >= s) & (d < e)),
            test_start=panel.calendar[s],
            test_end=panel.calendar[e - 1] + np.timedelta64(1, "D"),
        ))
    return folds


def daily_means(day: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of ``values`` per distinct ``day``, in day order."""
    uniq, inv = np.unique(day, return_inverse=True)
    return np.bincount(inv, weights=values) / np.bincount(inv)


def block_bootstrap(x: np.ndarray, block: int, n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> tuple[float, float, float]:
    """Mean of a time series and a moving-block bootstrap ``1 - alpha`` interval."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


@dataclass
class Comparison:
    name: str
    mse: float
    diff: float          # mean daily (MSE - reference MSE); negative = better
    ci_low: float
    ci_high: float
    folds_better: int
    n_folds: int

    @property
    def significant(self) -> bool:
        return self.ci_high < 0 or self.ci_low > 0


def compare(pred: np.ndarray, ref: np.ndarray, y: np.ndarray, day: np.ndarray,
            fold: np.ndarray, block: int, name: str = "", n_boot: int = 2000,
            seed: int = 0) -> Comparison:
    """MSE of ``pred`` against ``ref`` on the same rows, with a block-bootstrap CI
    on the per-day difference and the number of folds where ``pred`` wins."""
    se, se_ref = (pred - y) ** 2, (ref - y) ** 2
    diff, lo, hi = block_bootstrap(daily_means(day, se - se_ref), block, n_boot, seed=seed)
    folds = np.unique(fold)
    better = sum(se[fold == f].mean() < se_ref[fold == f].mean() for f in folds)
    return Comparison(name, float(se.mean()), diff, lo, hi, int(better), len(folds))


class HoldoutLock:
    """Seal every calendar day from ``start`` on until explicitly opened.

    ``log_path`` records each opening (time, start date, reason). A second
    opening for the same ``start`` raises unless ``allow_repeat`` is set,
    because every look at the holdout spends some of its credibility.
    """

    def __init__(self, start, log_path: str | Path = "quaml_holdout_log.jsonl"):
        self.start = np.datetime64(start, "D")
        self.log_path = Path(log_path)

    def _pos(self, panel: Panel) -> int:
        pos = int(np.searchsorted(panel.calendar, self.start))
        if pos == 0 or pos >= len(panel.calendar):
            raise ValueError(f"holdout start {self.start} must fall inside the calendar "
                             f"{panel.calendar[0]}..{panel.calendar[-1]}")
        return pos

    def development(self, panel: Panel) -> Panel:
        """The panel with every day from ``start`` on removed."""
        return panel.until(self._pos(panel))

    def openings(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        rows = [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]
        return [r for r in rows if r["start"] == str(self.start)]

    def open(self, panel: Panel, reason: str, allow_repeat: bool = False) -> np.ndarray:
        """Log the opening and return the holdout rows of the full ``panel``."""
        if not reason.strip():
            raise ValueError("give a reason for opening the holdout")
        previous = self.openings()
        if previous and not allow_repeat:
            raise PermissionError(
                f"holdout from {self.start} was already opened {len(previous)} time(s), "
                f"first at {previous[0]['time']} ({previous[0]['reason']!r}). "
                "Pass allow_repeat=True to look again.")
        pos = self._pos(panel)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "start": str(self.start), "reason": reason,
                                "repeat": len(previous)}) + "\n")
        return np.flatnonzero(panel.day >= pos)

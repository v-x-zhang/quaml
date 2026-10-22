"""Daily panels: one row per (asset, trading day), with leakage-checked operations.

Layering
--------
Ingestion and cleaning (CSV parsing, dates, joins) belong to a dataframe
library; :func:`read_price_csv` uses polars. :func:`from_frame` hands any
polars or pandas frame to a :class:`Panel`, which is where quaml's own
operations start.

Calendar alignment
------------------
Every asset is reindexed onto one trading calendar (normally the market
proxy's trading days) from its first to its last date. Missing days become NaN
rows, so a lag of ``k`` rows is always ``k`` trading days and a window of ``n``
rows always spans ``n`` trading days.

Operations declare their reach
------------------------------
Each :class:`Op` records how many rows of the past (``lookback``) and the
future (``lookahead``) it reads. :func:`design_matrix` refuses features that
look ahead, and :func:`time_split` purges each period's last ``horizon`` rows
so no label window crosses into the next period.

Rows of one asset are contiguous and in time order; asset histories never mix.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Panel:
    asset: np.ndarray            # int64, sorted; rows of one asset are contiguous
    day: np.ndarray              # int64 index into ``calendar``, +1 per row within an asset
    calendar: np.ndarray         # datetime64[D] trading days
    cols: dict[str, np.ndarray] = field(default_factory=dict)  # float64 columns

    def __post_init__(self):
        n = len(self.asset)
        if len(self.day) != n or any(len(v) != n for v in self.cols.values()):
            raise ValueError("all columns must have one value per row")
        if n > 1:
            same = self.asset[1:] == self.asset[:-1]
            if np.any(self.asset[1:] < self.asset[:-1]):
                raise ValueError("rows must be sorted by asset")
            if np.any(same & (self.day[1:] != self.day[:-1] + 1)):
                raise ValueError("each asset must cover consecutive calendar days")
        starts = np.flatnonzero(np.r_[True, self.asset[1:] != self.asset[:-1]]) if n else \
            np.zeros(0, dtype=np.int64)
        lengths = np.diff(np.r_[starts, n])
        object.__setattr__(self, "_segments", list(zip(starts, starts + lengths)))
        object.__setattr__(self, "_start", np.repeat(starts, lengths))
        object.__setattr__(self, "_end", np.repeat(starts + lengths, lengths))

    def __len__(self) -> int:
        return len(self.asset)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.cols[name]

    @property
    def date(self) -> np.ndarray:
        return self.calendar[self.day]

    def with_columns(self, **new: np.ndarray) -> Panel:
        return replace(self, cols={**self.cols, **{k: np.asarray(v, np.float64) for k, v in new.items()}})

    def until(self, pos: int) -> Panel:
        """Only calendar days before position ``pos``; later rows do not exist."""
        keep = self.day < pos
        return Panel(self.asset[keep], self.day[keep], self.calendar[:pos],
                     {k: v[keep] for k, v in self.cols.items()})

    def shifted(self, values: np.ndarray, k: int) -> np.ndarray:
        """``values`` from ``k`` rows earlier (``k < 0``: later) in the same asset, else NaN."""
        i = np.arange(len(self)) - k
        ok = (i >= self._start) & (i < self._end)
        out = np.full(len(self), np.nan)
        out[ok] = values[i[ok]]
        return out


# --- ingestion -------------------------------------------------------------------------

def csv_files(paths) -> list[Path]:
    """Expand a file, a directory (its ``*.csv``), a glob pattern, or a list of these."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(p.glob("*.csv"))
        elif glob.has_magic(str(p)):
            files += sorted(Path(f) for f in glob.glob(str(p)))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"no CSV files match {paths}")
    return files


def read_price_csv(paths, *, asset="Code", symbol="Symbol", date="Date", close="Close",
                   volume="Volume", factor: str | None = "Adjustment Factor",
                   date_format="%m/%d/%Y"):
    """Read one or many long-format daily price CSVs into one cleaned polars frame.

    ``paths`` is a file, a directory, a glob such as ``"data/*.csv"``, or a
    list. Files are stacked; rows repeated exactly across files are kept once
    (conflicting duplicates are caught later by :func:`from_frame`).

    Adds ``adj_close = close * factor`` and ``adj_volume = volume / factor``
    so prices and volumes are on today's share basis (``factor=None`` if the
    files have no adjustment column). Volume is parsed as float because
    spreadsheet exports write large values as ``1.65E+07``. The cumulative
    factor encodes future splits, so use it for returns and ratios, never as a
    raw price-level feature.
    """
    import polars as pl

    schema = {asset: pl.Int64, symbol: pl.Utf8, date: pl.Utf8, close: pl.Float64,
              volume: pl.Float64}
    if factor is not None:
        schema[factor] = pl.Float64
    frames = []
    for f in csv_files(paths):
        part = pl.read_csv(f, schema_overrides=schema)
        missing = [c for c in schema if c not in part.columns]
        if missing:
            raise ValueError(f"{f}: missing columns {missing}; found {part.columns}")
        frames.append(part.select(list(schema)))
    df = pl.concat(frames).unique()

    fac = pl.col(factor) if factor is not None else pl.lit(1.0)
    df = df.with_columns(
        pl.col(date).str.to_date(date_format),
        (pl.col(close) * fac).alias("adj_close"),
        (pl.col(volume) / fac).alias("adj_volume"),
    )
    return df.sort(asset, date)


def from_frame(df, *, asset: str, date: str, columns: list[str], calendar=None) -> Panel:
    """Build a :class:`Panel` from a polars or pandas frame.

    ``calendar`` is the list of trading days (default: every date in ``df``).
    Rows dated off the calendar, and duplicate (asset, date) rows, are errors.
    """
    a = np.asarray(df[asset].to_numpy(), dtype=np.int64)
    d = np.asarray(df[date].to_numpy()).astype("datetime64[D]")
    values = {c: np.asarray(df[c].to_numpy(), dtype=np.float64) for c in columns}

    cal = np.unique(d) if calendar is None else np.unique(np.asarray(calendar).astype("datetime64[D]"))
    pos = np.searchsorted(cal, d)
    off = (pos >= len(cal)) | (cal[np.minimum(pos, len(cal) - 1)] != d)
    if off.any():
        raise ValueError(f"{off.sum()} rows fall on dates outside the calendar, e.g. {d[off][0]}")

    order = np.lexsort((pos, a))
    a, pos = a[order], pos[order]
    if np.any((a[1:] == a[:-1]) & (pos[1:] == pos[:-1])):
        raise ValueError("duplicate (asset, date) rows")

    # Reindex every asset onto consecutive calendar days from its first to its last date.
    assets, first = np.unique(a, return_index=True)
    last = np.r_[first[1:], len(a)] - 1
    lo, hi = pos[first], pos[last]
    lengths = hi - lo + 1
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    asset_of_row = np.repeat(np.arange(len(assets)), np.diff(np.r_[first, len(a)]))
    target = offsets[asset_of_row] + (pos - lo[asset_of_row])

    n = int(lengths.sum())
    out_asset = np.repeat(assets, lengths)
    out_day = np.concatenate([np.arange(l, h + 1) for l, h in zip(lo, hi)]) if n else \
        np.zeros(0, dtype=np.int64)
    cols = {}
    for c, v in values.items():
        full = np.full(n, np.nan)
        full[target] = v[order]
        cols[c] = full
    return Panel(out_asset, out_day.astype(np.int64), cal, cols)


def add_market(panel: Panel, market_asset: int, col: str, name: str) -> Panel:
    """Copy the market asset's ``col`` onto every row of the same day as ``name``."""
    is_mkt = panel.asset == market_asset
    if not is_mkt.any():
        raise ValueError(f"market asset {market_asset} not in panel")
    by_day = np.full(len(panel.calendar), np.nan)
    by_day[panel.day[is_mkt]] = panel[col][is_mkt]
    return panel.with_columns(**{name: by_day[panel.day]})


def log_returns(panel: Panel, price: str, exclude: str | None = "excluded") -> np.ndarray:
    """``log(p[t] / p[t-1])``; NaN when either price is missing or row ``t`` is
    marked in the ``exclude`` column (see :func:`quaml.repair.apply_actions`)."""
    p = panel[price]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(p / panel.shifted(p, 1))
    if exclude is not None and exclude in panel.cols:
        r[panel[exclude] > 0] = np.nan
    return r


# --- operations ------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    """A per-row computation that reads ``lookback`` past and ``lookahead`` future rows."""

    name: str
    lookback: int
    lookahead: int
    fn: Callable[[Panel], np.ndarray]

    def __call__(self, panel: Panel) -> np.ndarray:
        return self.fn(panel)


def col(name: str) -> Op:
    return Op(name, 0, 0, lambda p: p[name])


def _op(x: str | Op) -> Op:
    return col(x) if isinstance(x, str) else x


def lag(x: str | Op, k: int) -> Op:
    """Value ``k`` trading days ago."""
    if k < 1:
        raise ValueError("lag must be at least 1; use lead() for the future")
    x = _op(x)
    return Op(f"{x.name}_lag{k}", x.lookback + k, x.lookahead, lambda p: p.shifted(x(p), k))


def lead(x: str | Op, k: int) -> Op:
    """Value ``k`` trading days ahead (a label, never a feature)."""
    x = _op(x)
    return Op(f"{x.name}_lead{k}", x.lookback, x.lookahead + k, lambda p: p.shifted(x(p), -k))


def derive(name: str, fn: Callable[..., np.ndarray], *inputs: str | Op) -> Op:
    """Combine other ops row-wise, e.g. ``derive("vol_ratio", np.divide, vol21, vol63)``."""
    ops = [_op(x) for x in inputs]
    return Op(name, max(o.lookback for o in ops), max(o.lookahead for o in ops),
              lambda p: fn(*(o(p) for o in ops)))


def _window_sums(p: Panel, arrays, n: int, end: int):
    """Sums over rows ``[i + end - n + 1, i + end]`` of the same asset, using only
    rows where every array is finite. Windows that leave the asset's history
    get count 0.

    Prefix sums restart at each asset, so an asset's results depend only on
    its own rows, bit for bit.
    """
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    i = np.arange(len(p))
    lo, hi = i + end - n + 1, i + end
    inside = (lo >= p._start) & (hi < p._end)
    hi = np.where(inside, hi, i)
    has_before = inside & (lo > p._start)
    before = np.where(has_before, lo - 1, i)

    def window(v):
        prefix = np.empty(len(v))
        for s, e in p._segments:
            prefix[s:e] = np.cumsum(v[s:e])
        return np.where(inside, prefix[hi] - np.where(has_before, prefix[before], 0), 0)

    sums = [window(np.where(valid, a, 0.0)) for a in arrays]
    return sums, window(valid.astype(np.float64)).astype(np.int64)


def _moments(x: Op, n: int, end: int, min_obs: int | None):
    min_obs = n if min_obs is None else min_obs

    def fn(p):
        v = x(p)
        (s1, s2), cnt = _window_sums(p, [v, v * v], n, end)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s1 / cnt
            var = (s2 - s1 * mean) / (cnt - 1)
        ok = cnt >= max(min_obs, 2)
        return np.where(ok, mean, np.nan), np.where(ok, np.sqrt(np.maximum(var, 0)), np.nan)
    return fn


def _comoments(y: Op, x: Op, n: int, end: int, min_obs: int | None, what: str):
    min_obs = n if min_obs is None else min_obs

    def fn(p):
        a, b = y(p), x(p)
        (sa, sb, sab, sbb, saa), cnt = _window_sums(p, [a, b, a * b, b * b, a * a], n, end)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sab - sa * sb / cnt
            var_b = sbb - sb * sb / cnt
            var_a = saa - sa * sa / cnt
            out = cov / var_b if what == "beta" else cov / np.sqrt(var_a * var_b)
        return np.where((cnt >= max(min_obs, 3)) & (var_b > 0), out, np.nan)
    return fn


def rolling_mean(x: str | Op, n: int, min_obs: int | None = None) -> Op:
    x = _op(x)
    return Op(f"{x.name}_mean{n}", x.lookback + n - 1, x.lookahead,
              lambda p: _moments(x, n, 0, min_obs)(p)[0])


def rolling_std(x: str | Op, n: int, min_obs: int | None = None) -> Op:
    x = _op(x)
    return Op(f"{x.name}_std{n}", x.lookback + n - 1, x.lookahead,
              lambda p: _moments(x, n, 0, min_obs)(p)[1])


def rolling_beta(y: str | Op, x: str | Op, n: int, min_obs: int | None = None) -> Op:
    """OLS slope of ``y`` on ``x`` over the last ``n`` trading days (including today)."""
    y, x = _op(y), _op(x)
    return Op(f"beta_{y.name}_{x.name}_{n}", max(y.lookback, x.lookback) + n - 1,
              max(y.lookahead, x.lookahead), _comoments(y, x, n, 0, min_obs, "beta"))


def rolling_corr(y: str | Op, x: str | Op, n: int, min_obs: int | None = None) -> Op:
    y, x = _op(y), _op(x)
    return Op(f"corr_{y.name}_{x.name}_{n}", max(y.lookback, x.lookback) + n - 1,
              max(y.lookahead, x.lookahead), _comoments(y, x, n, 0, min_obs, "corr"))


def forward_beta(y: str | Op, x: str | Op, n: int, min_obs: int | None = None) -> Op:
    """OLS slope over the next ``n`` trading days (``t+1 .. t+n``): a label."""
    y, x = _op(y), _op(x)
    return Op(f"fwd_beta_{y.name}_{x.name}_{n}", max(y.lookback, x.lookback),
              max(y.lookahead, x.lookahead) + n, _comoments(y, x, n, n, min_obs, "beta"))


def forward_sum(x: str | Op, n: int, min_obs: int | None = None) -> Op:
    """Sum over ``t+1 .. t+n`` (e.g. forward log return): a label."""
    x = _op(x)
    min_obs = n if min_obs is None else min_obs

    def fn(p):
        (s,), cnt = _window_sums(p, [x(p)], n, n)
        return np.where(cnt >= min_obs, s, np.nan)
    return Op(f"{x.name}_fwd_sum{n}", x.lookback, x.lookahead + n, fn)


# --- design matrix and splits ------------------------------------------------------------

@dataclass
class Design:
    X: np.ndarray        # [N, F] float32; NaN where a feature is undefined
    y: np.ndarray        # [N] float64
    names: list[str]
    warmup: int          # rows of history the features need
    horizon: int         # rows of future the label reads

    @property
    def usable(self) -> np.ndarray:
        return np.isfinite(self.X).all(axis=1) & np.isfinite(self.y)


def design_matrix(panel: Panel, features: list[Op], label: Op) -> Design:
    for f in features:
        if f.lookahead > 0:
            raise ValueError(f"feature {f.name!r} reads {f.lookahead} future rows")
    names = [f.name for f in features]
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    X = np.stack([f(panel) for f in features], axis=1).astype(np.float32)
    return Design(X, label(panel), names, max((f.lookback for f in features), default=0),
                  label.lookahead)


def time_split(panel: Panel, edges, horizon: int) -> list[np.ndarray]:
    """Row indices for each period ``[edges[k], edges[k+1])`` of the calendar.

    The last ``horizon`` trading days of every period are purged, so each
    row's label window ``t+1 .. t+horizon`` stays inside its own period and
    no two periods share label information. ``edges`` are dates.
    """
    pos = np.searchsorted(panel.calendar, np.asarray(edges).astype("datetime64[D]"))
    if np.any(np.diff(pos) <= 0):
        raise ValueError("edges must be increasing")
    return [np.flatnonzero((panel.day >= lo) & (panel.day < hi - horizon))
            for lo, hi in zip(pos[:-1], pos[1:])]


def cross_sectional_ic(day: np.ndarray, pred: np.ndarray, y: np.ndarray,
                       min_names: int = 10) -> float:
    """Mean over days of the Spearman correlation between ``pred`` and ``y``."""
    rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
    ics = []
    order = np.argsort(day, kind="stable")
    bounds = np.flatnonzero(np.diff(day[order])) + 1
    for rows in np.split(order, bounds):
        if len(rows) >= min_names:
            ics.append(np.corrcoef(rank(pred[rows]), rank(y[rows]))[0, 1])
    return float(np.nanmean(ics)) if ics else np.nan

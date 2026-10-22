"""Corporate-action repair and price-jump detection for daily panels.

Vendor adjustment factors are sometimes wrong (missed reverse splits on
leveraged ETFs) or absent (spin-offs), and single bad prints happen. Either
shows up as one huge return that then poisons rolling betas, labels and every
MSE computed on them. Two tools:

* :func:`apply_actions` applies a reviewed corrections table.
* :func:`detect_jumps` lists returns too large to be ordinary moves and
  guesses what each is, with a suggested correction.

Run repair before computing returns. Detection classifies a spike by whether
the *next* day reverts it, so it is a data-cleaning step over the whole
history, not something a live strategy could do on the day.
"""

from __future__ import annotations

import warnings

import numpy as np

from .panel import Panel

# rescale_before: multiply every price strictly before ``date`` by ``value``.
#   Unadjusted 2:1 split -> 0.5. 1:10 reverse split -> 10.
#   Spin-off -> 1 - (spun-off value per share) / (close before the ex-date).
# bad_price:   the price on ``date`` is wrong; set it missing (drops both
#   adjacent returns).
# drop_return: the price is right but the return into ``date`` is not a market
#   move (e.g. an event you cannot adjust for); exclude that one return.
ACTIONS = ("rescale_before", "bad_price", "drop_return")

# Common split and reverse-split ratios, as price multipliers on the ex-date.
_SPLIT_RATIOS = np.array(sorted({r for k in (2, 3, 4, 5, 8, 10, 15, 20, 1.5, 4 / 3)
                                 for r in (1 / k, k)}))


def apply_actions(panel: Panel, actions, price: str = "adj_close") -> Panel:
    """Apply a corrections table with columns ``code, date, action[, value]``.

    ``actions`` is a polars or pandas frame (``date`` as a date type), or a
    path to a CSV with ``M/D/YYYY`` or ISO dates. Adds an ``excluded`` column
    (1 on ``drop_return`` days) for :func:`quaml.panel.log_returns`.
    """
    actions = _as_frame(actions)
    codes = np.asarray(actions["code"].to_numpy(), dtype=np.int64)
    dates = np.asarray(actions["date"].to_numpy()).astype("datetime64[D]")
    kinds = [str(k) for k in actions["action"].to_list()]
    values = (np.asarray(actions["value"].to_numpy(), dtype=np.float64)
              if "value" in actions.columns else np.full(len(codes), np.nan))

    px = panel[price].copy()
    excluded = panel.cols.get("excluded", np.zeros(len(panel))).copy()
    problems = []
    for code, date, kind, value in zip(codes, dates, kinds, values):
        rows = np.flatnonzero(panel.asset == code)
        pos = np.searchsorted(panel.calendar, date)
        if kind not in ACTIONS:
            problems.append(f"{code} {date}: unknown action {kind!r} (expected {ACTIONS})")
        elif len(rows) == 0:
            problems.append(f"{code} {date}: code not in panel")
        elif pos >= len(panel.calendar) or panel.calendar[pos] != date:
            problems.append(f"{code} {date}: not a trading day")
        elif not panel.day[rows[0]] <= pos <= panel.day[rows[-1]]:
            problems.append(f"{code} {date}: outside the asset's history")
        elif kind == "rescale_before":
            if not value > 0:
                problems.append(f"{code} {date}: rescale_before needs a positive value")
            else:
                px[rows[panel.day[rows] < pos]] *= value
        else:
            row = rows[panel.day[rows] == pos][0]
            if kind == "bad_price":
                px[row] = np.nan
            else:
                excluded[row] = 1.0
    if problems:
        raise ValueError("invalid actions:\n  " + "\n  ".join(problems))
    return panel.with_columns(**{price: px, "excluded": excluded})


def detect_jumps(panel: Panel, ret: str = "ret", mkt: str | None = "mkt",
                 min_abs: float = 0.25, z: float = 8.0, lookback: int = 63,
                 split_tol: float = 0.03):
    """Flag returns too large to be ordinary moves; returns a polars frame.

    A return is flagged when its market-relative size exceeds both ``min_abs``
    (in log terms) and ``z`` times the stock's robust volatility over the
    previous ``lookback`` days. Each flag gets a ``kind``:

    * ``split_like`` -- the price ratio is within ``split_tol`` of a common
      split ratio; suggested fix ``rescale_before`` by that ratio.
    * ``reverting`` -- the next day undoes at least half of it; suggested fix
      ``bad_price`` on the day.
    * ``unexplained`` -- neither; suggested fix ``drop_return`` (review it).
    """
    import polars as pl

    r = panel[ret]
    excess = r - panel[mkt] if mkt is not None else r
    sigma = _lagged_robust_sigma(panel, excess, lookback)
    with np.errstate(invalid="ignore"):
        flagged = (np.abs(excess) > min_abs) & ~(np.abs(excess) <= z * sigma)
    flagged &= np.isfinite(excess)
    rows = np.flatnonzero(flagged)

    nxt = panel.shifted(r, -1)
    ratio = np.exp(r[rows])
    nearest = _SPLIT_RATIOS[np.abs(np.log(_SPLIT_RATIOS)[None, :] - np.log(ratio)[:, None]).argmin(1)]
    split_like = np.abs(ratio / nearest - 1) < split_tol
    reverting = ~split_like & (np.sign(nxt[rows]) == -np.sign(r[rows])) & \
        (np.abs(nxt[rows]) >= 0.5 * np.abs(r[rows]))
    # The day after a reverting spike is the same bad print undoing itself;
    # bad_price on the spike already removes it, so do not list it twice. (Row
    # + 1 is never another asset's row: an asset's first return is NaN.)
    echo = np.isin(rows, rows[reverting] + 1)
    rows, ratio, nearest = rows[~echo], ratio[~echo], nearest[~echo]
    split_like, reverting = split_like[~echo], reverting[~echo]
    kind = np.where(split_like, "split_like", np.where(reverting, "reverting", "unexplained"))
    action = np.where(split_like, "rescale_before",
                      np.where(reverting, "bad_price", "drop_return"))
    return pl.DataFrame({
        "code": panel.asset[rows],
        "date": panel.date[rows],
        "ret": r[rows],
        "mkt": panel[mkt][rows] if mkt is not None else np.full(len(rows), np.nan),
        "next_ret": nxt[rows],
        "robust_vol": sigma[rows],
        "kind": kind,
        "action": action,
        "value": np.where(split_like, nearest, np.nan),
    })


def suggested_actions(report):
    """The ``code, date, action, value`` columns of a :func:`detect_jumps` report."""
    return report.select("code", "date", "action", "value")


def _lagged_robust_sigma(panel: Panel, x: np.ndarray, lookback: int) -> np.ndarray:
    """1.4826 * median |x| over the previous ``lookback`` rows (today excluded)."""
    out = np.full(len(x), np.nan)
    min_obs = lookback // 2
    for s, e in panel._segments:
        m = e - s
        # Row t of the view is x[t - lookback : t] (NaN-padded before the start).
        padded = np.r_[np.full(lookback, np.nan), np.abs(x[s:e])][: m + lookback - 1]
        windows = np.lib.stride_tricks.sliding_window_view(padded, lookback)
        count = np.isfinite(windows).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
            med = np.nanmedian(windows, axis=1)
        out[s:e] = np.where(count >= min_obs, 1.4826 * med, np.nan)
    return out


def _as_frame(actions):
    import polars as pl

    if isinstance(actions, pl.DataFrame):
        df = actions
    elif hasattr(actions, "to_numpy") and hasattr(actions, "columns"):  # pandas
        df = pl.from_pandas(actions)
    else:
        df = pl.read_csv(actions, schema_overrides={"date": pl.Utf8})
    df = df.rename({c: c.lower() for c in df.columns})
    if df["date"].dtype == pl.Utf8:
        iso = df["date"].str.contains(r"^\d{4}-").all()
        df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d" if iso else "%m/%d/%Y"))
    return df

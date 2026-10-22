"""Synthetic daily price CSVs with known, time-varying betas.

The file layout copies a typical vendor export: ``Code,Symbol,Date,Close,
Volume,Adjustment Factor``, unsorted rows, ``M/D/YYYY`` dates, large volumes
written as ``1.65E+07``, split-adjustment factors, late listings, delistings,
and randomly missing days. ``SPY`` (code 1) is the market.

Each stock's beta follows a mean-reverting path around a level that rises with
its (latent) volatility state, so trailing betas and trailing volatility both
carry information about forward beta, and the volatility link is nonlinear.
"""

from __future__ import annotations

import csv

import numpy as np


def _excel_number(v: float) -> str:
    # Spreadsheets display large numbers in 3-significant-figure scientific form.
    return f"{v:.2E}" if v >= 1e7 else str(int(round(v)))


def write_synthetic_price_csv(path, n_assets: int = 150, n_days: int = 2520,
                              seed: int = 0, corrupt: bool = False) -> dict:
    """Write the CSV and return ``{"betas": {code: (days, beta)}, "corruptions": [...]}``.

    With ``corrupt=True`` the file also carries data errors of the kind vendor
    files have: one split whose adjustment factor was never recorded, and a
    few single-day bad prints. ``corruptions`` lists them as
    ``(code, date, kind)``.
    """
    rng = np.random.default_rng(seed)
    corruptions = []
    days = np.busday_offset("2010-01-04", np.arange(n_days), roll="forward")

    # Market with slowly varying volatility.
    mkt_vol = 0.011 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days)) * 0.3)
    mkt_vol = np.clip(mkt_vol, 0.005, 0.04)
    mkt = rng.normal(0.0004, 1, n_days) * mkt_vol

    rows = [(1, "SPY", d, 100 * np.exp(c), rng.lognormal(18, 0.3), 1.0)
            for d, c in zip(days, np.cumsum(mkt))]
    truth = {}
    for k in range(n_assets):
        code = 10_000 + 37 * k
        base = rng.uniform(0.5, 1.6)
        # Latent volatility state z: AR(1). Higher z -> more idiosyncratic vol and higher beta.
        z = np.zeros(n_days)
        for t in range(1, n_days):
            z[t] = 0.99 * z[t - 1] + rng.normal(0, 0.08)
        beta = np.empty(n_days)
        beta[0] = base
        for t in range(1, n_days):
            target = base + 0.6 * np.tanh(z[t])
            beta[t] = target + 0.995 * (beta[t - 1] - target) + rng.normal(0, 0.015)
        idio = 0.015 * np.exp(z) * rng.normal(size=n_days)
        ret = beta * mkt + idio
        adj = 30 * np.exp(np.cumsum(ret))

        first = int(rng.integers(0, n_days // 2)) if rng.random() < 0.3 else 0
        last = int(rng.integers(n_days // 2, n_days)) if rng.random() < 0.2 else n_days - 1
        keep = np.zeros(n_days, bool)
        keep[first:last + 1] = True
        keep &= rng.random(n_days) > 0.005

        factor = np.ones(n_days)
        split_day = None
        if rng.random() < 0.3:  # one split; earlier rows carry factor 1/ratio
            split_day = int(rng.integers(n_days // 4, n_days))
            factor[:split_day] = 1 / rng.choice([2, 4, 5])
        close = adj / factor
        reported_factor = factor.copy()
        live = np.flatnonzero(keep)
        visible = split_day is not None and live.min() < split_day <= live.max()
        if corrupt and visible and not any(c[2] == "missed_split" for c in corruptions):
            reported_factor[:] = 1.0  # vendor never recorded the split
            corruptions.append((code, days[live[live >= split_day][0]], "missed_split"))
        if corrupt and k % 50 == 7 and len(live) > 10:
            t = int(rng.choice(live[1:-1]))
            close[t] *= 1.6
            corruptions.append((code, days[t], "bad_print"))
        volume = rng.lognormal(15, 0.8, n_days)
        for t in live:
            rows.append((code, f"S{k:03d}", days[t], close[t],
                         volume[t] * factor[t], reported_factor[t]))
        truth[code] = (days, beta)

    order = rng.permutation(len(rows))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Code", "Symbol", "Date", "Close", "Volume", "Adjustment Factor"])
        for i in order:
            code, sym, d, close, vol, fac = rows[i]
            d = d.astype(object)
            w.writerow([code, sym, f"{d.month}/{d.day}/{d.year}", f"{close:.2f}",
                        _excel_number(vol), f"{fac:g}"])
    return {"betas": truth, "corruptions": corruptions}

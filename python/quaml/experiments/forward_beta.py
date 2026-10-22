"""Forecast each stock's beta over the next ``--horizon`` trading days.

    quaml-forward-beta --csv "data/*.csv"
    quaml-forward-beta --csv prices_2010.csv prices_2011.csv --out preds.csv
    quaml-forward-beta --synthetic                     # generated data with known betas

Compares, on purged chronological splits (60/20/20 by date):

* persistence -- the forecast is today's trailing beta (the bar to beat;
  beta is very persistent),
* OLS on lagged betas, volatilities and correlations,
* an MLP on the same features,
* a GRU over each stock's last ``--seq-length`` days of those features.

Metrics are MSE against realised forward beta and the mean daily
cross-sectional rank IC (does the model order stocks by beta correctly).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from quaml import panel as qp

MODELS = ("persistence", "ols", "mlp", "gru")


def features() -> list[qp.Op]:
    ret, mkt = "ret", "mkt"
    vol21, vol63, vol252 = (qp.rolling_std(ret, n, min_obs=int(0.8 * n)) for n in (21, 63, 252))
    mvol63 = qp.rolling_std(mkt, 63, min_obs=50)
    betas = [qp.rolling_beta(ret, mkt, n, min_obs=int(0.8 * n)) for n in (21, 63, 126, 252)]
    return [
        *betas,
        qp.lag(betas[1], 21),                          # last month's 63d beta
        qp.lag(betas[1], 63),                          # last quarter's
        qp.rolling_corr(ret, mkt, 63, min_obs=50),
        vol21, vol63, vol252,
        qp.derive("vol_ratio_21_252", np.divide, vol21, vol252),
        qp.derive("rel_vol_63", np.divide, vol63, mvol63),
        mvol63,
        qp.rolling_mean(ret, 252, min_obs=200),
        qp.derive("log_dollar_vol_63", np.log1p, qp.rolling_mean("dollar_vol", 63, min_obs=50)),
    ]


def in_chunks(fn, n: int, chunk: int = 65_536) -> np.ndarray:
    """Apply ``fn`` to index ranges so large evaluation sets fit in memory."""
    return np.concatenate([fn(np.arange(s, min(s + chunk, n))) for s in range(0, n, chunk)]) \
        if n else np.zeros(0, np.float32)


def train(model, get_x, y, val_pred, y_val, epochs, batch=1024, lr=1e-3):
    """Adam on MSE, keeping the epoch with the best validation MSE."""
    import torch

    y_t = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state = np.inf, None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(y)).split(batch):
            loss = torch.nn.functional.mse_loss(model(get_x(idx.numpy())), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean((val_pred() - y_val) ** 2))
        print(f"  epoch {epoch + 1:>2}/{epochs}: val mse {val:.4f}")
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def subsample(rows, n, rng):
    return np.sort(rng.choice(rows, size=n, replace=False)) if len(rows) > n else rows


def load(args):
    import polars as pl

    df = qp.read_price_csv(args.csv, date_format=args.date_format)
    market = df.filter(pl.col("Symbol") == args.market)
    if market.is_empty():
        raise SystemExit(f"market symbol {args.market!r} not found")
    market_code = int(market["Code"][0])
    df = df.with_columns((pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_vol"))
    p = qp.from_frame(df, asset="Code", date="Date", columns=["adj_close", "dollar_vol"],
                      calendar=market["Date"])
    p = p.with_columns(ret=qp.log_returns(p, "adj_close"))
    p = qp.add_market(p, market_code, "ret", "mkt")
    symbols = dict(df.group_by("Code").agg(pl.col("Symbol").last()).iter_rows())
    return df, p, market_code, symbols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", nargs="+", help="CSV files, directories, or globs")
    src.add_argument("--synthetic", action="store_true", help="generate a demo CSV")
    ap.add_argument("--market", default="SPY", help="symbol of the market proxy")
    ap.add_argument("--date-format", default="%m/%d/%Y")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--models", default=",".join(MODELS),
                    help=f"comma-separated subset of {','.join(MODELS)}")
    ap.add_argument("--seq-length", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-train", type=int, default=500_000,
                    help="subsample training rows for the neural models")
    ap.add_argument("--max-val", type=int, default=200_000)
    ap.add_argument("--out", type=Path, help="write test-period predictions to this CSV")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    models = [m.strip().lower() for m in args.models.split(",")]
    if unknown := set(models) - set(MODELS):
        ap.error(f"unknown models: {sorted(unknown)}")

    if args.synthetic:
        from quaml.synthetic import write_synthetic_price_csv
        args.csv = [Path(tempfile.gettempdir()) / "quaml_synthetic_prices.csv"]
        write_synthetic_price_csv(args.csv[0], seed=args.seed)

    t0 = time.perf_counter()
    df, p, market_code, symbols = load(args)
    label = qp.forward_beta("ret", "mkt", args.horizon, min_obs=int(0.8 * args.horizon))
    design = qp.design_matrix(p, features(), label)
    print(f"{len(qp.csv_files(args.csv))} file(s), {len(df):,} rows -> "
          f"{len(np.unique(p.asset)) - 1:,} stocks, {p.calendar[0]}..{p.calendar[-1]}, "
          f"{len(design.names)} features ({time.perf_counter() - t0:.1f}s)")

    cal = p.calendar
    edges = [cal[0], cal[int(0.6 * len(cal))], cal[int(0.8 * len(cal))],
             cal[-1] + np.timedelta64(1, "D")]
    usable = design.usable & (p.asset != market_code)
    train_rows, val_rows, test_rows = (r[usable[r]] for r in qp.time_split(p, edges, args.horizon))
    if min(len(train_rows), len(val_rows), len(test_rows)) == 0:
        raise SystemExit("not enough history: a split is empty (need ~1 year of warm-up "
                         f"plus {args.horizon} days per split)")
    print(f"rows: train {len(train_rows):,} | val {len(val_rows):,} | test {len(test_rows):,} "
          f"(purged {args.horizon} days after each period)")

    mean = design.X[train_rows].mean(0)
    std = design.X[train_rows].std(0) + 1e-12
    Xs = np.nan_to_num((design.X - mean) / std).astype(np.float32)
    y = np.nan_to_num(design.y).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    nn_train = subsample(train_rows, args.max_train, rng)
    nn_val = subsample(val_rows, args.max_val, rng)

    preds = {}
    trailing = design.names.index(f"beta_ret_mkt_{args.horizon}") \
        if f"beta_ret_mkt_{args.horizon}" in design.names else design.names.index("beta_ret_mkt_63")
    if "persistence" in models:
        preds["persistence"] = design.X[test_rows, trailing].astype(np.float64)
    if "ols" in models:
        A = lambda rows: np.c_[Xs[rows], np.ones(len(rows))]
        w, *_ = np.linalg.lstsq(A(train_rows), y[train_rows], rcond=None)
        preds["ols"] = A(test_rows) @ w

    if "mlp" in models or "gru" in models:
        import torch
        from torch import nn

        torch.manual_seed(args.seed)
        X_t = torch.from_numpy(Xs)

    if "mlp" in models:
        print("training MLP")
        mlp = nn.Sequential(nn.Linear(Xs.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                            nn.Linear(64, 1), nn.Flatten(0))
        x_of = lambda rows: X_t[torch.from_numpy(rows)]
        train(mlp, lambda i: x_of(nn_train[i]), y[nn_train],
              lambda: in_chunks(lambda i: mlp(x_of(nn_val[i])).numpy(), len(nn_val)),
              y[nn_val], args.epochs)
        with torch.no_grad():
            preds["mlp"] = in_chunks(lambda i: mlp(x_of(test_rows[i])).numpy(), len(test_rows))

    if "gru" in models:
        from quaml.datasets import EventWindowDataset, Sessions

        class GRU(nn.Module):
            def __init__(self, channels, hidden=32):
                super().__init__()
                self.gru = nn.GRU(channels, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                return self.head(self.gru(x)[0][:, -1]).squeeze(-1)

        print("training GRU")
        # Rows that are not anchors (warm-up, missing label) still feed windows.
        sessions = Sessions(Xs, p.day, p.asset, np.zeros(len(p), np.int32), y, [])
        windows = EventWindowDataset(sessions, np.zeros(0, np.int64), args.seq_length,
                                     age_unit=1)
        gru = GRU(len(design.names) + 2)
        seq_of = lambda rows: windows.gather(rows)["x"]
        train(gru, lambda i: seq_of(nn_train[i]), y[nn_train],
              lambda: in_chunks(lambda i: gru(seq_of(nn_val[i])).numpy(), len(nn_val), 16_384),
              y[nn_val], max(args.epochs // 3, 1))
        with torch.no_grad():
            preds["gru"] = in_chunks(lambda i: gru(seq_of(test_rows[i])).numpy(), len(test_rows),
                                     16_384)

    y_test = design.y[test_rows]
    print(f"\ntest {cal[p.day[test_rows].min()]}..{cal[p.day[test_rows].max()]}, "
          f"{args.horizon}-day forward beta, {len(test_rows):,} stock-days")
    base = np.mean((preds["persistence"] - y_test) ** 2) if "persistence" in preds else None
    print(f"{'model':<13}{'MSE':>9}{'vs persist':>12}{'xs rank IC':>12}")
    for name, pred in preds.items():
        mse = np.mean((pred - y_test) ** 2)
        gain = f"{1 - mse / base:>+11.1%}" if base else f"{'':>11}"
        ic = qp.cross_sectional_ic(p.day[test_rows], pred, y_test)
        print(f"{name:<13}{mse:>9.4f}{gain} {ic:>11.3f}")

    if args.out:
        import polars as pl
        pl.DataFrame({
            "date": cal[p.day[test_rows]],
            "code": p.asset[test_rows],
            "symbol": [symbols.get(int(c)) for c in p.asset[test_rows]],
            "realised_beta": y_test,
            **{f"pred_{k}": v for k, v in preds.items()},
        }).write_csv(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

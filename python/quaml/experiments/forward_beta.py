"""Forecast each stock's beta over the next ``--horizon`` trading days.

    quaml-forward-beta --csv "data/*.csv" --jump-report jumps.csv
    quaml-forward-beta --csv "data/*.csv" --actions fixes.csv --holdout-start 2022-01-03
    quaml-forward-beta --csv "data/*.csv" --actions fixes.csv --holdout-start 2022-01-03 \\
        --open-holdout "final model, chosen on walk-forward"
    quaml-forward-beta --synthetic --corrupt            # generated data with known errors

Pipeline: read CSVs -> apply corrections (``--actions``, or ``--auto-repair``
to accept the detector's suggestions) -> flag remaining price jumps -> build
features -> evaluate.

Evaluation is expanding walk-forward over the development period (everything
before ``--holdout-start``). Each fold standardises on its own training rows;
the MLP and GRU get the same epoch budget and early-stop on a purged
validation block. Every model is compared with persistence (today's trailing
beta) by the per-day MSE difference, with a block-bootstrap 95% interval and
the number of folds it won.

``--open-holdout REASON`` instead trains on the whole development period and
scores the sealed holdout once; openings are logged and a second one is
refused unless ``--allow-reopen``.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from quaml import panel as qp
from quaml import repair, validation

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
    """Adam on MSE; returns the model at its best validation epoch and that epoch."""
    import torch

    y_t = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_epoch, best_state = np.inf, 0, None
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
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, best_epoch


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
    symbols = dict(df.group_by("Code").agg(pl.col("Symbol").last()).iter_rows())
    return df, p, market_code, symbols


def with_returns(p, market_code):
    p = p.with_columns(ret=qp.log_returns(p, "adj_close"))
    return qp.add_market(p, market_code, "ret", "mkt")


def repair_step(p, market_code, args):
    """Apply corrections, report jumps that remain; returns the panel with returns."""
    if args.actions:
        p = repair.apply_actions(p, args.actions)
        print(f"applied corrections from {args.actions}")
    p = with_returns(p, market_code)
    report = repair.detect_jumps(p)
    if args.auto_repair and report.height:
        p = with_returns(repair.apply_actions(p, repair.suggested_actions(report)), market_code)
        print(f"auto-repair: applied {report.height} suggested correction(s)")
        report = repair.detect_jumps(p)
    counts = dict(report.group_by("kind").len().iter_rows()) if report.height else {}
    print(f"price jumps still flagged: {report.height}"
          + (f" ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})" if counts else ""))
    if args.jump_report:
        report.write_csv(args.jump_report)
        print(f"wrote {args.jump_report} (edit and pass back with --actions)")
    elif report.height:
        print("  review them with --jump-report PATH, then --actions or --auto-repair")
    return p


def fit_predict(models, design, p, train_rows, val_rows, test_rows, args, seed):
    """Fit every model on one split; predictions for ``test_rows`` plus fit notes."""
    rng = np.random.default_rng(seed)
    mean = design.X[train_rows].mean(0)
    std = design.X[train_rows].std(0) + 1e-12
    Xs = np.nan_to_num((design.X - mean) / std).astype(np.float32)
    y = np.nan_to_num(design.y).astype(np.float32)
    nn_train = subsample(train_rows, args.max_train, rng)
    nn_val = subsample(val_rows, args.max_val, rng)

    preds, notes = {}, []
    trailing = design.names.index(f"beta_ret_mkt_{args.horizon}") \
        if f"beta_ret_mkt_{args.horizon}" in design.names else design.names.index("beta_ret_mkt_63")
    preds["persistence"] = design.X[test_rows, trailing].astype(np.float64)
    if "ols" in models:
        A = lambda rows: np.c_[Xs[rows], np.ones(len(rows))]
        w, *_ = np.linalg.lstsq(A(train_rows), y[train_rows], rcond=None)
        preds["ols"] = A(test_rows) @ w

    if "mlp" in models or "gru" in models:
        import torch
        from torch import nn
        X_t = torch.from_numpy(Xs)

    if "mlp" in models:
        torch.manual_seed(seed)
        mlp = nn.Sequential(nn.Linear(Xs.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                            nn.Linear(64, 1), nn.Flatten(0))
        x_of = lambda rows: X_t[torch.from_numpy(rows)]
        val, ep = train(mlp, lambda i: x_of(nn_train[i]), y[nn_train],
                        lambda: in_chunks(lambda i: mlp(x_of(nn_val[i])).numpy(), len(nn_val)),
                        y[nn_val], args.epochs)
        notes.append(f"mlp best epoch {ep}/{args.epochs} (val mse {val:.4f})")
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

        torch.manual_seed(seed)
        # Rows that are not anchors (warm-up, missing label) still feed windows.
        sessions = Sessions(Xs, p.day, p.asset, np.zeros(len(p), np.int32), y, [])
        windows = EventWindowDataset(sessions, np.zeros(0, np.int64), args.seq_length, age_unit=1)
        gru = GRU(len(design.names) + 2)
        seq_of = lambda rows: windows.gather(rows)["x"]
        val, ep = train(gru, lambda i: seq_of(nn_train[i]), y[nn_train],
                        lambda: in_chunks(lambda i: gru(seq_of(nn_val[i])).numpy(), len(nn_val),
                                          16_384),
                        y[nn_val], args.epochs)
        notes.append(f"gru best epoch {ep}/{args.epochs} (val mse {val:.4f})")
        with torch.no_grad():
            preds["gru"] = in_chunks(lambda i: gru(seq_of(test_rows[i])).numpy(), len(test_rows),
                                     16_384)
    return preds, notes


def label_stats(y):
    return f"label sd {np.std(y):.3f}, max |y| {np.max(np.abs(y)):.2f}" if len(y) else "empty"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[2])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", nargs="+", help="CSV files, directories, or globs")
    src.add_argument("--synthetic", action="store_true", help="generate a demo CSV")
    ap.add_argument("--corrupt", action="store_true",
                    help="with --synthetic: add a missed split and bad prints")
    ap.add_argument("--market", default="SPY", help="symbol of the market proxy")
    ap.add_argument("--date-format", default="%m/%d/%Y")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--models", default=",".join(MODELS),
                    help=f"comma-separated subset of {','.join(MODELS)} "
                         "(persistence is always the reference)")

    rep = ap.add_argument_group("data repair")
    rep.add_argument("--actions", type=Path, help="corrections CSV: code,date,action[,value]")
    rep.add_argument("--jump-report", type=Path, help="write flagged price jumps to this CSV")
    rep.add_argument("--auto-repair", action="store_true",
                     help="apply every suggested correction from the jump detector")

    val = ap.add_argument_group("validation")
    val.add_argument("--folds", type=int, default=4)
    val.add_argument("--first-test", type=float, default=0.5,
                     help="fraction of the development calendar before the first test fold")
    val.add_argument("--val-days", type=int, default=252,
                     help="early-stopping block before each test fold")
    val.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")

    hold = ap.add_argument_group("holdout")
    hold.add_argument("--holdout-start", help="seal every day from this date (YYYY-MM-DD) on")
    hold.add_argument("--open-holdout", metavar="REASON",
                      help="train on all development data and score the holdout (logged)")
    hold.add_argument("--allow-reopen", action="store_true")
    hold.add_argument("--holdout-log", type=Path, default=Path("quaml_holdout_log.jsonl"))

    nn_ = ap.add_argument_group("neural models")
    nn_.add_argument("--seq-length", type=int, default=21)
    nn_.add_argument("--epochs", type=int, default=15, help="same budget for MLP and GRU")
    nn_.add_argument("--max-train", type=int, default=500_000,
                     help="subsample training rows for the neural models")
    nn_.add_argument("--max-val", type=int, default=200_000)

    ap.add_argument("--out", type=Path, help="write out-of-fold (or holdout) predictions here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    models = [m.strip().lower() for m in args.models.split(",")]
    if unknown := set(models) - set(MODELS):
        ap.error(f"unknown models: {sorted(unknown)}")
    if args.open_holdout is not None and not args.holdout_start:
        ap.error("--open-holdout needs --holdout-start")
    if args.corrupt and not args.synthetic:
        ap.error("--corrupt only applies to --synthetic")

    if args.synthetic:
        from quaml.synthetic import write_synthetic_price_csv
        args.csv = [Path(tempfile.gettempdir()) / "quaml_synthetic_prices.csv"]
        made = write_synthetic_price_csv(args.csv[0], seed=args.seed, corrupt=args.corrupt)
        for code, day, kind in made["corruptions"]:
            print(f"synthetic corruption: {kind} code {code} on {day}")

    t0 = time.perf_counter()
    df, p, market_code, symbols = load(args)
    print(f"{len(qp.csv_files(args.csv))} file(s), {len(df):,} rows -> "
          f"{len(np.unique(p.asset)) - 1:,} stocks, {p.calendar[0]}..{p.calendar[-1]} "
          f"({time.perf_counter() - t0:.1f}s)")
    p = repair_step(p, market_code, args)

    label = qp.forward_beta("ret", "mkt", args.horizon, min_obs=int(0.8 * args.horizon))
    lock = validation.HoldoutLock(args.holdout_start, args.holdout_log) \
        if args.holdout_start else None

    splits = []  # (name, panel, design, train, val, test)
    if args.open_holdout is not None:
        dev_end = int(np.searchsorted(p.calendar, lock.start))
        design = qp.design_matrix(p, features(), label)
        usable = design.usable & (p.asset != market_code)
        test = lock.open(p, args.open_holdout, allow_repeat=args.allow_reopen)
        v = dev_end - args.horizon - args.val_days
        tr = np.flatnonzero(usable & (p.day + args.horizon < v))
        va = np.flatnonzero(usable & (p.day >= v) & (p.day + args.horizon < dev_end))
        splits.append(("holdout", p, design, tr, va, test[usable[test]]))
        print(f"HOLDOUT OPENED from {lock.start} (logged to {args.holdout_log})")
    else:
        work = lock.development(p) if lock else p
        if lock:
            print(f"holdout sealed from {lock.start}; developing on "
                  f"{work.calendar[0]}..{work.calendar[-1]}")
        design = qp.design_matrix(work, features(), label)
        usable = design.usable & (work.asset != market_code)
        for f in validation.walk_forward(work, args.horizon, args.folds, args.first_test,
                                         args.val_days):
            splits.append((f"fold {f.k}", work, design, f.train[usable[f.train]],
                           f.val[usable[f.val]], f.test[usable[f.test]]))

    all_rows, fold_ids, panels, preds = [], [], None, {}
    for k, (name, panel, design, tr, va, te) in enumerate(splits):
        if min(len(tr), len(va), len(te)) == 0:
            raise SystemExit(f"{name}: empty split (train {len(tr)}, val {len(va)}, test "
                             f"{len(te)}); need more history or smaller --val-days")
        cal = panel.calendar
        print(f"{name}: test {cal[panel.day[te].min()]}..{cal[panel.day[te].max()]} | "
              f"train {len(tr):,}, val {len(va):,}, test {len(te):,} | "
              f"val {label_stats(design.y[va])}; test {label_stats(design.y[te])}")
        fold_preds, notes = fit_predict(models, design, panel, tr, va, te, args, args.seed + k)
        for note in notes:
            print(f"  {note}")
        for m, v in fold_preds.items():
            preds.setdefault(m, []).append(v)
        all_rows.append(te)
        fold_ids.append(np.full(len(te), k))
        panels = (panel, design)

    panel, design = panels
    rows = np.concatenate(all_rows)
    fold = np.concatenate(fold_ids)
    preds = {m: np.concatenate(v) for m, v in preds.items()}
    y, day = design.y[rows], panel.day[rows]
    ref = preds["persistence"]

    scope = "holdout" if args.open_holdout is not None else \
        f"{len(splits)} walk-forward folds"
    print(f"\n{args.horizon}-day forward beta, {scope}, {len(rows):,} stock-days")
    print(f"{'model':<12}{'MSE':>8}{'vs persistence':>16}{'95% CI':>20}{'folds won':>11}"
          f"{'xs rank IC':>12}")
    for m, pred in preds.items():
        ic = qp.cross_sectional_ic(day, pred, y)
        if m == "persistence":
            print(f"{m:<12}{np.mean((pred - y) ** 2):>8.4f}{'(reference)':>16}{'':>20}{'':>11}"
                  f"{ic:>12.3f}")
            continue
        c = validation.compare(pred, ref, y, day, fold, block=args.horizon, name=m,
                               n_boot=args.boot, seed=args.seed)
        mark = " *" if c.significant else "  "
        print(f"{m:<12}{c.mse:>8.4f}{c.diff:>+16.4f}{f'[{c.ci_low:+.4f}, {c.ci_high:+.4f}]':>20}"
              f"{f'{c.folds_better}/{c.n_folds}':>11}{ic:>12.3f}{mark}")
    print("MSE difference is per-day mean squared error minus persistence's (negative = "
          "better); * = 95% CI excludes 0.")

    # Overlapping intervals against persistence say nothing about which of two
    # models is better; that needs the paired difference.
    neural = [m for m in ("mlp", "gru") if m in preds]
    if "ols" in preds and neural:
        print(f"\npaired against ols:")
        for m in neural:
            c = validation.compare(preds[m], preds["ols"], y, day, fold, block=args.horizon,
                                   name=m, n_boot=args.boot, seed=args.seed)
            mark = " *" if c.significant else "  "
            print(f"{m:<12}{c.diff:>+8.4f}  [{c.ci_low:+.4f}, {c.ci_high:+.4f}]  "
                  f"won {c.folds_better}/{c.n_folds} folds{mark}")

    if args.out:
        import polars as pl
        pl.DataFrame({
            "fold": fold,
            "date": panel.calendar[day],
            "code": panel.asset[rows],
            "symbol": [symbols.get(int(c)) for c in panel.asset[rows]],
            "realised_beta": y,
            **{f"pred_{k}": v for k, v in preds.items()},
        }).write_csv(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

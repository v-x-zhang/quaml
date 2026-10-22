"""Forecast 30s forward mid returns of option contracts from their last 32 trades.

    python examples/train_option_windows.py --symbol SPY --sessions 6

Splits by session in time order (train on the oldest, validate on the next,
test on the newest), compares a small GRU against OLS on the anchor row, and
reports both forecast quality and pipeline throughput.

Labels start from the first quote observed after the anchor trade by default
(``--label-start next``), because TCBBO quotes are pre-trade.

Caveats: the information coefficient is pooled across contracts, and labels
of neighbouring anchors in one contract overlap in time, so it overstates how
much independent evidence there is. Mid returns are before costs; a median
relative spread of ~1.6% on SPY options dwarfs most 30s moves.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from quaml.datasets import WINDOW_CHANNELS, EventWindowDataset, fit_standardizer, load_sessions


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
    return pearson(rank(a), rank(b))


class GRUForecaster(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)  # the anchor is always the last slot


@torch.no_grad()
def predict(model, ds, batch_size):
    model.eval()
    return np.concatenate([model(b["x"]).numpy() for b in ds.loader(batch_size, shuffle=False)])


def sample(rows, n, rng):
    return np.sort(rng.choice(rows, size=n, replace=False)) if len(rows) > n else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="C:/repos/conetworks/data/raw/options")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--sessions", type=int, default=6, help="newest N sessions; last 2 are val/test")
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--label-start", choices=["next", "anchor"], default="next",
                    help="reference quote for the label; 'anchor' includes the quote's "
                         "reaction to the anchor trade itself")
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--max-eval", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.sessions < 3:
        ap.error("need at least 3 sessions (train, val, test)")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(Path(args.data_root, args.symbol, "tcbbo_clean").glob("*.tcbbo.dbn.zst"))
    paths = paths[-args.sessions:]
    names = [p.name.split(".")[0].split("-")[-1] for p in paths]
    print(f"train {names[0]}..{names[-3]} | val {names[-2]} | test {names[-1]}")

    t0 = time.perf_counter()
    data = load_sessions(paths, window_ns=args.window_ms * 1_000_000,
                         horizon_ns=int(args.horizon_s * 1e9), label_start=args.label_start)
    load_s = time.perf_counter() - t0
    print(f"loaded {len(data.ts):,} trades in {load_s:.1f}s "
          f"({len(data.ts) / load_s / 1e6:.2f}M trades/s incl. features and labels)")

    labelled = np.isfinite(data.y)
    n = len(paths)
    split = {
        "train": sample(np.flatnonzero(labelled & (data.session < n - 2)), args.max_train, rng),
        "val": sample(np.flatnonzero(labelled & (data.session == n - 2)), args.max_eval, rng),
        "test": sample(np.flatnonzero(labelled & (data.session == n - 1)), args.max_eval, rng),
    }
    print("anchors: " + ", ".join(f"{k} {len(v):,}" for k, v in split.items()))

    # Winsorise labels on training quantiles; cheap contracts have huge log moves.
    lo, hi = np.quantile(data.y[split["train"]], [0.01, 0.99])
    y_scale = np.clip(data.y[split["train"]], lo, hi).std()
    mean, std = fit_standardizer(data.X, split["train"])
    ds = {k: EventWindowDataset(data, v, args.length, mean, std) for k, v in split.items()}
    y = {k: data.y[v] for k, v in split.items()}

    # Pipeline throughput without the model.
    t0 = time.perf_counter()
    for _ in ds["train"].loader(args.batch, shuffle=True):
        pass
    gather_s = time.perf_counter() - t0
    print(f"batch gather: {len(split['train']) / gather_s / 1e3:.0f}k windows/s "
          f"([{args.batch}, {args.length}, {len(WINDOW_CHANNELS)}] float32)")

    # Baseline: OLS on the anchor row's standardised inputs.
    Xa = {k: (data.X[v] - mean) / std for k, v in split.items()}
    A = np.c_[Xa["train"], np.ones(len(Xa["train"]))]
    w, *_ = np.linalg.lstsq(A, np.clip(y["train"], lo, hi), rcond=None)
    ols_test = np.c_[Xa["test"], np.ones(len(Xa["test"]))] @ w

    model = GRUForecaster(len(WINDOW_CHANNELS), args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_ic, best_state = -np.inf, None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total, count = 0.0, 0
        for b in ds["train"].loader(args.batch, shuffle=True):
            target = torch.clamp(b["y"], lo, hi) / y_scale
            loss = nn.functional.mse_loss(model(b["x"]), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(target)
            count += len(target)
        epoch_s = time.perf_counter() - t0
        val_ic = spearman(predict(model, ds["val"], 4096), y["val"])
        print(f"epoch {epoch + 1}: train mse {total / count:.4f} | val rank IC {val_ic:+.4f} | "
              f"{count / epoch_s / 1e3:.0f}k windows/s")
        if val_ic > best_ic:
            best_ic, best_state = val_ic, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    gru_test = predict(model, ds["test"], 4096)

    print(f"\ntest session {names[-1]} ({len(y['test']):,} anchors)")
    print(f"{'model':<8}{'IC':>9}{'rank IC':>10}")
    for name, pred in [("OLS", ols_test), ("GRU", gru_test)]:
        print(f"{name:<8}{pearson(pred, y['test']):>+9.4f}{spearman(pred, y['test']):>+10.4f}")


if __name__ == "__main__":
    main()

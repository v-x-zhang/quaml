"""Reference (NumPy + databento) vs native (C++) on one TBBO/TCBBO session.

    python benchmarks/bench_trade_quotes.py <file.dbn.zst> [--window-ms 1000] [--repeat 5]

Reports the median of ``--repeat`` runs after one warm-up, and checks the two
backends agree before timing.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time

import numpy as np

import quaml
from quaml import native
from quaml.reference import features


def median_time(fn, repeat):
    fn()  # warm-up: file cache, allocator, lazy imports
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    window_ns = args.window_ms * 1_000_000

    ev = native.read_dbn(args.path)
    ref = features.trade_quote_features(ev, window_ns)
    nat = native.trade_quote_features(ev, window_ns)
    for name in ref:
        np.testing.assert_array_equal(nat[name], ref[name], err_msg=name)

    rows = [
        ("read", lambda: quaml.read_dbn(args.path), lambda: native.read_dbn(args.path)),
        ("features", lambda: features.trade_quote_features(ev, window_ns),
                     lambda: native.trade_quote_features(ev, window_ns)),
    ]

    print(f"{os.path.basename(args.path)}: {len(ev):,} trades, "
          f"{len(np.unique(ev.instrument_id)):,} instruments, window {args.window_ms} ms")
    print(f"{platform.processor()} | single thread | median of {args.repeat}")
    print(f"{'stage':<10}{'reference':>12}{'native':>12}{'speedup':>10}{'native Mev/s':>14}")
    for name, ref_fn, nat_fn in rows:
        r = median_time(ref_fn, args.repeat)
        n = median_time(nat_fn, args.repeat)
        print(f"{name:<10}{r * 1e3:>10.1f}ms{n * 1e3:>10.1f}ms{r / n:>9.1f}x"
              f"{len(ev) / n / 1e6:>14.1f}")


if __name__ == "__main__":
    main()

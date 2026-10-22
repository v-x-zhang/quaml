# C++ Quant Finance Library - compression, backtesting, & ML tooling

## Install

Python 3.11+. Pick the extras you need: `panel` (polars, for CSV panels),
`torch` (neural models), `dbn` (Databento reference reader), `test`.

**From a prebuilt wheel.** Built on Python 3.12+ the wheel targets the stable
ABI (`cp312-abi3`), so one file works on every Python >= 3.12 for that OS and
CPU (e.g. `win_amd64`). Built on 3.11 it only works on 3.11.

```
pip install "quaml-0.0.1-cp312-abi3-win_amd64.whl[panel,torch]" --extra-index-url https://download.pytorch.org/whl/cpu
```

**From source** (needs a C++20 compiler: MSVC Build Tools on Windows, clang or
gcc elsewhere; CMake fetches zstd on first build):

```
git clone <repo> && cd quaml
pip install ".[panel,torch]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Build a wheel to copy to other machines with `pip wheel . --no-deps -w dist`.
Drop `--extra-index-url` to get a CUDA build of torch instead of CPU-only.

## Forecast forward beta from daily price CSVs

CSV columns: `Code,Symbol,Date,Close,Volume,Adjustment Factor` (dates
`M/D/YYYY`; one row per stock per day; the market proxy, SPY by default, must
be in the files). Files may be split any way (by year, by ticker) and overlap.

```
quaml-forward-beta --csv "data/*.csv" --out predictions.csv
quaml-forward-beta --synthetic               # demo on generated data
quaml-forward-beta --help                    # horizon, models, epochs, market symbol...
```

It compares persistence (trailing beta), OLS, an MLP and a GRU on purged
chronological 60/20/20 splits and writes test-period predictions.

To build your own features and models, see `quaml.panel`:

```python
from quaml import panel as qp

df = qp.read_price_csv("data/*.csv")
p = qp.from_frame(df, asset="Code", date="Date", columns=["adj_close"],
                  calendar=df.filter(df["Symbol"] == "SPY")["Date"])
p = p.with_columns(ret=qp.log_returns(p, "adj_close"))
p = qp.add_market(p, market_asset=<SPY code>, col="ret", name="mkt")

design = qp.design_matrix(
    p,
    features=[qp.rolling_beta("ret", "mkt", 63), qp.lag(qp.rolling_std("ret", 21), 5)],
    label=qp.forward_beta("ret", "mkt", 63),
)
train, val, test = qp.time_split(p, edges, horizon=63)
# design.X[train], design.y[train] -> any model
```

## Tests

```
pip install -e ".[all]" --no-build-isolation   # editable dev install
pytest
```

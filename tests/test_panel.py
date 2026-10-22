import numpy as np
import pytest

pl = pytest.importorskip("polars")

from quaml import panel as qp
from quaml.synthetic import write_synthetic_price_csv


def frame(rows):
    return pl.DataFrame(rows, schema=["asset", "date", "x"], orient="row").with_columns(
        pl.col("date").str.to_date())


def test_from_frame_reindexes_onto_calendar():
    df = frame([(2, "2024-01-03", 3.0), (1, "2024-01-02", 1.0), (1, "2024-01-04", 2.0),
                (2, "2024-01-04", 4.0)])
    cal = ["2024-01-02", "2024-01-03", "2024-01-04"]
    p = qp.from_frame(df, asset="asset", date="date", columns=["x"], calendar=cal)
    np.testing.assert_array_equal(p.asset, [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(p.day, [0, 1, 2, 1, 2])
    np.testing.assert_array_equal(p["x"], [1, np.nan, 2, 3, 4])


def test_from_frame_rejects_duplicates_and_off_calendar_dates():
    with pytest.raises(ValueError, match="duplicate"):
        qp.from_frame(frame([(1, "2024-01-02", 1.0), (1, "2024-01-02", 2.0)]),
                      asset="asset", date="date", columns=["x"])
    with pytest.raises(ValueError, match="outside the calendar"):
        qp.from_frame(frame([(1, "2024-01-06", 1.0)]), asset="asset", date="date",
                      columns=["x"], calendar=["2024-01-05"])


def simple_panel(n_assets=3, n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    asset = np.repeat(np.arange(n_assets), n_days)
    day = np.tile(np.arange(n_days), n_assets)
    cal = np.arange(n_days).astype("datetime64[D]")
    m = rng.normal(size=n_days)
    x = np.tile(m, n_assets)
    y = 1.5 * x + rng.normal(size=len(x))
    y[rng.random(len(y)) < 0.05] = np.nan
    return qp.Panel(asset, day, cal, {"y": y, "m": x})


def test_lag_and_lead_stay_inside_each_asset():
    p = simple_panel(n_days=4)
    lagged = qp.lag("m", 1)(p)
    assert np.isnan(lagged[[0, 4, 8]]).all()
    np.testing.assert_array_equal(lagged[1:4], p["m"][0:3])
    assert np.isnan(qp.lead("m", 1)(p)[[3, 7, 11]]).all()


def test_rolling_beta_matches_polyfit():
    p = simple_panel()
    b = qp.rolling_beta("y", "m", 60, min_obs=40)(p)
    i = 250
    w = slice(i - 59, i + 1)
    ok = np.isfinite(p["y"][w])
    slope = np.polyfit(p["m"][w][ok], p["y"][w][ok], 1)[0]
    assert b[i] == pytest.approx(slope)
    assert np.isnan(b[:20]).all()  # not enough history


def test_forward_beta_is_trailing_beta_shifted():
    p = simple_panel()
    fwd = qp.forward_beta("y", "m", 40, min_obs=30)(p)
    trail = qp.rolling_beta("y", "m", 40, min_obs=30)(p)
    np.testing.assert_allclose(fwd, p.shifted(trail, -40), equal_nan=True)


def test_reach_is_tracked_through_composition():
    op = qp.lag(qp.rolling_std("y", 20), 5)
    assert (op.lookback, op.lookahead) == (24, 0)
    label = qp.forward_beta("y", "m", 63)
    assert (label.lookback, label.lookahead) == (0, 63)
    with pytest.raises(ValueError, match="future rows"):
        qp.design_matrix(simple_panel(), [qp.lead("y", 1)], label)


def test_features_never_see_the_future():
    p = simple_panel()
    features = [qp.lag("y", 3), qp.rolling_beta("y", "m", 30, 20), qp.rolling_std("y", 10),
                qp.derive("ratio", np.divide, qp.rolling_std("y", 5), qp.rolling_std("y", 20))]
    cut = 200
    future = p.day > cut
    rng = np.random.default_rng(1)
    tampered = p.with_columns(y=np.where(future, rng.normal(size=len(p)), p["y"]),
                              m=np.where(future, rng.normal(size=len(p)), p["m"]))
    for f in features:
        np.testing.assert_array_equal(f(p)[~future], f(tampered)[~future], err_msg=f.name)


def test_time_split_purges_label_overlap():
    p = simple_panel()
    edges = p.calendar[[0, 100, 200]].tolist() + [p.calendar[-1] + np.timedelta64(1, "D")]
    parts = qp.time_split(p, edges, horizon=20)
    for rows, hi in zip(parts, [100, 200, 300]):
        assert p.day[rows].max() + 20 < hi  # label window ends inside the period
    assert all(len(np.intersect1d(a, b)) == 0 for a in parts for b in parts if a is not b)


def test_read_synthetic_csv(tmp_path):
    path = tmp_path / "prices.csv"
    write_synthetic_price_csv(path, n_assets=5, n_days=300, seed=3)
    df = qp.read_price_csv(path)
    assert df["Date"].dtype == pl.Date
    assert df["Volume"].dtype == pl.Float64  # survives "1.65E+07"
    cal = df.filter(pl.col("Symbol") == "SPY")["Date"]
    p = qp.from_frame(df, asset="Code", date="Date", columns=["adj_close"], calendar=cal)
    p = p.with_columns(ret=qp.log_returns(p, "adj_close"))
    # Split days would show |log return| > log(2) without the adjustment factor.
    assert np.nanmax(np.abs(p["ret"])) < 0.5
    p = qp.add_market(p, market_asset=1, col="ret", name="mkt")
    spy = p.asset == 1
    np.testing.assert_array_equal(p["mkt"][spy], p["ret"][spy])


def test_read_many_csvs_equals_one(tmp_path):
    whole = tmp_path / "all.csv"
    write_synthetic_price_csv(whole, n_assets=4, n_days=200, seed=4)
    header, *lines = whole.read_text().splitlines()
    parts = tmp_path / "parts"
    parts.mkdir()
    # Three overlapping chunks: rows shared between files must be kept once.
    for k, (a, b) in enumerate([(0, 400), (300, 700), (650, len(lines))]):
        (parts / f"part{k}.csv").write_text("\n".join([header, *lines[a:b]]) + "\n")
    one = qp.read_price_csv(whole)
    for source in (parts, str(parts / "part*.csv"), sorted(parts.glob("*.csv"))):
        many = qp.read_price_csv(source)
        assert many.equals(one)


def test_read_csv_reports_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Code,Symbol,Date,Close\n1,SPY,1/4/2010,100\n")
    with pytest.raises(ValueError, match="missing columns"):
        qp.read_price_csv(bad)

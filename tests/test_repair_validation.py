import numpy as np
import pytest

pl = pytest.importorskip("polars")

from quaml import panel as qp
from quaml import repair, validation


def price_panel(n_days=400, n_assets=3, seed=0):
    rng = np.random.default_rng(seed)
    cal = np.datetime64("2020-01-01") + np.arange(n_days).astype("timedelta64[D]")
    mkt = rng.normal(0, 0.01, n_days)
    asset, day, px = [], [], []
    for a in range(n_assets):
        r = (0.8 + 0.2 * a) * mkt + rng.normal(0, 0.01, n_days)
        asset += [a] * n_days
        day += list(range(n_days))
        px += list(50 * np.exp(np.cumsum(r)))
    asset[:n_days] = [0] * n_days  # asset 0 is the market
    px[:n_days] = list(100 * np.exp(np.cumsum(mkt)))
    return qp.Panel(np.array(asset), np.array(day), cal, {"adj_close": np.array(px)})


def with_returns(p):
    p = p.with_columns(ret=qp.log_returns(p, "adj_close"))
    return qp.add_market(p, 0, "ret", "mkt")


def test_detects_and_repairs_unadjusted_split_and_bad_print():
    p = price_panel()
    px = p["adj_close"].copy()
    a1 = np.flatnonzero(p.asset == 1)
    a2 = np.flatnonzero(p.asset == 2)
    px[a1[:200]] *= 2.0          # missed 2:1 split on day 200: price halves overnight
    px[a2[300]] *= 1.6           # bad print on day 300
    p = p.with_columns(adj_close=px)

    report = repair.detect_jumps(with_returns(p))
    got = {(int(c), int((np.datetime64(d, "D") - p.calendar[0]).astype(int)), k)
           for c, d, k in report.select("code", "date", "kind").iter_rows()}
    assert got == {(1, 200, "split_like"), (2, 300, "reverting")}
    split = report.filter(pl.col("kind") == "split_like")
    assert split["value"][0] == pytest.approx(0.5)

    fixed = with_returns(repair.apply_actions(p, repair.suggested_actions(report)))
    assert np.nanmax(np.abs(fixed["ret"] - fixed["mkt"])) < 0.15
    assert np.isnan(fixed["ret"][a2[[300, 301]]]).all()  # both returns around the bad print


def test_drop_return_and_validation_of_actions():
    p = price_panel()
    actions = pl.DataFrame({"code": [1], "date": p.calendar[[50]], "action": ["drop_return"]})
    r = with_returns(repair.apply_actions(p, actions))["ret"]
    rows = np.flatnonzero(p.asset == 1)
    assert np.isnan(r[rows[50]]) and np.isfinite(r[rows[[49, 51]]]).all()

    bad = pl.DataFrame({"code": [9, 1, 1], "date": p.calendar[[5, 5, 5]],
                        "action": ["bad_price", "teleport", "rescale_before"]})
    with pytest.raises(ValueError) as e:
        repair.apply_actions(p, bad)
    msg = str(e.value)
    assert "not in panel" in msg and "unknown action" in msg and "positive value" in msg


def test_actions_from_csv(tmp_path):
    p = price_panel()
    path = tmp_path / "actions.csv"
    path.write_text("Code,Date,Action,Value\n1,1/11/2020,rescale_before,0.5\n")
    fixed = repair.apply_actions(p, path)
    rows = np.flatnonzero(p.asset == 1)
    np.testing.assert_allclose(fixed["adj_close"][rows[:10]], p["adj_close"][rows[:10]] * 0.5)
    np.testing.assert_array_equal(fixed["adj_close"][rows[10:]], p["adj_close"][rows[10:]])


def test_walk_forward_keeps_label_windows_apart():
    p = price_panel(n_days=1000)
    h = 63
    folds = validation.walk_forward(p, horizon=h, n_folds=3, first_test=0.5, val_days=100)
    for f in folds:
        tr, va, te = (p.day[r] for r in (f.train, f.val, f.test))
        assert tr.max() + h < va.min()          # training labels end before validation
        assert va.max() + h < te.min()          # validation labels end before test
    tests = np.concatenate([p.day[f.test] for f in folds])
    assert len(np.unique(tests)) == 500 and tests.min() == 500  # tiles the back half
    with pytest.raises(ValueError, match="no room"):
        validation.walk_forward(p, horizon=h, n_folds=3, first_test=0.1, val_days=100)


def test_block_bootstrap_interval():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 2000)
    mean, lo, hi = validation.block_bootstrap(x, block=20)
    assert lo < 0 < hi and lo < mean < hi
    mean, lo, hi = validation.block_bootstrap(x + 1.0, block=20)
    assert lo > 0.8


def test_compare_detects_a_real_improvement_only():
    rng = np.random.default_rng(1)
    day = np.repeat(np.arange(500), 50)
    fold = day // 125
    y = rng.normal(size=len(day))
    ref = y + rng.normal(0, 1.0, len(y))
    good = y + rng.normal(0, 0.8, len(y))
    same = ref + rng.normal(0, 1e-3, len(y))
    c = validation.compare(good, ref, y, day, fold, block=10)
    assert c.significant and c.diff < 0 and c.folds_better == 4
    c = validation.compare(same, ref, y, day, fold, block=10)
    assert not c.significant


def test_holdout_lock(tmp_path):
    p = with_returns(price_panel())
    lock = validation.HoldoutLock("2020-12-01", tmp_path / "log.jsonl")
    dev = lock.development(p)
    assert dev.calendar[-1] < np.datetime64("2020-12-01")
    assert len(dev) < len(p) and dev.day.max() < len(dev.calendar)
    # Labels near the cut read past it, so on the development panel they are undefined.
    fwd = qp.forward_beta("ret", "mkt", 20)(dev)
    last = dev.day == dev.day.max()
    assert np.isnan(fwd[last]).all()

    with pytest.raises(ValueError, match="reason"):
        lock.open(p, reason=" ")
    rows = lock.open(p, reason="final score")
    assert (p.date[rows] >= np.datetime64("2020-12-01")).all()
    with pytest.raises(PermissionError, match="already opened 1 time"):
        lock.open(p, reason="peek again")
    assert len(lock.open(p, reason="agreed rerun", allow_repeat=True)) == len(rows)
    assert len(lock.openings()) == 2

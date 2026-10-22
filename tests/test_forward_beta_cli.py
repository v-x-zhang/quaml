"""End-to-end tests of the quaml-forward-beta command on small synthetic panels."""

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("torch")

from quaml.experiments import forward_beta
from quaml.synthetic import write_synthetic_price_csv

SMALL = ["--epochs", "1", "--folds", "2", "--first-test", "0.6", "--val-days", "100",
         "--boot", "200"]


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "prices.csv"
    made = write_synthetic_price_csv(path, n_assets=20, n_days=1400, seed=5, corrupt=True)
    assert made["corruptions"]
    return path


def test_walk_forward_run_writes_out_of_fold_predictions(csv, tmp_path, capsys):
    out = tmp_path / "preds.csv"
    forward_beta.main(["--csv", str(csv), "--out", str(out), "--auto-repair", *SMALL])

    printed = capsys.readouterr().out
    assert "auto-repair: applied" in printed and "price jumps still flagged: 0" in printed
    assert "fold 0:" in printed and "fold 1:" in printed
    for model in forward_beta.MODELS:
        assert model in printed
    preds = pl.read_csv(out)
    assert preds.columns == ["fold", "date", "code", "symbol", "realised_beta",
                             *(f"pred_{m}" for m in forward_beta.MODELS)]
    assert sorted(preds["fold"].unique()) == [0, 1]
    assert preds.null_count().sum_horizontal()[0] == 0


def test_jump_report_lists_the_corruptions(csv, tmp_path, capsys):
    report = tmp_path / "jumps.csv"
    forward_beta.main(["--csv", str(csv), "--jump-report", str(report), "--models",
                       "persistence,ols", *SMALL])
    kinds = set(pl.read_csv(report)["kind"])
    assert {"split_like", "reverting"} <= kinds


def test_holdout_is_sealed_then_opened_once(csv, tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    common = ["--csv", str(csv), "--auto-repair", "--models", "persistence,ols",
              "--holdout-start", "2014-06-02", "--holdout-log", str(log), *SMALL]
    forward_beta.main(common)
    printed = capsys.readouterr().out
    assert "holdout sealed from 2014-06-02" in printed and not log.exists()

    forward_beta.main([*common, "--open-holdout", "final check"])
    assert "HOLDOUT OPENED" in capsys.readouterr().out
    with pytest.raises(PermissionError):
        forward_beta.main([*common, "--open-holdout", "one more look"])


def test_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        forward_beta.main(["--synthetic", "--models", "ols,xgboost"])
    with pytest.raises(SystemExit):
        forward_beta.main(["--synthetic", "--open-holdout", "no date given"])

"""End-to-end smoke test of the quaml-forward-beta command on a small synthetic panel."""

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("torch")

from quaml.experiments import forward_beta
from quaml.synthetic import write_synthetic_price_csv


def test_runs_all_models_and_writes_predictions(tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    write_synthetic_price_csv(csv, n_assets=20, n_days=900, seed=5)
    out = tmp_path / "preds.csv"
    forward_beta.main(["--csv", str(csv), "--epochs", "1", "--out", str(out)])

    printed = capsys.readouterr().out
    for model in forward_beta.MODELS:
        assert model in printed
    preds = pl.read_csv(out)
    assert preds.columns == ["date", "code", "symbol", "realised_beta",
                             *(f"pred_{m}" for m in forward_beta.MODELS)]
    assert preds.height > 0
    assert preds.null_count().sum_horizontal()[0] == 0


def test_rejects_unknown_model(tmp_path):
    with pytest.raises(SystemExit):
        forward_beta.main(["--synthetic", "--models", "ols,xgboost"])

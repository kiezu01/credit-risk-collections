"""End-to-end run on the synthetic sample: dbt build (with data tests) + model training."""

import json
import os

import pytest

from credit_risk_collections import cli, sample_data


@pytest.fixture(scope="module")
def sample_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("crc")
    raw = sample_data.generate(root / "raw", n_train=2500, n_test=300, seed=7)
    env = {
        "CRC_RAW_DIR": str(raw),
        "CRC_DUCKDB_PATH": str(root / "warehouse.duckdb"),
        "CRC_REPORTS_DIR": str(root / "reports"),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    yield root
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_build_and_train_end_to_end(sample_env):
    cli.build()  # raises if any dbt model or data test fails
    results = cli.train()

    assert 0.5 <= results["collections"]["oot"]["auc"] <= 1.0
    captured = results["collections"]["uncured_amt_captured_at_30pct"]
    # any sensible ordering must beat a random queue
    assert captured["expected_uncured_amt"] > captured["random"]

    reports = sample_env / "reports"
    assert (
        json.loads((reports / "results.json").read_text())["scorecard"]["n_selected_features"] > 0
    )
    for name in ["pd_roc.png", "collections_capture_curve.png"]:
        assert (reports / "figures" / name).exists()
    assert (reports / "app_data" / "collections_capture.csv").exists()

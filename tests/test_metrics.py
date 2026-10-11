import numpy as np
import pytest

from credit_risk_collections import metrics


def test_ks_is_one_for_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert metrics.ks_statistic(y, s) == pytest.approx(1.0)


def test_gini_matches_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)
    out = metrics.summary(y, s)
    assert out["gini"] == pytest.approx(2 * out["auc"] - 1, abs=1e-3)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20_000)
    assert metrics.psi(base, rng.normal(0, 1, 20_000)) < 0.01
    assert metrics.psi(base, rng.normal(1.5, 1, 20_000)) > 0.25


def test_calibration_table_has_requested_bins():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    table = metrics.calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 1000

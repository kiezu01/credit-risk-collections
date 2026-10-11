"""Model evaluation metrics used in credit risk: AUC/Gini, KS, PSI, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y_true, y_score) -> float:
    """Gini = 2*AUC - 1, the usual way scorecard power is reported."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    """Max distance between the score CDFs of bads (y=1) and goods (y=0)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score samples.

    Bins come from quantiles of ``expected``. Rule of thumb: < 0.10 stable,
    0.10-0.25 monitor, > 0.25 significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate by probability decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )


def summary(y_true, y_score) -> dict[str, float]:
    """AUC, Gini and KS for a score where higher = more likely y=1."""
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
    }

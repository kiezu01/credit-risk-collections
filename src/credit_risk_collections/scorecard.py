"""Application scorecard: WoE binning + logistic regression, scaled to points.

Higher score = lower risk. Scaling: 600 points at 50:1 good:bad odds, 20 points
to double the odds (PDO), a common industry convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from optbinning import Scorecard as OBScorecard
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import metrics
from .config import SEED

TARGET = "target_default"
ID = "customer_id"
GRADES = ["A", "B", "C", "D", "E", "F"]  # A = lowest risk
MIN_IV = 0.02


@dataclass
class ScorecardResult:
    model: OBScorecard
    features: list[str]
    train: pd.DataFrame  # customer_id, target, score, grade
    test: pd.DataFrame
    metrics: dict
    iv_table: pd.DataFrame
    points_table: pd.DataFrame
    grade_table: pd.DataFrame
    cutoff_table: pd.DataFrame
    grade_edges: list[float]


def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # fixed row order keeps the random split reproducible across warehouse rebuilds
    df = con.sql("select * from mart_customer_features order by customer_id").df()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def split(df: pd.DataFrame, test_size: float = 0.3):
    return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=SEED)


def _feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feats = [c for c in df.columns if c not in (ID, TARGET)]
    cats = [c for c in feats if df[c].dtype == object or str(df[c].dtype) in ("string", "str")]
    return feats, cats


def assign_grades(scores: np.ndarray, edges: list[float]) -> np.ndarray:
    """Map scores to A-F. ``edges`` are the 5 inner cut points, ascending."""
    # lowest scores -> F, highest -> A
    idx = np.searchsorted(edges, scores, side="right")
    return np.array(GRADES[::-1])[idx]


def grade_report(frame: pd.DataFrame) -> pd.DataFrame:
    g = (
        frame.groupby("grade")
        .agg(
            customers=(TARGET, "size"),
            defaults=(TARGET, "sum"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reindex(GRADES)
        .reset_index()
    )
    g["share"] = g["customers"] / g["customers"].sum()
    g["default_rate"] = g["defaults"] / g["customers"]
    return g


def cutoff_report(frame: pd.DataFrame, steps: int = 20) -> pd.DataFrame:
    """Approve everyone at or above a cut-off: approval rate vs. default rate trade-off."""
    rows = []
    total_bads = frame[TARGET].sum()
    for q in np.linspace(0, 0.95, steps):
        cut = float(np.quantile(frame["score"], q))
        approved = frame[frame["score"] >= cut]
        rows.append(
            {
                "cutoff_score": round(cut, 1),
                "approval_rate": len(approved) / len(frame),
                "default_rate_approved": approved[TARGET].mean(),
                "defaults_avoided_share": 1 - approved[TARGET].sum() / total_bads,
            }
        )
    return pd.DataFrame(rows)


def train(df: pd.DataFrame) -> ScorecardResult:
    train_df, test_df = split(df)
    feats, cats = _feature_lists(df)

    binning = BinningProcess(
        variable_names=feats,
        categorical_variables=cats,
        selection_criteria={"iv": {"min": MIN_IV}},
        min_prebin_size=0.02,
    )
    model = OBScorecard(
        binning_process=binning,
        estimator=LogisticRegression(max_iter=2000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
    )
    model.fit(train_df[feats], train_df[TARGET])

    out = {}
    for name, part in (("train", train_df), ("test", test_df)):
        scores = model.score(part[feats])
        out[name] = pd.DataFrame(
            {ID: part[ID].to_numpy(), TARGET: part[TARGET].to_numpy(), "score": scores}
        )

    # grade cut points from the train score distribution (equal-sized groups)
    edges = list(np.quantile(out["train"]["score"], np.linspace(0, 1, 7)[1:-1]))
    for part in out.values():
        part["grade"] = assign_grades(part["score"].to_numpy(), edges)

    # higher score = lower risk, so negate it to measure power on "default"
    m = {name: metrics.summary(part[TARGET], -part["score"]) for name, part in out.items()}
    m["psi_train_vs_test"] = round(metrics.psi(out["train"]["score"], out["test"]["score"]), 4)

    # the scorecard fits a clone of the binning process
    iv = model.binning_process_.summary().sort_values("iv", ascending=False)
    selected = iv.loc[iv["selected"], "name"].tolist()
    m["n_candidate_features"] = len(feats)
    m["n_selected_features"] = len(selected)
    m["default_rate_train"] = round(float(out["train"][TARGET].mean()), 4)

    return ScorecardResult(
        model=model,
        features=feats,
        train=out["train"],
        test=out["test"],
        metrics=m,
        iv_table=iv[["name", "dtype", "selected", "n_bins", "iv", "gini"]].reset_index(drop=True),
        points_table=model.table(style="detailed")[
            ["Variable", "Bin", "Count", "Event rate", "WoE", "IV", "Points"]
        ],
        grade_table=grade_report(out["test"]),
        cutoff_table=cutoff_report(out["test"]),
        grade_edges=[round(e, 1) for e in edges],
    )

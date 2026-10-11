"""Early-delinquency collections: who will pay without help, and who to call first.

1. Roll-rate model: probability that an overdue instalment rolls to DPD 30+
   (is not paid within the cure window), trained on older events and validated
   on the most recent ones (out-of-time), the way it would be used in production.
2. Prioritisation: collectors can only work part of each day's new delinquencies.
   We compare queue-ordering strategies by how much of the money that is NOT
   self-cured lands in the worked share of the queue. This uses observed outcomes
   only, no assumed treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from . import challenger, metrics
from .config import SEED

TARGET = "is_cured"
FEATURES = [
    "instalment_number",
    "due_amt",
    "shortfall_amt",
    "shortfall_ratio",
    "is_partial_payment",
    "prior_inst_cnt",
    "prior_delinquent_cnt",
    "prior_delinquent_rate",
    "prior_known_entry_cnt",
    "prior_cure_rate",
    "days_since_last_entry",
    "loan_prior_delinquent_cnt",
    "loan_prior_inst_cnt",
    "loan_contract_type",
    "loan_credit_amt",
    "loan_term_months",
    "loan_progress_ratio",
    "gender",
    "education_type",
    "family_status",
    "income_type",
    "age_at_due_years",
]
STRATEGIES = {
    "random": "Random order (no prioritisation)",
    "largest_shortfall": "Largest overdue amount first (common rule of thumb)",
    "lowest_cure_prob": "Highest roll probability first (model)",
    "expected_uncured_amt": "Highest expected rolling amount first (model x amount)",
}
CAPACITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class CollectionsResult:
    metrics: dict
    capture_curve: pd.DataFrame
    calibration: pd.DataFrame
    shap_importance: pd.DataFrame
    cure_rate_by_month: pd.DataFrame
    oot_scored: pd.DataFrame


def load_events(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.sql("select * from mart_collections_events order by event_id").df()
    df[TARGET] = df[TARGET].astype(int)
    df["is_partial_payment"] = df["is_partial_payment"].astype(int)
    return df


def out_of_time_split(df: pd.DataFrame, oot_share: float = 0.3):
    """Most recent ``oot_share`` of events (by due date) become the validation set."""
    cut = df["due_day"].quantile(1 - oot_share)
    return df[df["due_day"] < cut].copy(), df[df["due_day"] >= cut].copy(), float(cut)


def capture_curve(scored: pd.DataFrame, cohort_days: int = 7) -> pd.DataFrame:
    """Share of uncured money reached when working the top X% of each cohort's queue.

    Events are grouped into cohorts of ``cohort_days`` by due date (the queue a team
    would see), ranked within the cohort by each strategy, and the top share is worked.
    """
    df = scored.copy()
    df["uncured_amt"] = np.where(df[TARGET] == 0, df["shortfall_amt"], 0.0)
    df["cohort"] = (df["due_day"] // cohort_days).astype(int)
    rng = np.random.default_rng(SEED)
    keys = {
        "random": rng.random(len(df)),
        "largest_shortfall": df["shortfall_amt"].to_numpy(),
        "lowest_cure_prob": 1 - df["p_cure"].to_numpy(),
        "expected_uncured_amt": ((1 - df["p_cure"]) * df["shortfall_amt"]).to_numpy(),
    }
    total_uncured = df["uncured_amt"].sum()
    total_uncured_events = (df[TARGET] == 0).sum()
    rows = []
    for name, key in keys.items():
        df["_key"] = key
        pct = df.groupby("cohort")["_key"].rank(method="first", ascending=False, pct=True)
        for cap in CAPACITIES:
            worked = df[pct <= cap]
            rows.append(
                {
                    "strategy": name,
                    "capacity": cap,
                    "uncured_amt_captured": worked["uncured_amt"].sum() / total_uncured,
                    "uncured_events_captured": (worked[TARGET] == 0).sum() / total_uncured_events,
                    "self_cured_share_of_worked": worked[TARGET].mean(),
                }
            )
    return pd.DataFrame(rows)


def run(con: duckdb.DuckDBPyConnection) -> CollectionsResult:
    events = load_events(con)
    train_df, oot_df, cut = out_of_time_split(events)

    res = challenger.train(train_df, oot_df, FEATURES, TARGET, shap_rows=5000)
    oot_df["p_cure"] = res.test_pred

    curve = capture_curve(oot_df)
    at = curve[curve["capacity"] == 0.3].set_index("strategy")["uncured_amt_captured"]

    monthly = (
        events.assign(month=(events["due_day"] // 30).astype(int))
        .groupby("month")
        .agg(events=(TARGET, "size"), cure_rate=(TARGET, "mean"))
        .reset_index()
    )

    m = {
        "events_total": int(len(events)),
        "events_train": int(len(train_df)),
        "events_oot": int(len(oot_df)),
        "oot_cut_due_day": cut,
        "cure_rate_train": round(float(train_df[TARGET].mean()), 4),
        "cure_rate_oot": round(float(oot_df[TARGET].mean()), 4),
        "oot": res.metrics["test"],
        "psi_p_cure_train_vs_oot": round(
            metrics.psi(
                res.model.predict_proba(challenger.prepare(train_df, FEATURES))[:, 1],
                oot_df["p_cure"],
            ),
            4,
        ),
        "uncured_amt_captured_at_30pct": {k: round(float(v), 4) for k, v in at.items()},
    }
    return CollectionsResult(
        metrics=m,
        capture_curve=curve,
        calibration=metrics.calibration_table(oot_df[TARGET], oot_df["p_cure"]),
        shap_importance=res.shap_importance,
        cure_rate_by_month=monthly,
        oot_scored=oot_df[["event_id", "due_day", "shortfall_amt", TARGET, "p_cure"]],
    )

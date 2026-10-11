"""Interactive results explorer. Reads only the aggregate files in reports/.

Run locally:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_risk_collections.experiment import sample_size_per_arm  # noqa: E402

REPORTS = Path(os.environ.get("CRC_REPORTS_DIR", ROOT / "reports"))
DATA = REPORTS / "app_data"
NAVY, ORANGE, GREY = "#1F3864", "#E07B39", "#9AA5B1"
STRATEGY_LABELS = {
    "expected_uncured_amt": "Model x amount (expected uncured amount)",
    "largest_shortfall": "Largest overdue amount first",
    "lowest_cure_prob": "Least likely to self-cure first",
    "random": "Random order",
}
STRATEGY_COLORS = {
    "expected_uncured_amt": NAVY,
    "largest_shortfall": ORANGE,
    "lowest_cure_prob": "#6C8EBF",
    "random": GREY,
}

st.set_page_config(page_title="Credit Risk & Collections Analytics", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def results() -> dict:
    return json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))


if not (REPORTS / "results.json").exists():
    st.error("No results yet. Run `uv run crc all` first.")
    st.stop()

R = results()
sc, ch, col = R["scorecard"], R["challenger"], R["collections"]

st.title("Credit Risk & Collections Analytics")
st.caption(
    "Home Credit Default Risk data · dbt + DuckDB warehouse · WoE scorecard · LightGBM + SHAP · "
    "out-of-time cure model · A/B test design. "
    "[Code on GitHub](https://github.com/kiezu01/credit-risk-collections)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scorecard Gini (test)", f"{sc['test']['gini']:.3f}")
c2.metric("LightGBM Gini (test)", f"{ch['test']['gini']:.3f}")
c3.metric("Cure model AUC (out-of-time)", f"{col['oot']['auc']:.3f}")
cap = col["uncured_amt_captured_at_30pct"]
c4.metric(
    "Uncured amount reached at 30% capacity",
    f"{cap['expected_uncured_amt']:.0%}",
    f"{(cap['expected_uncured_amt'] - cap['random']) * 100:+.0f} pp vs random",
)

tab_pd, tab_col, tab_ab, tab_about = st.tabs(
    [
        "1 · Application scorecard",
        "2 · Collections prioritisation",
        "3 · A/B test design",
        "About the method",
    ]
)

# --------------------------------------------------------------------------- scorecard
with tab_pd:
    grades = load("pd_grades.csv")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Default rate by risk grade")
        fig = px.bar(
            grades, x="grade", y="default_rate", text_auto=".1%", color_discrete_sequence=[NAVY]
        )
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Default rate", xaxis_title="Grade")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Six equal-sized grades cut on the training score distribution; test set shown.")
    with right:
        st.subheader("Where to set the approval cut-off?")
        cut = load("pd_cutoffs.csv")
        approval = st.slider("Target approval rate", 0.05, 1.0, 0.8, 0.05, format="%.2f")
        row = cut.iloc[(cut["approval_rate"] - approval).abs().argmin()]
        a, b, c = st.columns(3)
        a.metric("Cut-off score", f"{row['cutoff_score']:.0f}")
        b.metric("Default rate of approved", f"{row['default_rate_approved']:.2%}")
        c.metric("Defaults avoided", f"{row['defaults_avoided_share']:.0%}")
        fig = go.Figure()
        fig.add_scatter(
            x=cut["approval_rate"],
            y=cut["default_rate_approved"],
            mode="lines+markers",
            line=dict(color=NAVY),
        )
        fig.add_scatter(
            x=[row["approval_rate"]],
            y=[row["default_rate_approved"]],
            mode="markers",
            marker=dict(size=14, color=ORANGE),
            showlegend=False,
        )
        fig.update_layout(
            xaxis_title="Approval rate",
            yaxis_title="Default rate of approved",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".1%",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Feature strength (Information Value)")
        iv = load("pd_iv.csv")
        st.dataframe(
            iv[iv["selected"]].head(20)[["name", "iv", "n_bins"]],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("Scorecard vs. black-box challenger")
        shap_pd = load("pd_shap.csv").head(12).iloc[::-1]
        fig = px.bar(
            shap_pd,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color_discrete_sequence=[ORANGE],
            title="LightGBM drivers (mean |SHAP|)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"The transparent scorecard reaches Gini {sc['test']['gini']:.3f} against "
            f"{ch['test']['gini']:.3f} for LightGBM on the same test set."
        )

    st.subheader("Points table")
    points = load("pd_points.csv")
    var = st.selectbox("Variable", sorted(points["Variable"].unique()))
    st.dataframe(points[points["Variable"] == var], hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- collections
with tab_col:
    st.markdown(
        "Collectors cannot call every overdue customer on day one. Many will pay anyway "
        "(**self-cure**). The question: **which share of the queue should be worked first so "
        "the money that would otherwise stay unpaid reaches a collector?** Measured on the most "
        "recent 30% of events (out-of-time), with real outcomes only."
    )
    curve = load("collections_capture.csv")
    capacity = st.slider(
        "Share of each cohort's queue the team can work", 0.1, 1.0, 0.3, 0.1, format="%.1f"
    )
    at = curve[curve["capacity"].round(2) == round(capacity, 2)].set_index("strategy")
    cols = st.columns(4)
    for col_, key in zip(cols, STRATEGY_LABELS, strict=True):
        col_.metric(
            STRATEGY_LABELS[key],
            f"{at.loc[key, 'uncured_amt_captured']:.0%}",
            f"{at.loc[key, 'self_cured_share_of_worked']:.0%} of worked would self-cure",
            delta_color="off",
        )
    fig = go.Figure()
    for key, label in STRATEGY_LABELS.items():
        part = curve[curve["strategy"] == key]
        fig.add_scatter(
            x=part["capacity"],
            y=part["uncured_amt_captured"],
            name=label,
            mode="lines+markers",
            line=dict(color=STRATEGY_COLORS[key], width=4 if key == "expected_uncured_amt" else 2),
        )
    fig.add_vline(x=capacity, line_dash="dot", line_color=GREY)
    fig.update_layout(
        xaxis_title="Share of queue worked",
        yaxis_title="Uncured amount reached",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        cal = load("collections_calibration.csv")
        fig = go.Figure()
        fig.add_scatter(
            x=cal["predicted"],
            y=cal["observed"],
            mode="lines+markers",
            line=dict(color=NAVY),
            name="Model",
        )
        lo, hi = (
            cal[["predicted", "observed"]].min().min(),
            cal[["predicted", "observed"]].max().max(),
        )
        fig.add_scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color=GREY), name="Perfect"
        )
        fig.update_layout(
            title="Calibration (out-of-time)",
            xaxis_title="Predicted cure probability",
            yaxis_title="Observed cure rate",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        shap_c = load("collections_shap.csv").head(12).iloc[::-1]
        fig = px.bar(
            shap_c,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color_discrete_sequence=[NAVY],
            title="Cure model drivers (mean |SHAP|)",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{col['events_total']:,} delinquency events · trained on {col['events_train']:,}, "
        f"validated on the latest {col['events_oot']:,} · cure rate {col['cure_rate_oot']:.1%} "
        f"(validation) · PSI train vs. validation {col['psi_p_cure_train_vs_oot']:.3f}"
    )

# --------------------------------------------------------------------------- experiment
with tab_ab:
    st.markdown(
        "The analysis above shows the model routes more at-risk money to collectors. It does "
        "**not** prove that calling changes the outcome. That needs a randomised test: split "
        "the prioritised queue, call one half with the new strategy, keep the current process "
        "for the other, compare the **roll rate to DPD 30+**."
    )
    design = load("experiment_design.csv")
    base_default = float(design["baseline_roll_rate"].iloc[0])
    a, b, c = st.columns(3)
    baseline = a.number_input(
        "Baseline roll rate to DPD 30+", 0.001, 0.9, base_default, 0.005, format="%.3f"
    )
    cut = b.number_input("Smallest relative reduction worth detecting", 0.01, 0.9, 0.20, 0.01)
    per_day = c.number_input("New DPD 5 entries per day", 100, 100_000, 2000, 100)
    n = sample_size_per_arm(baseline, baseline * (1 - cut))
    a2, b2, c2 = st.columns(3)
    a2.metric(
        "Accounts per arm",
        f"{n:,}",
        f"{baseline:.2%} → {baseline * (1 - cut):.2%}",
        delta_color="off",
    )
    b2.metric("Total accounts", f"{2 * n:,}")
    c2.metric("Days to run", f"{math.ceil(2 * n / per_day)}")
    st.dataframe(design, hide_index=True, use_container_width=True)
    st.caption(
        "Two-sided z-test on two proportions, alpha 0.05, power 0.80. Baseline defaults to the "
        "roll rate of the accounts the model would put in front of collectors (top 30% by "
        "expected uncured amount). Daily volume is an assumption: replace it with your portfolio's."
    )

# --------------------------------------------------------------------------- about
with tab_about:
    st.markdown((ROOT / "docs" / "methodology.md").read_text(encoding="utf-8"))

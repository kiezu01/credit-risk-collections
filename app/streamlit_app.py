"""Interactive results explorer. Reads only the aggregate files in reports/.

Run locally:  uv run streamlit run app/streamlit_app.py
English by default; add ?lang=vi to the URL (or use the switch) for Vietnamese.
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
STRATEGY_COLORS = {
    "expected_uncured_amt": NAVY,
    "largest_shortfall": ORANGE,
    "lowest_cure_prob": "#6C8EBF",
    "random": GREY,
}
LANGUAGES = {"en": "English", "vi": "Tiếng Việt"}

st.set_page_config(page_title="Credit Risk & Collections Analytics", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def results() -> dict:
    return json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- language
head, switch = st.columns([5, 1])
with switch:
    default = st.query_params.get("lang", "en")
    lang = st.radio(
        "Language",
        list(LANGUAGES),
        index=list(LANGUAGES).index(default) if default in LANGUAGES else 0,
        format_func=LANGUAGES.get,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.query_params["lang"] = lang
VI = lang == "vi"


def t(en: str, vi: str) -> str:
    return vi if VI else en


STRATEGY_LABELS = {
    "expected_uncured_amt": t(
        "Model x amount (expected rolling amount)", "Mô hình x số tiền (số tiền dự kiến roll)"
    ),
    "largest_shortfall": t("Largest overdue amount first", "Số tiền quá hạn lớn nhất trước"),
    "lowest_cure_prob": t("Highest roll probability first", "Xác suất roll cao nhất trước"),
    "random": t("Random order", "Thứ tự ngẫu nhiên"),
}

if not (REPORTS / "results.json").exists():
    st.error(
        t(
            "No results yet. Run `uv run crc all` first.",
            "Chưa có kết quả. Chạy `uv run crc all` trước.",
        )
    )
    st.stop()

R = results()
sc, ch, col = R["scorecard"], R["challenger"], R["collections"]

with head:
    st.title("Credit Risk & Collections Analytics")
st.caption(
    t(
        "Home Credit Default Risk data · dbt + DuckDB warehouse · WoE scorecard · "
        "LightGBM + SHAP · out-of-time roll-rate model · A/B test design. ",
        "Dữ liệu Home Credit Default Risk · kho dữ liệu dbt + DuckDB · scorecard WoE · "
        "LightGBM + SHAP · mô hình roll rate kiểm định out-of-time · thiết kế A/B test. ",
    )
    + t("[Code on GitHub]", "[Mã nguồn trên GitHub]")
    + "(https://github.com/kiezu01/credit-risk-collections)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("Scorecard Gini (test)", "Gini scorecard (tập kiểm tra)"), f"{sc['test']['gini']:.3f}")
c2.metric(t("LightGBM Gini (test)", "Gini LightGBM (tập kiểm tra)"), f"{ch['test']['gini']:.3f}")
c3.metric(
    t("Roll-rate model AUC (out-of-time)", "AUC mô hình roll rate (out-of-time)"),
    f"{col['oot']['auc']:.3f}",
)
cap = col["uncured_amt_captured_at_30pct"]
c4.metric(
    t("Rolling amount reached at 30% capacity", "Số tiền roll được tiếp cận ở 30% năng lực"),
    f"{cap['expected_uncured_amt']:.0%}",
    f"{(cap['expected_uncured_amt'] - cap['random']) * 100:+.0f} "
    + t("pp vs random", "điểm % so với ngẫu nhiên"),
)

tab_pd, tab_col, tab_ab, tab_about = st.tabs(
    [
        t("1 · Application scorecard", "1 · Scorecard phê duyệt"),
        t("2 · Collections prioritisation", "2 · Ưu tiên thu hồi nợ"),
        t("3 · A/B test design", "3 · Thiết kế A/B test"),
        t("About the method", "Phương pháp"),
    ]
)

# --------------------------------------------------------------------------- scorecard
with tab_pd:
    grades = load("pd_grades.csv")
    left, right = st.columns([1, 1])
    with left:
        st.subheader(t("Default rate by risk grade", "Tỷ lệ vỡ nợ theo hạng rủi ro"))
        fig = px.bar(
            grades, x="grade", y="default_rate", text_auto=".1%", color_discrete_sequence=[NAVY]
        )
        fig.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title=t("Default rate", "Tỷ lệ vỡ nợ"),
            xaxis_title=t("Grade", "Hạng"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            t(
                "Six equal-sized grades cut on the training score distribution; test set shown.",
                "Sáu hạng có số lượng bằng nhau, cắt theo phân phối điểm tập huấn luyện; "
                "biểu đồ hiển thị tập kiểm tra.",
            )
        )
    with right:
        st.subheader(t("Where to set the approval cut-off?", "Đặt điểm cắt phê duyệt ở đâu?"))
        cut = load("pd_cutoffs.csv")
        approval = st.slider(
            t("Target approval rate", "Tỷ lệ phê duyệt mục tiêu"),
            0.05,
            1.0,
            0.8,
            0.05,
            format="%.2f",
        )
        row = cut.iloc[(cut["approval_rate"] - approval).abs().argmin()]
        a, b, c = st.columns(3)
        a.metric(t("Cut-off score", "Điểm cắt"), f"{row['cutoff_score']:.0f}")
        b.metric(
            t("Default rate of approved", "Tỷ lệ vỡ nợ của hồ sơ duyệt"),
            f"{row['default_rate_approved']:.2%}",
        )
        c.metric(
            t("Defaults avoided", "Số ca vỡ nợ tránh được"), f"{row['defaults_avoided_share']:.0%}"
        )
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
            xaxis_title=t("Approval rate", "Tỷ lệ phê duyệt"),
            yaxis_title=t("Default rate of approved", "Tỷ lệ vỡ nợ của hồ sơ duyệt"),
            xaxis_tickformat=".0%",
            yaxis_tickformat=".1%",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader(t("Feature strength (Information Value)", "Sức mạnh biến (Information Value)"))
        iv = load("pd_iv.csv")
        st.dataframe(
            iv[iv["selected"]].head(20)[["name", "iv", "n_bins"]],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader(t("Scorecard vs. black-box challenger", "Scorecard so với mô hình hộp đen"))
        shap_pd = load("pd_shap.csv").head(12).iloc[::-1]
        fig = px.bar(
            shap_pd,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color_discrete_sequence=[ORANGE],
            title=t("LightGBM drivers (mean |SHAP|)", "Yếu tố tác động của LightGBM (mean |SHAP|)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            t(
                f"The transparent scorecard reaches Gini {sc['test']['gini']:.3f} against "
                f"{ch['test']['gini']:.3f} for LightGBM on the same test set.",
                f"Scorecard minh bạch đạt Gini {sc['test']['gini']:.3f}, so với "
                f"{ch['test']['gini']:.3f} của LightGBM trên cùng tập kiểm tra.",
            )
        )

    st.subheader(t("Points table", "Bảng điểm"))
    points = load("pd_points.csv")
    var = st.selectbox(t("Variable", "Biến"), sorted(points["Variable"].unique()))
    st.dataframe(points[points["Variable"] == var], hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- collections
with tab_col:
    st.markdown(
        t(
            "Collectors cannot call every overdue customer on day one. Many will pay anyway "
            "(**self-cure**). The question: **which share of the queue should be worked first so "
            "the money that would otherwise stay unpaid reaches a collector?** Measured on the "
            "most recent 30% of events (out-of-time), with real outcomes only.",
            "Nhân viên thu hồi không thể gọi mọi khách hàng quá hạn ngay ngày đầu. Nhiều người "
            "vẫn sẽ tự trả (**self-cure**). Câu hỏi: **nên xử lý phần nào của hàng đợi trước để "
            "số tiền lẽ ra không được trả đến được tay nhân viên thu hồi?** Đo trên 30% sự kiện "
            "mới nhất (out-of-time), chỉ dùng kết quả thực tế.",
        )
    )
    curve = load("collections_capture.csv")
    capacity = st.slider(
        t(
            "Share of each cohort's queue the team can work",
            "Tỷ lệ hàng đợi mỗi nhóm đội xử lý được",
        ),
        0.1,
        1.0,
        0.3,
        0.1,
        format="%.1f",
    )
    at = curve[curve["capacity"].round(2) == round(capacity, 2)].set_index("strategy")
    cols = st.columns(4)
    for col_, key in zip(cols, STRATEGY_LABELS, strict=True):
        col_.metric(
            STRATEGY_LABELS[key],
            f"{at.loc[key, 'uncured_amt_captured']:.0%}",
            f"{at.loc[key, 'self_cured_share_of_worked']:.0%} "
            + t("of worked would self-cure", "số khoản xử lý vẫn sẽ tự trả"),
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
        xaxis_title=t("Share of queue worked", "Tỷ lệ hàng đợi được xử lý"),
        yaxis_title=t("Share of rolling amount reached", "Tỷ lệ số tiền roll được tiếp cận"),
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
            x=1 - cal["predicted"],
            y=1 - cal["observed"],
            mode="lines+markers",
            line=dict(color=NAVY),
            name=t("Model", "Mô hình"),
        )
        lo, hi = (
            1 - cal[["predicted", "observed"]].max().max(),
            1 - cal[["predicted", "observed"]].min().min(),
        )
        fig.add_scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color=GREY),
            name=t("Perfect", "Hoàn hảo"),
        )
        fig.update_layout(
            title=t("Calibration (out-of-time)", "Hiệu chỉnh xác suất (out-of-time)"),
            xaxis_title=t("Predicted roll probability", "Xác suất roll dự đoán"),
            yaxis_title=t("Observed roll rate", "Roll rate thực tế"),
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
            title=t(
                "Roll-rate model drivers (mean |SHAP|)",
                "Yếu tố tác động của mô hình roll rate (mean |SHAP|)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    roll_oot = 1 - col["cure_rate_oot"]
    psi = col["psi_p_cure_train_vs_oot"]
    st.caption(
        t(
            f"{col['events_total']:,} delinquency events · trained on {col['events_train']:,}, "
            f"validated on the latest {col['events_oot']:,} · roll rate {roll_oot:.1%} "
            f"(validation) · PSI train vs. validation {psi:.3f}",
            f"{col['events_total']:,} sự kiện quá hạn · huấn luyện trên {col['events_train']:,}, "
            f"kiểm định trên {col['events_oot']:,} sự kiện mới nhất · roll rate {roll_oot:.1%} "
            f"(tập kiểm định) · PSI huấn luyện so với kiểm định {psi:.3f}",
        )
    )

# --------------------------------------------------------------------------- experiment
with tab_ab:
    st.markdown(
        t(
            "The analysis above shows the model routes more at-risk money to collectors. It does "
            "**not** prove that calling changes the outcome. That needs a randomised test: split "
            "the prioritised queue, call one half with the new strategy, keep the current process "
            "for the other, compare the **roll rate to DPD 30+**.",
            "Phân tích ở trên cho thấy mô hình đưa nhiều tiền có rủi ro hơn đến tay nhân viên thu "
            "hồi. Nó **không** chứng minh cuộc gọi làm thay đổi kết quả. Điều đó cần một thử "
            "nghiệm ngẫu nhiên: chia đôi hàng đợi đã ưu tiên, một nửa gọi theo chiến lược mới, "
            "nửa còn lại giữ quy trình hiện tại, rồi so sánh **roll rate sang DPD 30+**.",
        )
    )
    design = load("experiment_design.csv")
    base_default = float(design["baseline_roll_rate"].iloc[0])
    a, b, c = st.columns(3)
    baseline = a.number_input(
        t("Baseline roll rate to DPD 30+", "Roll rate sang DPD 30+ hiện tại"),
        0.001,
        0.9,
        base_default,
        0.005,
        format="%.3f",
    )
    cut = b.number_input(
        t(
            "Smallest relative reduction worth detecting",
            "Mức giảm tương đối nhỏ nhất cần phát hiện",
        ),
        0.01,
        0.9,
        0.20,
        0.01,
    )
    per_day = c.number_input(
        t("New DPD 5 entries per day", "Số khoản mới vào DPD 5 mỗi ngày"), 100, 100_000, 2000, 100
    )
    n = sample_size_per_arm(baseline, baseline * (1 - cut))
    a2, b2, c2 = st.columns(3)
    a2.metric(
        t("Accounts per arm", "Số khoản mỗi nhóm"),
        f"{n:,}",
        f"{baseline:.2%} → {baseline * (1 - cut):.2%}",
        delta_color="off",
    )
    b2.metric(t("Total accounts", "Tổng số khoản"), f"{2 * n:,}")
    c2.metric(t("Days to run", "Số ngày chạy"), f"{math.ceil(2 * n / per_day)}")
    st.dataframe(design, hide_index=True, use_container_width=True)
    st.caption(
        t(
            "Two-sided z-test on two proportions, alpha 0.05, power 0.80. Baseline defaults to the "
            "roll rate of the accounts the model would put in front of collectors (top 30% by "
            "expected rolling amount). Daily volume is an assumption: replace it with your "
            "portfolio's.",
            "Kiểm định z hai phía cho hai tỷ lệ, alpha 0,05, power 0,80. Giá trị nền mặc định là "
            "roll rate của các khoản mô hình sẽ đưa cho nhân viên thu hồi (30% đầu theo số tiền "
            "dự kiến roll). Khối lượng mỗi ngày là giả định: thay bằng số liệu danh mục thực tế.",
        )
    )

# --------------------------------------------------------------------------- about
with tab_about:
    doc = "methodology.vi.md" if VI else "methodology.md"
    st.markdown((ROOT / "docs" / doc).read_text(encoding="utf-8"))

"""Write figures, the results summary and the small aggregate files the dashboard reads."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import Paths  # noqa: E402

NAVY, ORANGE, GREY, GREEN = "#1F3864", "#E07B39", "#9AA5B1", "#2E8B57"
STRATEGY_COLORS = {
    "random": GREY,
    "largest_shortfall": ORANGE,
    "lowest_cure_prob": "#6C8EBF",
    "expected_uncured_amt": NAVY,
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_figure(sc, ch, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, y, s, color in (
        (
            f"Scorecard (AUC {sc.metrics['test']['auc']:.3f})",
            sc.test["target_default"],
            -sc.test["score"],
            NAVY,
        ),
        (
            f"LightGBM (AUC {ch.metrics['test']['auc']:.3f})",
            sc.test["target_default"],
            ch.test_pred,
            ORANGE,
        ),
    ):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=label, color=color, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Default model: test set ROC",
    )
    ax.legend(loc="lower right", frameon=False)
    _save(fig, path)


def grade_figure(grade_table: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(grade_table["grade"], grade_table["default_rate"] * 100, color=NAVY)
    for x, v in zip(grade_table["grade"], grade_table["default_rate"] * 100, strict=True):
        ax.text(x, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set(
        xlabel="Risk grade (A = lowest risk)",
        ylabel="Default rate (%)",
        title="Default rate by scorecard grade (test set)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def capture_figure(curve: pd.DataFrame, labels: dict, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, part in curve.groupby("strategy"):
        ax.plot(
            part["capacity"] * 100,
            part["uncured_amt_captured"] * 100,
            marker="o",
            ms=3,
            lw=2.5 if name == "expected_uncured_amt" else 1.5,
            color=STRATEGY_COLORS[name],
            label=labels[name],
        )
    ax.axvline(30, ls=":", color=GREY)
    ax.set(
        xlabel="Share of each day's queue that collectors can work (%)",
        ylabel="Share of rolling overdue amount reached (%)",
        title="Who to call first: out-of-time validation",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def shap_figure(importance: pd.DataFrame, title: str, path, top: int = 12):
    top_df = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color=NAVY)
    ax.set(xlabel="Mean |SHAP value|", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def calibration_figure(cal: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(1 - cal["predicted"], 1 - cal["observed"], marker="o", color=NAVY, lw=2)
    lo, hi = (
        1 - max(cal["predicted"].max(), cal["observed"].max()),
        1 - min(cal["predicted"].min(), cal["observed"].min()),
    )
    ax.plot([lo, hi], [lo, hi], ls="--", color=GREY)
    ax.set(
        xlabel="Predicted roll probability",
        ylabel="Observed roll rate",
        title="Roll-rate model calibration (out-of-time)",
    )
    _save(fig, path)


def write_all(
    paths: Paths, sc, ch, col, exp_table: pd.DataFrame, labels: dict, events_per_day: int
) -> dict:
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.app_data.mkdir(parents=True, exist_ok=True)

    roc_figure(sc, ch, paths.figures / "pd_roc.png")
    grade_figure(sc.grade_table, paths.figures / "pd_default_rate_by_grade.png")
    shap_figure(
        ch.shap_importance,
        "Default model: top drivers (LightGBM SHAP)",
        paths.figures / "pd_shap.png",
    )
    capture_figure(col.capture_curve, labels, paths.figures / "collections_capture_curve.png")
    shap_figure(
        col.shap_importance,
        "Roll-rate model: top drivers (SHAP)",
        paths.figures / "collections_shap.png",
    )
    calibration_figure(col.calibration, paths.figures / "collections_calibration.png")

    # aggregates only: no row-level competition data leaves the machine
    sc.grade_table.to_csv(paths.app_data / "pd_grades.csv", index=False)
    sc.cutoff_table.to_csv(paths.app_data / "pd_cutoffs.csv", index=False)
    sc.iv_table.to_csv(paths.app_data / "pd_iv.csv", index=False)
    sc.points_table.to_csv(paths.app_data / "pd_points.csv", index=False)
    ch.shap_importance.to_csv(paths.app_data / "pd_shap.csv", index=False)
    col.capture_curve.to_csv(paths.app_data / "collections_capture.csv", index=False)
    col.calibration.to_csv(paths.app_data / "collections_calibration.csv", index=False)
    col.shap_importance.to_csv(paths.app_data / "collections_shap.csv", index=False)
    col.cure_rate_by_month.to_csv(paths.app_data / "collections_cure_by_month.csv", index=False)
    exp_table.to_csv(paths.app_data / "experiment_design.csv", index=False)

    results = {
        "scorecard": sc.metrics,
        "scorecard_grade_edges": sc.grade_edges,
        "challenger": ch.metrics,
        "collections": col.metrics,
        "experiment": {
            "assumed_events_per_day": events_per_day,
            "design": exp_table.to_dict(orient="records"),
        },
    }
    paths.results_json.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return results

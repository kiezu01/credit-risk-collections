"""Gradient-boosting challenger to the scorecard, explained with SHAP.

Answers the usual model-risk question: how much predictive power does the
transparent scorecard give up versus a black-box model on the same data?
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from . import metrics
from .config import SEED


@dataclass
class ChallengerResult:
    model: lgb.LGBMClassifier
    metrics: dict
    shap_importance: pd.DataFrame
    shap_values: np.ndarray
    shap_sample: pd.DataFrame
    test_pred: np.ndarray


def prepare(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("string", "str"):
            X[c] = X[c].astype("category")
        elif X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X


def train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
    shap_rows: int = 5000,
    params: dict | None = None,
) -> ChallengerResult:
    X, y = prepare(train_df, features), train_df[target].astype(int)
    X_test, y_test = prepare(test_df, features), test_df[target].astype(int)
    # align category levels between train and test
    for c in X.select_dtypes("category"):
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
        **(params or {}),
    )
    model.fit(
        X_tr,
        y_tr,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    pred = model.predict_proba(X_test)[:, 1]
    m = {"test": metrics.summary(y_test, pred), "best_iteration": int(model.best_iteration_)}

    sample = X_test.sample(min(shap_rows, len(X_test)), random_state=SEED)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    importance = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return ChallengerResult(model, m, importance, values, sample, pred)

"""Generate a small synthetic dataset with the same schema as Home Credit.

Used by the test suite and CI so the whole pipeline can run without the real
competition data (which may not be redistributed). Numbers produced from this
sample are meaningless and must never be reported as results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "CODE_GENDER": ["M", "F"],
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant"],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
    ],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Rented apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers"],
}


def _applications(rng: np.random.Generator, ids: np.ndarray, risk: np.ndarray, train: bool):
    n = len(ids)
    income = rng.lognormal(11.9, 0.5, n).round(-2)
    credit = (income * rng.uniform(1.5, 6, n)).round(-2)
    annuity = (credit / rng.choice([12, 24, 36, 48], n)).round(2)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "NAME_CONTRACT_TYPE": rng.choice(CATEGORIES["NAME_CONTRACT_TYPE"], n, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(CATEGORIES["CODE_GENDER"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.35, 0.65]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.7, 0.3]),
            "CNT_CHILDREN": rng.poisson(0.4, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n)).round(-2),
            "NAME_INCOME_TYPE": rng.choice(CATEGORIES["NAME_INCOME_TYPE"], n),
            "NAME_EDUCATION_TYPE": rng.choice(CATEGORIES["NAME_EDUCATION_TYPE"], n),
            "NAME_FAMILY_STATUS": rng.choice(CATEGORIES["NAME_FAMILY_STATUS"], n),
            "NAME_HOUSING_TYPE": rng.choice(CATEGORIES["NAME_HOUSING_TYPE"], n),
            "OCCUPATION_TYPE": rng.choice(CATEGORIES["OCCUPATION_TYPE"] + [None], n),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.1, 0.75, 0.15]),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n),
            "DAYS_BIRTH": -rng.integers(21 * 365, 68 * 365, n),
            "DAYS_EMPLOYED": np.where(rng.random(n) < 0.15, 365243, -rng.integers(30, 20 * 365, n)),
            "DAYS_REGISTRATION": -rng.integers(0, 20 * 365, n).astype(float),
            "DAYS_ID_PUBLISH": -rng.integers(0, 15 * 365, n),
            "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n).astype(float),
            "OWN_CAR_AGE": np.where(rng.random(n) < 0.35, rng.integers(0, 25, n), np.nan),
            "EXT_SOURCE_1": np.where(
                rng.random(n) < 0.5,
                np.clip(0.6 - 0.4 * risk + rng.normal(0, 0.15, n), 0, 1),
                np.nan,
            ),
            "EXT_SOURCE_2": np.clip(0.6 - 0.45 * risk + rng.normal(0, 0.15, n), 0, 1),
            "EXT_SOURCE_3": np.where(
                rng.random(n) < 0.8,
                np.clip(0.55 - 0.45 * risk + rng.normal(0, 0.15, n), 0, 1),
                np.nan,
            ),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.poisson(1.5 + 2 * risk, n).astype(float),
        }
    )
    if train:
        p_default = 1 / (1 + np.exp(-(-3.2 + 3.0 * risk)))
        df.insert(1, "TARGET", (rng.random(n) < p_default).astype(int))
    return df


def _bureau(rng, ids, risk):
    rows = []
    bid = 5_000_000
    for cid, r in zip(ids, risk, strict=True):
        for _ in range(rng.poisson(3)):
            credit = float(rng.lognormal(12, 0.8))
            overdue = credit * 0.1 if rng.random() < 0.05 + 0.3 * r else 0.0
            rows.append(
                {
                    "SK_ID_CURR": cid,
                    "SK_ID_BUREAU": bid,
                    "CREDIT_ACTIVE": rng.choice(
                        ["Active", "Closed", "Bad debt"], p=[0.4, 0.58, 0.02]
                    ),
                    "CREDIT_CURRENCY": "currency 1",
                    "DAYS_CREDIT": -int(rng.integers(1, 2900)),
                    "CREDIT_DAY_OVERDUE": int(rng.integers(0, 60)) if overdue else 0,
                    "DAYS_CREDIT_ENDDATE": float(rng.integers(-2000, 2000)),
                    "AMT_CREDIT_MAX_OVERDUE": overdue or np.nan,
                    "CNT_CREDIT_PROLONG": int(rng.random() < 0.02),
                    "AMT_CREDIT_SUM": credit,
                    "AMT_CREDIT_SUM_DEBT": credit * float(rng.uniform(0, 0.9)),
                    "AMT_CREDIT_SUM_LIMIT": 0.0,
                    "AMT_CREDIT_SUM_OVERDUE": overdue,
                    "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"]),
                    "DAYS_CREDIT_UPDATE": -int(rng.integers(0, 500)),
                    "AMT_ANNUITY": np.nan,
                }
            )
            bid += 1
    return pd.DataFrame(rows)


def _previous_and_instalments(rng, ids, risk):
    prev_rows, inst_rows = [], []
    pid = 1_000_000
    for cid, r in zip(ids, risk, strict=True):
        for _ in range(1 + rng.poisson(1.2)):
            status = rng.choice(
                ["Approved", "Refused", "Canceled"], p=[0.65, 0.2 + 0.1 * r, 0.15 - 0.1 * r]
            )
            applied = float(rng.lognormal(11.5, 0.7))
            term = int(rng.choice([6, 12, 24]))
            decision = -int(rng.integers(60, 2800))
            prev_rows.append(
                {
                    "SK_ID_PREV": pid,
                    "SK_ID_CURR": cid,
                    "NAME_CONTRACT_TYPE": rng.choice(["Consumer loans", "Cash loans"]),
                    "AMT_ANNUITY": applied / term,
                    "AMT_APPLICATION": applied,
                    "AMT_CREDIT": applied * float(rng.uniform(0.9, 1.1)),
                    "NAME_CONTRACT_STATUS": status,
                    "DAYS_DECISION": decision,
                    "CNT_PAYMENT": float(term),
                }
            )
            if status == "Approved":
                due_amt = round(applied / term, 2)
                # behaviour: a latent per-loan "struggle" level drives lateness and cures
                struggle = np.clip(r + rng.normal(0, 0.2), 0, 1)
                for k in range(1, term + 1):
                    due_day = decision + 30 * k
                    if due_day > -1:
                        break
                    if rng.random() < 0.05 + 0.4 * struggle:
                        if rng.random() < 0.85 - 0.6 * struggle:
                            # late but cured within 30 days, sometimes in two parts
                            if rng.random() < 0.3:
                                inst_rows.append(
                                    (pid, cid, 1.0, k, due_day, due_day - 2, due_amt, due_amt / 2)
                                )
                                inst_rows.append(
                                    (
                                        pid,
                                        cid,
                                        1.0,
                                        k,
                                        due_day,
                                        due_day + int(rng.integers(1, 30)),
                                        due_amt,
                                        due_amt / 2,
                                    )
                                )
                            else:
                                inst_rows.append(
                                    (
                                        pid,
                                        cid,
                                        1.0,
                                        k,
                                        due_day,
                                        due_day + int(rng.integers(1, 30)),
                                        due_amt,
                                        due_amt,
                                    )
                                )
                        elif rng.random() < 0.5:
                            inst_rows.append(
                                (
                                    pid,
                                    cid,
                                    1.0,
                                    k,
                                    due_day,
                                    due_day + int(rng.integers(31, 120)),
                                    due_amt,
                                    due_amt,
                                )
                            )
                        else:
                            inst_rows.append((pid, cid, 1.0, k, due_day, np.nan, due_amt, np.nan))
                    else:
                        inst_rows.append(
                            (
                                pid,
                                cid,
                                1.0,
                                k,
                                due_day,
                                due_day - int(rng.integers(0, 10)),
                                due_amt,
                                due_amt,
                            )
                        )
            pid += 1
    inst = pd.DataFrame(
        inst_rows,
        columns=[
            "SK_ID_PREV",
            "SK_ID_CURR",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
    )
    return pd.DataFrame(prev_rows), inst


def generate(out_dir: str | Path, n_train: int = 3000, n_test: int = 500, seed: int = 42) -> Path:
    """Write the five Home Credit CSV files used by the pipeline into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ids = np.arange(100_000, 100_000 + n_train + n_test)
    risk = rng.beta(2, 5, len(ids))
    train_ids, test_ids = ids[:n_train], ids[n_train:]

    _applications(rng, train_ids, risk[:n_train], train=True).to_csv(
        out / "application_train.csv", index=False
    )
    _applications(rng, test_ids, risk[n_train:], train=False).to_csv(
        out / "application_test.csv", index=False
    )
    _bureau(rng, ids, risk).to_csv(out / "bureau.csv", index=False)
    prev, inst = _previous_and_instalments(rng, ids, risk)
    prev.to_csv(out / "previous_application.csv", index=False)
    inst.to_csv(out / "installments_payments.csv", index=False)
    return out


if __name__ == "__main__":
    print(generate("data/sample"))

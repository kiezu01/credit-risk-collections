"""Design the A/B test that would prove the new call strategy works.

The capture analysis shows the model puts more at-risk money in front of
collectors; it cannot show that calling those customers changes their outcome.
That needs a randomised test. This module sizes it on the metric collections
teams manage: the roll rate to DPD 30+.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import norm

RELATIVE_REDUCTIONS = [0.05, 0.10, 0.15, 0.20, 0.30]


def sample_size_per_arm(
    p_control: float, p_treatment: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Events per arm to tell two proportions apart with a two-sided z-test
    (pooled variance under H0)."""
    p_bar = (p_control + p_treatment) / 2
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    numerator = (
        z_a * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))
    ) ** 2
    return math.ceil(numerator / (p_control - p_treatment) ** 2)


def days_needed(events_per_arm: int, events_per_day: float, eligible_share: float = 1.0):
    """Days to enrol both arms when only ``eligible_share`` of daily entries can join."""
    eligible_per_day = events_per_day * eligible_share
    if eligible_per_day <= 0:
        return None
    return math.ceil(2 * events_per_arm / eligible_per_day)


def design_table(
    baseline_roll_rate: float,
    events_per_day: float,
    eligible_share: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> pd.DataFrame:
    """Sample size and duration to detect a relative reduction in the roll rate.

    ``events_per_day`` is the daily number of new DPD 5 entries. The test only
    randomises the accounts that would be prioritised (the top
    ``eligible_share`` of each day's queue), and the baseline roll rate is
    measured on those same accounts, so only that share of the daily volume
    counts towards the sample.
    """
    if not 0 < eligible_share <= 1:
        raise ValueError("eligible_share must be in (0, 1]")
    rows = []
    for cut in RELATIVE_REDUCTIONS:
        target = baseline_roll_rate * (1 - cut)
        n = sample_size_per_arm(baseline_roll_rate, target, alpha, power)
        rows.append(
            {
                "relative_reduction": cut,
                "baseline_roll_rate": round(baseline_roll_rate, 4),
                "target_roll_rate": round(target, 4),
                "events_per_arm": n,
                "eligible_per_day": round(events_per_day * eligible_share),
                "days_needed": days_needed(n, events_per_day, eligible_share),
            }
        )
    return pd.DataFrame(rows)

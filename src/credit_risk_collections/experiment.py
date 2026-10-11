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


def design_table(
    baseline_roll_rate: float, events_per_day: float, alpha: float = 0.05, power: float = 0.8
) -> pd.DataFrame:
    """Sample size and duration to detect a relative reduction in the roll rate."""
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
                "days_needed": math.ceil(2 * n / events_per_day) if events_per_day else None,
            }
        )
    return pd.DataFrame(rows)

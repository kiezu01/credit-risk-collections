import math

import pytest

from credit_risk_collections import experiment


def test_known_sample_size():
    # 50% vs 55% at alpha 0.05, power 0.8 needs about 1,565 per arm
    n = experiment.sample_size_per_arm(0.50, 0.55)
    assert 1500 < n < 1650


def test_bigger_reduction_needs_fewer_events():
    sizes = experiment.design_table(0.06, events_per_day=1000)["events_per_arm"].tolist()
    assert sizes == sorted(sizes, reverse=True)


def test_duration_counts_only_the_eligible_share_of_daily_volume():
    full = experiment.design_table(0.0556, events_per_day=2000)
    prioritised = experiment.design_table(0.0556, events_per_day=2000, eligible_share=0.3)
    assert (full["events_per_arm"] == prioritised["events_per_arm"]).all()
    row = prioritised[prioritised["relative_reduction"] == 0.20].iloc[0]
    assert row["eligible_per_day"] == 600
    # both arms must be filled from 600 eligible accounts a day
    assert row["days_needed"] == math.ceil(2 * row["events_per_arm"] / 600)
    assert (prioritised["days_needed"] >= full["days_needed"] * 3).all()


def test_eligible_share_is_validated():
    with pytest.raises(ValueError):
        experiment.design_table(0.05, 1000, eligible_share=0)


def test_rare_outcome_needs_large_samples():
    # roll rate 5% -> 4.5% is a small absolute change: tens of thousands per arm
    assert experiment.sample_size_per_arm(0.05, 0.045) > 20_000

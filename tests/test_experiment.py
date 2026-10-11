from credit_risk_collections import experiment


def test_known_sample_size():
    # 50% vs 55% at alpha 0.05, power 0.8 needs about 1,565 per arm
    n = experiment.sample_size_per_arm(0.50, 0.55)
    assert 1500 < n < 1650


def test_bigger_reduction_needs_fewer_events():
    sizes = experiment.design_table(0.06, events_per_day=1000)["events_per_arm"].tolist()
    assert sizes == sorted(sizes, reverse=True)


def test_rare_outcome_needs_large_samples():
    # roll rate 5% -> 4.5% is a small absolute change: tens of thousands per arm
    assert experiment.sample_size_per_arm(0.05, 0.045) > 20_000

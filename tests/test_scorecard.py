import numpy as np

from credit_risk_collections.scorecard import GRADES, assign_grades


def test_grades_run_from_f_to_a_as_score_rises():
    edges = [500, 550, 600, 650, 700]
    grades = assign_grades(np.array([400, 520, 575, 620, 680, 750]), edges)
    assert list(grades) == GRADES[::-1]

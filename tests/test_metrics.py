import pandas as pd

from ribs.metrics import conditional_asr, summarize_curve, trapezoid_auc


def test_auc():
    assert trapezoid_auc([0, 1], [1, 0]) == 0.5


def test_conditional_asr_excludes_clean_incorrect():
    frame = pd.DataFrame({"clean_correct": [True, False, True], "successful": [True, True, False]})
    assert conditional_asr(frame) == 0.5


def test_curve_is_nested_per_sample():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "radius": [0.0, 1.0, 0.0, 1.0],
            "clean_correct": [True, True, True, True],
            "successful": [False, True, True, False],
        }
    )
    curve, summary = summarize_curve(frame)
    assert curve.robust_accuracy.tolist() == [0.5, 0.0]
    assert curve.conditional_asr.tolist() == [0.5, 1.0]
    assert summary["robust_auc"] == 0.25


def test_curve_adds_zero_radius_and_uses_fixed_maximum():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "radius": [0.1, 0.3, 0.1, 0.3],
            "clean_correct": [True, True, True, True],
            "successful": [False, True, True, True],
        }
    )
    curve, summary = summarize_curve(frame)
    assert curve.radius.tolist() == [0.0, 0.1, 0.3]
    assert abs(summary["robust_auc"] - 5 / 12) < 1e-12


def test_kaplan_meier_event_at_maximum_is_not_marked_censored():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "radius": [1.0, 1.0, 1.0],
            "clean_correct": [True, True, True],
            "successful": [True, True, False],
        }
    )
    _, summary = summarize_curve(frame)
    assert summary["median_min_success_radius"] == 1.0
    assert not summary["median_min_success_right_censored"]

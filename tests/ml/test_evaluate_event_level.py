"""
Tests for scripts/ml/evaluate_event_level.py.

Place at tests/ml/test_evaluate_event_level.py and run from the project root:
    pytest tests/ml -q
All timestamps are on 2024-01-01 UTC; minute N means 00:NN.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ml"))
import evaluate_event_level as ev  # noqa: E402


def at(minute: int) -> str:
    return f"2024-01-01 00:{minute:02d}:00"


def build_pred(n=60, positives=(), missing=()):
    ts = pd.date_range("2024-01-01 00:00", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "actual": 0,
        "probability": 0.1,
        "predicted": 0,
    })
    for m in positives:
        df.loc[m, ["predicted", "probability"]] = [1, 0.9]
    if missing:
        df = df.drop(index=list(missing))
    return df.reset_index(drop=True)


def ev_row(minute: int, cls: str = "M"):
    return (at(minute), at(minute + 5), cls)


def run(tmp_path, pred_df, events=(), horizon="15m", max_gap=1):
    pred_path = tmp_path / f"m_{horizon}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    goes_path = tmp_path / "goes.csv"
    pd.DataFrame(list(events), columns=["start", "end", "class_type"]).to_csv(goes_path, index=False)
    all_events = ev.load_goes_events(goes_path)
    return ev.evaluate_one(pred_path, all_events, tmp_path, max_gap)


# ---------------------------------------------------------------- windows

def test_event_after_period_within_horizon_is_not_false_alarm(tmp_path):
    # Period is 00:00-00:29. Alert at 00:20. Event onset 00:35 is 15 min later,
    # so it is outside the evaluation set but inside the association window.
    pred = build_pred(n=30, positives=[20])
    s = run(tmp_path, pred, events=[ev_row(35)], horizon="15m")
    assert s["goes_mx_events_in_period"] == 0
    assert s["alert_episodes"] == 1
    assert s["false_alarm_episodes"] == 0
    assert s["episodes_forecasting_post_period_event_only"] == 1


def test_event_exactly_at_horizon_boundary_is_associated(tmp_path):
    # Alert at 00:20, event at 00:35: difference == horizon (15) -> included.
    pred = build_pred(n=30, positives=[20])
    s = run(tmp_path, pred, events=[ev_row(35)], horizon="15m")
    assert s["false_alarm_episodes"] == 0


def test_event_one_minute_beyond_horizon_is_false_alarm(tmp_path):
    # Alert at 00:19, event at 00:35: difference == 16 > 15 -> not forecast.
    pred = build_pred(n=30, positives=[19])
    s = run(tmp_path, pred, events=[ev_row(35)], horizon="15m")
    assert s["false_alarm_episodes"] == 1
    assert s["episodes_forecasting_post_period_event_only"] == 0


def test_alert_at_exact_onset_is_not_a_forecast(tmp_path):
    # Event at 00:10 is inside the period. Alert at 00:10 must not count.
    pred = build_pred(n=60, positives=[10])
    s = run(tmp_path, pred, events=[ev_row(10)], horizon="15m")
    assert s["detected_events"] == 0
    assert s["missed_events"] == 1
    assert s["false_alarm_episodes"] == 1


# ---------------------------------------------------------- episode logic

def test_multiple_episodes_forecasting_one_event(tmp_path):
    # Two separate episodes (00:05 and 00:08, gap 3 > 1) both precede event 00:20.
    pred = build_pred(n=60, positives=[5, 8])
    s = run(tmp_path, pred, events=[ev_row(20)], horizon="15m")
    assert s["detected_events"] == 1
    assert s["alert_episodes"] == 2
    assert s["episodes_associated_with_event"] == 2
    assert s["redundant_associated_episodes"] == 1
    assert s["false_alarm_episodes"] == 0


def test_earliest_qualifying_alert_is_assigned(tmp_path):
    pred = build_pred(n=60, positives=[5, 8])
    s = run(tmp_path, pred, events=[ev_row(20)], horizon="15m")
    matches = pd.read_csv(tmp_path / "m_15m_event_matches.csv")
    assert len(matches) == 1
    assert pd.Timestamp(matches.loc[0, "first_qualifying_alert"]) == pd.Timestamp("2024-01-01 00:05:00", tz="UTC")
    assert matches.loc[0, "lead_time_minutes"] == pytest.approx(15.0)


def test_one_episode_associated_with_multiple_events(tmp_path):
    # Sustained alert 00:10-00:12 forecasts events at 00:20 and 00:25.
    pred = build_pred(n=60, positives=[10, 11, 12])
    s = run(tmp_path, pred, events=[ev_row(20), ev_row(25)], horizon="15m")
    assert s["alert_episodes"] == 1
    assert s["detected_events"] == 2
    assert s["alert_episode_precision"] == pytest.approx(1.0)
    assert s["false_alarm_episodes"] == 0
    assert s["redundant_associated_episodes"] == 0


# ------------------------------------------------------------- coverage

def test_missing_timestamps_are_reported_and_excluded_from_coverage(tmp_path):
    # 60-minute span, minutes 10-14 removed -> 5 missing minutes, 55 observed.
    pred = build_pred(n=60, positives=[30], missing=range(10, 15))
    s = run(tmp_path, pred, events=[], horizon="15m")
    assert s["missing_timestamp_minutes"] == 5
    assert s["expected_minutes"] == 60
    assert s["observed_coverage_days"] == pytest.approx(55 / 1440)
    assert s["calendar_span_days"] == pytest.approx(60 / 1440)
    assert s["false_alarms_per_day"] == pytest.approx(1 / (55 / 1440))


# ----------------------------------------------------------- validation

@pytest.mark.parametrize("col, value, match", [
    ("predicted", 2, "only 0 and 1"),
    ("predicted", 0.5, "only 0 and 1"),
    ("actual", 2, "only 0 and 1"),
    ("probability", 1.5, r"within \[0, 1\]"),
    ("probability", -0.1, r"within \[0, 1\]"),
])
def test_invalid_prediction_values_raise(tmp_path, col, value, match):
    pred = build_pred(n=10, positives=[3])

    # Allow fractional values in the predicted column for validation testing.
    if col == "predicted" and value == 0.5:
        pred[col] = pred[col].astype(float)

    pred.loc[5, col] = value

    path = tmp_path / "bad_15m_predictions.csv"
    pred.to_csv(path, index=False)

    with pytest.raises(ValueError, match=match):
        ev.load_predictions(path)


def test_missing_predicted_label_raises(tmp_path):
    pred = build_pred(n=10)
    pred.loc[4, "predicted"] = np.nan
    path = tmp_path / "bad_15m_predictions.csv"
    pred.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing value"):
        ev.load_predictions(path)


def test_missing_probability_is_allowed(tmp_path):
    pred = build_pred(n=10)
    pred.loc[4, "probability"] = np.nan
    path = tmp_path / "ok_15m_predictions.csv"
    pred.to_csv(path, index=False)
    loaded = ev.load_predictions(path)
    assert len(loaded) == 10


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_max_gap_must_be_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        ev.validate_max_gap(value)


def test_max_gap_accepts_positive_integer():
    assert ev.validate_max_gap(2) == 2


# ------------------------------------------------------------ duplicates

def test_duplicate_goes_rows_are_kept_and_reported(tmp_path):
    events = ev.load_goes_events(_write_goes(tmp_path, [ev_row(20), ev_row(20)]))
    assert len(events) == 2
    dups = ev.find_potential_duplicates(events)
    assert len(dups) == 2


def _write_goes(tmp_path, rows):
    p = tmp_path / "goes_dup.csv"
    pd.DataFrame(list(rows), columns=["start", "end", "class_type"]).to_csv(p, index=False)
    return p

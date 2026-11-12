#!/usr/bin/env python3
"""
Event-level evaluation for ARUN-X solar-flare forecasting.

Expected prediction CSV columns:
    timestamp, actual, probability, predicted

Expected GOES CSV columns:
    start, end, class_type

Evaluation rules:
1. Consecutive positive minute predictions form one alert episode
   (gap controlled by --max-gap-minutes).
2. Evaluation events: GOES M/X onsets in [start_time, end_time]. These form the
   denominator for detection metrics.
3. Association events: GOES M/X onsets in [start_time, end_time + horizon]. These
   decide whether an alert episode forecasts a real event, so a valid forecast of
   an event just after the test period is not counted as a false alarm.
4. A GOES event is detected if a positive prediction timestamp t satisfies
   0 < event_start - t <= horizon. Each evaluation event is counted at most once.
   If several episodes qualify, the earliest qualifying alert is assigned.
5. Coverage: observed_coverage_days counts one minute per prediction row
   (assumes a 1-minute cadence). calendar_span_days is the wall-clock span.
   False alarms per day use observed_coverage_days.
6. GOES rows are never silently deduplicated. Rows sharing onset and class are
   written to goes_potential_duplicates.csv for manual inspection.

Run from the project root:
    python scripts/ml/evaluate_event_level.py

Optional:
    python scripts/ml/evaluate_event_level.py --pred-dir output/model_results \
        --goes-file output/goes_events_parsed.csv \
        --output-dir output/model_results/event_level
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


DEFAULT_PRED_DIR = Path("output/model_results")
DEFAULT_GOES_FILE = Path("output/goes_events_parsed.csv")
DEFAULT_OUTPUT_DIR = Path("output/model_results/event_level")
HORIZONS = {"15m": 15, "30m": 30, "60m": 60}
MINUTES_PER_DAY = 1440.0
MATCH_COLUMNS = [
    "event_id", "event_start", "event_end", "class_type", "episode_id",
    "episode_start", "first_qualifying_alert", "lead_time_minutes",
]


def validate_max_gap(value) -> int:
    """Return value as int if it is a positive integer, otherwise raise."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"max_gap_minutes must be a positive integer, got {value!r}")
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate solar flare predictions at event level.")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--goes-file", type=Path, default=DEFAULT_GOES_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=1,
        help="Maximum gap between positive prediction timestamps in one episode. "
             "1 means strictly consecutive one-minute predictions.",
    )
    args = parser.parse_args()
    args.max_gap_minutes = validate_max_gap(args.max_gap_minutes)
    return args


def load_goes_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GOES catalogue not found: {path}")

    events = pd.read_csv(path)
    required = {"start", "end", "class_type"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"GOES CSV missing columns: {sorted(missing)}")

    events["start"] = pd.to_datetime(events["start"], utc=True, errors="coerce")
    events["end"] = pd.to_datetime(events["end"], utc=True, errors="coerce")
    events["class_type"] = events["class_type"].astype(str).str.upper().str.strip()

    events = events[
        events["class_type"].isin(["M", "X"])
        & events["start"].notna()
        & events["end"].notna()
        & (events["end"] >= events["start"])
    ].copy()

    # Preserve catalogue rows; do not deduplicate events.
    events = events.sort_values(["start", "end", "class_type"]).reset_index(drop=True)
    events.insert(0, "event_id", [f"GOES_MX_{i + 1:04d}" for i in range(len(events))])
    return events


def find_potential_duplicates(events: pd.DataFrame) -> pd.DataFrame:
    """Rows sharing onset and class with another row. Reported, never removed."""
    mask = events.duplicated(subset=["start", "class_type"], keep=False)
    return events.loc[mask].copy()


def _parse_numeric(pred: pd.DataFrame, col: str, path: Path, required: bool) -> pd.Series:
    raw = pred[col]
    values = pd.to_numeric(raw, errors="coerce")
    non_numeric = values.isna() & raw.notna()
    if non_numeric.any():
        raise ValueError(f"{path.name}: column '{col}' has {int(non_numeric.sum())} non-numeric value(s)")
    if required and values.isna().any():
        raise ValueError(f"{path.name}: column '{col}' has {int(values.isna().sum())} missing value(s)")
    return values


def _check_binary(values: pd.Series, col: str, path: Path) -> None:
    v = values.dropna()
    bad = v[~v.isin([0, 1])]
    if not bad.empty:
        raise ValueError(
            f"{path.name}: column '{col}' must contain only 0 and 1; "
            f"found {sorted(set(bad.tolist()))[:5]}"
        )


def _check_probability(values: pd.Series, path: Path) -> None:
    v = values.dropna()
    bad = v[(v < 0) | (v > 1)]
    if not bad.empty:
        raise ValueError(
            f"{path.name}: probability must be within [0, 1]; "
            f"found {len(bad)} out-of-range value(s), e.g. {bad.iloc[0]}"
        )


def load_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    required = {"timestamp", "actual", "probability", "predicted"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
    bad_ts = timestamps.isna()
    if bad_ts.any():
        csv_lines = (bad_ts[bad_ts].index[:5] + 2).tolist()  # +2: header + 1-based
        raise ValueError(
            f"{path.name}: {int(bad_ts.sum())} missing or unparseable timestamp(s); "
            f"CSV lines {csv_lines}"
        )
    pred["timestamp"] = timestamps

    pred["actual"] = _parse_numeric(pred, "actual", path, required=False)
    pred["probability"] = _parse_numeric(pred, "probability", path, required=False)
    pred["predicted"] = _parse_numeric(pred, "predicted", path, required=True)

    _check_binary(pred["predicted"], "predicted", path)
    _check_binary(pred["actual"], "actual", path)
    _check_probability(pred["probability"], path)

    pred["predicted"] = pred["predicted"].astype(int)
    pred = pred.sort_values("timestamp", kind="mergesort")
    n_dup = int(pred["timestamp"].duplicated().sum())
    pred = pred.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    pred.attrs["duplicate_timestamp_rows"] = n_dup
    return pred


def get_model_horizon(path: Path):
    name = path.stem
    match = re.match(r"(.+?)_(15m|30m|60m)_predictions$", name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def build_alert_episodes(pred: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    max_gap_minutes = validate_max_gap(max_gap_minutes)
    columns = [
        "episode_id", "episode_start", "episode_end", "positive_minutes",
        "max_probability", "mean_probability",
    ]
    positive = pred.loc[pred["predicted"].eq(1), ["timestamp", "probability"]].copy()
    if positive.empty:
        return pd.DataFrame(columns=columns)

    positive = positive.sort_values("timestamp").reset_index(drop=True)
    gap = positive["timestamp"].diff().dt.total_seconds().div(60)
    new_episode = gap.isna() | gap.gt(max_gap_minutes)
    positive["episode_num"] = new_episode.cumsum()

    episodes = (
        positive.groupby("episode_num", sort=True)
        .agg(
            episode_start=("timestamp", "min"),
            episode_end=("timestamp", "max"),
            positive_minutes=("timestamp", "size"),
            max_probability=("probability", "max"),
            mean_probability=("probability", "mean"),
        )
        .reset_index(drop=True)
    )
    episodes.insert(0, "episode_id", [f"ALERT_{i + 1:04d}" for i in range(len(episodes))])
    return episodes


def match_events_to_episodes(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    positive_timestamps: pd.Series,
    horizon_minutes: int,
):
    """
    A GOES event is detected if any positive timestamp t satisfies
    0 < event_start - t <= H. If several alerts qualify, the earliest is assigned.
    Each event is counted at most once.
    """
    if events.empty or episodes.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS), set()

    h = pd.Timedelta(minutes=horizon_minutes)
    positive_ts = pd.DatetimeIndex(positive_timestamps.sort_values().unique())
    matches = []

    for event in events.itertuples(index=False):
        # Strictly before event start; event - t <= horizon.
        lo = positive_ts.searchsorted(event.start - h, side="left")
        hi = positive_ts.searchsorted(event.start, side="left")
        qualifying = positive_ts[lo:hi]
        if len(qualifying) == 0:
            continue

        first_alert = qualifying[0]
        candidates = episodes[
            (episodes["episode_start"] <= first_alert)
            & (episodes["episode_end"] >= first_alert)
        ]
        if candidates.empty:
            continue

        episode = candidates.iloc[0]
        matches.append({
            "event_id": event.event_id,
            "event_start": event.start,
            "event_end": event.end,
            "class_type": event.class_type,
            "episode_id": episode["episode_id"],
            "episode_start": episode["episode_start"],
            "first_qualifying_alert": first_alert,
            "lead_time_minutes": (event.start - first_alert).total_seconds() / 60.0,
        })

    match_df = pd.DataFrame(matches, columns=MATCH_COLUMNS)
    matched_episode_ids = set(match_df["episode_id"]) if not match_df.empty else set()
    return match_df, matched_episode_ids


def associate_episodes(
    episodes: pd.DataFrame,
    pred: pd.DataFrame,
    assoc_events: pd.DataFrame,
    horizon_minutes: int,
) -> dict:
    """
    Map each episode_id to the set of association-event IDs it forecasts.
    An episode is associated with an event when some positive timestamp t in the
    episode satisfies 0 < event_start - t <= H.
    """
    if episodes.empty:
        return {}
    result = {ep_id: set() for ep_id in episodes["episode_id"]}
    if assoc_events.empty:
        return result

    h = pd.Timedelta(minutes=horizon_minutes)
    event_starts = pd.DatetimeIndex(assoc_events["start"].sort_values().unique())
    # Keep duplicate catalogue rows: map each onset back to all of its IDs.
    ids_by_start = assoc_events.groupby("start")["event_id"].apply(list).to_dict()
    positive_frame = pred.loc[pred["predicted"].eq(1), ["timestamp"]].sort_values("timestamp")

    for ep in episodes.itertuples(index=False):
        ep_positive = positive_frame.loc[
            positive_frame["timestamp"].between(ep.episode_start, ep.episode_end, inclusive="both"),
            "timestamp",
        ]
        associated = set()
        for alert_time in ep_positive:
            lo = event_starts.searchsorted(alert_time, side="right")
            hi = event_starts.searchsorted(alert_time + h, side="right")
            for event_start in event_starts[lo:hi]:
                associated.update(ids_by_start[event_start])
        result[ep.episode_id] = associated
    return result


def coverage_stats(pred: pd.DataFrame, start_time, end_time) -> dict:
    """
    Coverage assumes one prediction row per observed minute.
    calendar_span_days is the wall-clock span from first to last timestamp.
    observed_coverage_days counts only rows that exist (missing minutes excluded).
    """
    span_minutes = int(round((end_time - start_time).total_seconds() / 60.0)) + 1
    observed_minutes = len(pred)
    return {
        "calendar_span_days": span_minutes / MINUTES_PER_DAY,
        "observed_coverage_days": observed_minutes / MINUTES_PER_DAY,
        "expected_minutes": span_minutes,
        "observed_minutes": observed_minutes,
        "missing_timestamp_minutes": span_minutes - observed_minutes,
    }


def evaluate_one(pred_path: Path, events_all: pd.DataFrame, output_dir: Path, max_gap: int):
    model, horizon = get_model_horizon(pred_path)
    if model is None:
        return None

    max_gap = validate_max_gap(max_gap)
    horizon_minutes = HORIZONS[horizon]
    pred = load_predictions(pred_path)
    if pred.empty:
        print(f"Skipping empty prediction file: {pred_path}")
        return None

    episodes = build_alert_episodes(pred, max_gap)
    positive_ts = pred.loc[pred["predicted"].eq(1), "timestamp"]

    start_time = pred["timestamp"].min()
    end_time = pred["timestamp"].max()
    horizon_end = end_time + pd.Timedelta(minutes=horizon_minutes)

    # Evaluation events: onset inside the prediction period. Denominator for detection.
    eval_events = events_all[
        events_all["start"].between(start_time, end_time, inclusive="both")
    ].copy()
    # Association events: onset inside the period or within H after it.
    # Used only to decide whether an alert episode forecasts a real event.
    assoc_events = events_all[
        events_all["start"].between(start_time, horizon_end, inclusive="both")
    ].copy()

    matches, matched_episode_ids = match_events_to_episodes(
        eval_events, episodes, positive_ts, horizon_minutes
    )
    episode_event_ids = associate_episodes(episodes, pred, assoc_events, horizon_minutes)

    eval_ids = set(eval_events["event_id"])
    associated_episode_ids = {eid for eid, ids in episode_event_ids.items() if ids}
    eval_associated_ids = {eid for eid, ids in episode_event_ids.items() if ids & eval_ids}
    post_period_only_ids = associated_episode_ids - eval_associated_ids

    false_alarm_episodes = len(episodes) - len(associated_episode_ids)
    redundant_associated_episodes = max(0, len(eval_associated_ids) - len(matched_episode_ids))

    total_events = len(eval_events)
    detected_events = len(matches)
    missed_events = total_events - detected_events
    event_recall = detected_events / total_events if total_events else np.nan
    # Fraction of alert episodes whose forecast window contains an M/X event
    # (in-period or just after the period).
    event_precision = len(associated_episode_ids) / len(episodes) if len(episodes) else np.nan

    cov = coverage_stats(pred, start_time, end_time)
    false_alarms_per_day = (
        false_alarm_episodes / cov["observed_coverage_days"]
        if cov["observed_coverage_days"] > 0 else np.nan
    )

    actual = pred["actual"]
    valid_actual = actual.isin([0, 1])
    tp = int(((pred["predicted"] == 1) & (actual == 1) & valid_actual).sum())
    fp = int(((pred["predicted"] == 1) & (actual == 0) & valid_actual).sum())
    fn = int(((pred["predicted"] == 0) & (actual == 1) & valid_actual).sum())

    summary = {
        "model": model,
        "horizon": horizon,
        "prediction_rows": len(pred),
        "test_start": start_time,
        "test_end": end_time,
        "calendar_span_days": cov["calendar_span_days"],
        "observed_coverage_days": cov["observed_coverage_days"],
        "expected_minutes": cov["expected_minutes"],
        "missing_timestamp_minutes": cov["missing_timestamp_minutes"],
        "duplicate_timestamp_rows": pred.attrs.get("duplicate_timestamp_rows", 0),
        "goes_mx_events_in_period": total_events,
        "goes_mx_events_association_window": len(assoc_events),
        "detected_events": detected_events,
        "missed_events": missed_events,
        "event_detection_rate": event_recall,
        "alert_episode_precision": event_precision,
        "alert_episodes": len(episodes),
        "episodes_associated_with_event": len(associated_episode_ids),
        "episodes_forecasting_post_period_event_only": len(post_period_only_ids),
        "false_alarm_episodes": false_alarm_episodes,
        "redundant_associated_episodes": redundant_associated_episodes,
        "false_alarms_per_day": false_alarms_per_day,
        "mean_lead_time_minutes": matches["lead_time_minutes"].mean() if not matches.empty else np.nan,
        "median_lead_time_minutes": matches["lead_time_minutes"].median() if not matches.empty else np.nan,
        "min_lead_time_minutes": matches["lead_time_minutes"].min() if not matches.empty else np.nan,
        "max_lead_time_minutes": matches["lead_time_minutes"].max() if not matches.empty else np.nan,
        "row_level_tp": tp,
        "row_level_fp": fp,
        "row_level_fn": fn,
    }

    episode_details = episodes.copy()
    episode_details["associated_event_count"] = episode_details["episode_id"].map(
        lambda x: len(episode_event_ids.get(x, set()))
    )
    episode_details["eval_event_count"] = episode_details["episode_id"].map(
        lambda x: len(episode_event_ids.get(x, set()) & eval_ids)
    )
    episode_details["assigned_event_count"] = episode_details["episode_id"].isin(
        matched_episode_ids
    ).astype(int)
    episode_details["episode_status"] = np.select(
        [
            episode_details["associated_event_count"].eq(0),
            episode_details["assigned_event_count"].eq(1),
            episode_details["eval_event_count"].gt(0),
        ],
        ["false_alarm", "matched_alert", "redundant_alert"],
        default="post_period_alert",
    )

    prefix = f"{model}_{horizon}"
    matches.to_csv(output_dir / f"{prefix}_event_matches.csv", index=False)
    episode_details.to_csv(output_dir / f"{prefix}_alert_episodes.csv", index=False)

    print(
        f"{model:28s} {horizon:>3s} | "
        f"events {detected_events}/{total_events} "
        f"({event_recall:.1%}) | "
        f"false alarms/day {false_alarms_per_day:.2f} | "
        f"episodes {len(episodes)} | "
        f"missing min {cov['missing_timestamp_minutes']}"
    )
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")

    events = load_goes_events(args.goes_file)

    duplicates = find_potential_duplicates(events)
    if not duplicates.empty:
        dup_path = args.output_dir / "goes_potential_duplicates.csv"
        duplicates.to_csv(dup_path, index=False)
        print(
            f"WARNING: {len(duplicates)} GOES rows share onset and class with another row. "
            f"Kept for manual inspection: {dup_path}"
        )

    prediction_files = sorted(args.pred_dir.rglob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No *_predictions.csv files found under {args.pred_dir}")

    print("=" * 78)
    print("ARUN-X EVENT-LEVEL EVALUATION")
    print("=" * 78)
    print(f"GOES M/X catalogue events: {len(events)}")
    print(f"Prediction files found: {len(prediction_files)}")
    print(f"Alert episode gap: {args.max_gap_minutes} minute(s)")
    print()

    summaries = []
    for path in prediction_files:
        try:
            result = evaluate_one(path, events, args.output_dir, args.max_gap_minutes)
            if result is not None:
                summaries.append(result)
        except Exception as exc:
            print(f"ERROR processing {path.name}: {exc}", file=sys.stderr)

    if not summaries:
        raise RuntimeError("No prediction files were evaluated successfully.")

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        ["horizon", "event_detection_rate", "false_alarms_per_day"],
        ascending=[True, False, True],
        na_position="last",
    )
    summary_path = args.output_dir / "event_level_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("EVENT-LEVEL SUMMARY")
    print("=" * 78)
    display_cols = [
        "horizon", "model", "event_detection_rate", "alert_episode_precision",
        "false_alarms_per_day", "mean_lead_time_minutes",
        "detected_events", "missed_events", "alert_episodes",
        "missing_timestamp_minutes",
    ]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved summary: {summary_path}")
    print(f"Detailed outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
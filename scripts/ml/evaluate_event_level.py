#!/usr/bin/env python3
"""
Corrected event-level evaluation for ARUN-X solar-flare forecasting.

Expected prediction CSV columns:
    timestamp, actual, probability, predicted

Expected GOES CSV columns:
    start, end, class_type

The evaluator:
1. Groups consecutive positive minute predictions into alert episodes.
2. Matches an M/X event when at least one positive prediction timestamp t
   satisfies 0 < event_start - t <= forecast_horizon.
3. Counts each GOES event at most once per model/horizon.
4. Reports episode-level false alarms and event-level detection metrics.

Run from the project root:
    python scripts/ml/evaluate_event_level.py

Optional:
    python scripts/ml/evaluate_event_level.py ^
        --pred-dir output/model_results ^
        --goes-file output/goes_events_parsed.csv ^
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
    return parser.parse_args()


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

    # Preserve catalogue rows; do not deduplicate events by start time.
    events = events.sort_values(["start", "end", "class_type"]).reset_index(drop=True)
    events.insert(0, "event_id", [f"GOES_MX_{i + 1:04d}" for i in range(len(events))])
    return events


def load_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    required = {"timestamp", "actual", "probability", "predicted"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
    pred["actual"] = pd.to_numeric(pred["actual"], errors="coerce")
    pred["probability"] = pd.to_numeric(pred["probability"], errors="coerce")
    pred["predicted"] = pd.to_numeric(pred["predicted"], errors="coerce")
    pred = pred.dropna(subset=["timestamp", "predicted"]).copy()
    pred["predicted"] = pred["predicted"].astype(int)
    pred = pred.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return pred


def get_model_horizon(path: Path):
    name = path.stem
    match = re.match(r"(.+?)_(15m|30m|60m)_predictions$", name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def build_alert_episodes(pred: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    positive = pred.loc[pred["predicted"].eq(1), ["timestamp", "probability"]].copy()
    if positive.empty:
        return pd.DataFrame(columns=[
            "episode_id", "episode_start", "episode_end", "positive_minutes",
            "max_probability", "mean_probability"
        ])

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
    A GOES event is forecast if any positive timestamp t occurs strictly before
    the event and no more than H minutes before it.

    If several episodes forecast the same event, assign it to the earliest
    qualifying alert timestamp. Each GOES event is counted once.
    """
    if events.empty or episodes.empty:
        return (
            pd.DataFrame(columns=[
                "event_id", "event_start", "event_end", "class_type",
                "episode_id", "episode_start", "first_qualifying_alert",
                "lead_time_minutes"
            ]),
            set(),
        )

    h = pd.Timedelta(minutes=horizon_minutes)
    positive_ts = pd.DatetimeIndex(positive_timestamps.sort_values().unique())
    matches = []

    for event in events.itertuples(index=False):
        # Strictly before event start; event - t <= horizon.
        left = event.start - h
        right = event.start
        lo = positive_ts.searchsorted(left, side="left")
        hi = positive_ts.searchsorted(right, side="left")
        qualifying = positive_ts[lo:hi]

        if len(qualifying) == 0:
            continue

        # Find the episode containing the earliest qualifying positive timestamp.
        # The timestamp-to-episode lookup is built from the episode boundaries
        # and exact positive prediction timestamps below.
        first_alert = qualifying[0]
        candidates = episodes[
            (episodes["episode_start"] <= first_alert)
            & (episodes["episode_end"] >= first_alert)
        ]
        if candidates.empty:
            continue

        episode = candidates.iloc[0]
        lead = (event.start - first_alert).total_seconds() / 60.0
        matches.append({
            "event_id": event.event_id,
            "event_start": event.start,
            "event_end": event.end,
            "class_type": event.class_type,
            "episode_id": episode["episode_id"],
            "episode_start": episode["episode_start"],
            "first_qualifying_alert": first_alert,
            "lead_time_minutes": lead,
        })

    match_df = pd.DataFrame(matches)
    matched_episode_ids = set(match_df["episode_id"]) if not match_df.empty else set()
    return match_df, matched_episode_ids


def evaluate_one(pred_path: Path, events_all: pd.DataFrame, output_dir: Path, max_gap: int):
    model, horizon = get_model_horizon(pred_path)
    if model is None:
        return None

    horizon_minutes = HORIZONS[horizon]
    pred = load_predictions(pred_path)
    if pred.empty:
        print(f"Skipping empty prediction file: {pred_path}")
        return None

    episodes = build_alert_episodes(pred, max_gap)
    positive_ts = pred.loc[pred["predicted"].eq(1), "timestamp"]

    # Only score events whose onset falls inside the prediction period.
    # This keeps the denominator aligned to the supplied test predictions.
    start_time = pred["timestamp"].min()
    end_time = pred["timestamp"].max()
    events = events_all[
        events_all["start"].between(start_time, end_time, inclusive="both")
    ].copy()

    matches, matched_episode_ids = match_events_to_episodes(
        events, episodes, positive_ts, horizon_minutes
    )

    # Episode-level association: an episode is associated with an event only
    # when at least one actual positive prediction timestamp t in that episode
    # satisfies 0 < event_start - t <= horizon. This checks each alert minute
    # individually; episode boundary overlap alone is not sufficient.
    episode_event_ids = {row["episode_id"]: set() for _, row in episodes.iterrows()}
    if not episodes.empty and not events.empty:
        event_starts = pd.DatetimeIndex(events["start"].sort_values().unique())
        # Preserve duplicate catalogue rows (same start can represent separate
        # catalogue events) by mapping each qualifying start back to all IDs.
        event_ids_by_start = (
            events.groupby("start")["event_id"].apply(list).to_dict()
        )
        positive_frame = pred.loc[
            pred["predicted"].eq(1), ["timestamp"]
        ].sort_values("timestamp")

        # Assign each positive timestamp to its episode using the episode
        # boundaries. Episodes are disjoint and consist only of positive rows.
        for ep in episodes.itertuples(index=False):
            ep_positive = positive_frame.loc[
                positive_frame["timestamp"].between(
                    ep.episode_start, ep.episode_end, inclusive="both"
                ),
                "timestamp",
            ]
            if ep_positive.empty:
                continue

            associated_ids = set()
            for alert_time in ep_positive:
                # Strictly later than the alert, and no later than alert + H.
                lo = event_starts.searchsorted(alert_time, side="right")
                hi = event_starts.searchsorted(
                    alert_time + pd.Timedelta(minutes=horizon_minutes),
                    side="right",
                )
                for event_start in event_starts[lo:hi]:
                    associated_ids.update(event_ids_by_start[event_start])
            episode_event_ids[ep.episode_id] = associated_ids

    associated_episode_ids = {
        episode_id for episode_id, event_ids in episode_event_ids.items() if event_ids
    }
    false_alarm_episodes = len(episodes) - len(associated_episode_ids)
    redundant_associated_episodes = max(0, len(associated_episode_ids) - len(matched_episode_ids))

    total_events = len(events)
    detected_events = len(matches)
    missed_events = total_events - detected_events
    event_recall = detected_events / total_events if total_events else np.nan
    # Alert-episode precision: fraction of alert episodes whose forecast
    # window contains at least one M/X event. This cannot exceed 1 even when
    # one sustained episode forecasts multiple distinct events.
    event_precision = (
        len(associated_episode_ids) / len(episodes) if len(episodes) else np.nan
    )

    # Evaluate over the actual prediction coverage duration.
    coverage_days = (
        (end_time - start_time).total_seconds() / 86400.0 + 1.0 / 1440.0
    )
    false_alarms_per_day = false_alarm_episodes / coverage_days if coverage_days > 0 else np.nan

    # Positive-row classification metrics for reference.
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
        "coverage_days": coverage_days,
        "goes_mx_events_in_period": total_events,
        "detected_events": detected_events,
        "missed_events": missed_events,
        "event_detection_rate": event_recall,
        "alert_episode_precision": event_precision,
        "alert_episodes": len(episodes),
        "episodes_associated_with_event": len(associated_episode_ids),
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

    # Detailed episode table.
    episode_details = episodes.copy()
    episode_details["associated_event_count"] = episode_details["episode_id"].map(
        lambda x: len(episode_event_ids.get(x, set()))
    )
    episode_details["assigned_event_count"] = episode_details["episode_id"].isin(
        matched_episode_ids
    ).astype(int)
    episode_details["episode_status"] = np.where(
        episode_details["associated_event_count"].eq(0),
        "false_alarm",
        np.where(
            episode_details["assigned_event_count"].eq(1),
            "matched_alert",
            "redundant_alert",
        ),
    )

    prefix = f"{model}_{horizon}"
    matches.to_csv(output_dir / f"{prefix}_event_matches.csv", index=False)
    episode_details.to_csv(output_dir / f"{prefix}_alert_episodes.csv", index=False)

    print(
        f"{model:28s} {horizon:>3s} | "
        f"events {detected_events}/{total_events} "
        f"({event_recall:.1%}) | "
        f"false alarms/day {false_alarms_per_day:.2f} | "
        f"episodes {len(episodes)}"
    )
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")

    events = load_goes_events(args.goes_file)
    prediction_files = sorted(args.pred_dir.rglob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(
            f"No *_predictions.csv files found under {args.pred_dir}"
        )

    print("=" * 78)
    print("ARUN-X EVENT-LEVEL EVALUATION (CORRECTED EPISODE ASSOCIATION)")
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
    ]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved summary: {summary_path}")
    print(f"Detailed outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

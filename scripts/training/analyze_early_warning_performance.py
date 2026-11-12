
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "output" / "ml_ready"
MODEL_DIR = ROOT / "output" / "models"

MODEL_PATH = MODEL_DIR / "xgboost_flare_detection.pkl"
IMPUTER_PATH = MODEL_DIR / "flare_detection_imputer.pkl"

GOES_PATH = ROOT / "output" / "goes_events_parsed.csv"

OUTPUT_DIR = MODEL_DIR / "early_warning_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "June": DATA_DIR / "classification_validation.parquet",
    "July": DATA_DIR / "classification_test.parquet"
}

THRESHOLDS = [0.05, 0.10, 0.20]

WARNING_WINDOW_MINUTES = 60
PEAK_WARNING_MINUTES = [5, 10, 15]

MINUTE_TOLERANCE_SECONDS = 90


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model and imputer...")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print("Model and imputer loaded.")


# ============================================================
# LOAD GOES EVENTS
# ============================================================

def load_goes_events():

    print("\nLoading GOES catalogue...")

    df = pd.read_csv(GOES_PATH)

    required = ["start", "peak", "end", "class"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing GOES columns: {missing}")

    events = pd.DataFrame({
        "start_time": pd.to_datetime(
            df["start"], utc=True, errors="coerce"
        ),
        "peak_time": pd.to_datetime(
            df["peak"], utc=True, errors="coerce"
        ),
        "end_time": pd.to_datetime(
            df["end"], utc=True, errors="coerce"
        ),
        "goes_class": df["class"].astype(str).str.strip()
    })

    events = events.dropna(
        subset=["start_time", "peak_time", "end_time"]
    ).copy()

    events["flare_type"] = (
        events["goes_class"].str.upper().str[0]
    )

    events = events[
        events["flare_type"].isin(["M", "X"])
    ]

    events = events[
        (events["start_time"] <= events["peak_time"]) &
        (events["peak_time"] <= events["end_time"])
    ]

    events = events.sort_values(
        "start_time"
    ).reset_index(drop=True)

    events.insert(
        0,
        "event_id",
        [f"GOES_{i + 1:04d}" for i in range(len(events))]
    )

    print(f"Total valid M/X events: {len(events)}")
    print(events["flare_type"].value_counts())

    return events


# ============================================================
# GENERATE ROW-LEVEL PREDICTIONS
# ============================================================

def generate_predictions(path, dataset_name):

    print(f"\nGenerating predictions for {dataset_name}...")

    df = pd.read_parquet(path)

    timestamps = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    X = df.drop(
        columns=["timestamp", "target"],
        errors="ignore"
    )

    probabilities = model.predict_proba(
        imputer.transform(X)
    )[:, 1]

    result = pd.DataFrame({
        "timestamp": timestamps,
        "true_class": df["target"].to_numpy(),
        "probability": probabilities
    })

    result = result.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Observations: {len(result)}")

    return result


# ============================================================
# SELECT EVENTS OVERLAPPING THE DATASET
# ============================================================

def select_events(events, predictions):

    data_start = predictions["timestamp"].min()
    data_end = predictions["timestamp"].max()

    selected = events[
        (events["end_time"] >= data_start) &
        (events["start_time"] <= data_end)
    ].copy()

    return selected.reset_index(drop=True)


# ============================================================
# CREATE CONTIGUOUS ALARM EPISODES
# ============================================================

def extract_alarm_episodes(predictions, threshold):

    df = predictions.copy()

    df["alarm"] = (
        df["probability"] >= threshold
    )

    positive = df[df["alarm"]].copy()

    if positive.empty:
        return pd.DataFrame(
            columns=["alarm_start", "alarm_end", "duration_minutes"]
        )

    # A new episode begins if:
    # 1. The previous row was not positive, or
    # 2. There is a gap greater than one minute.
    time_gap = positive["timestamp"].diff()

    new_episode = (
        time_gap.isna() |
        (time_gap > pd.Timedelta(minutes=1))
    )

    positive["episode_id"] = new_episode.cumsum()

    episodes = (
        positive.groupby("episode_id")
        .agg(
            alarm_start=("timestamp", "min"),
            alarm_end=("timestamp", "max"),
            peak_probability=("probability", "max"),
            mean_probability=("probability", "mean"),
            observations=("timestamp", "count")
        )
        .reset_index(drop=True)
    )

    episodes["duration_minutes"] = (
        episodes["alarm_end"] - episodes["alarm_start"]
    ).dt.total_seconds() / 60 + 1

    return episodes


# ============================================================
# CLASSIFY ALARM EPISODES
# ============================================================

def classify_alarm_episodes(episodes, events):

    episodes = episodes.copy()

    classifications = []
    associated_events = []

    for _, alarm in episodes.iterrows():

        alarm_time = alarm["alarm_start"]

        # Alarm starts during an active GOES event.
        active_matches = events[
            (events["start_time"] <= alarm_time) &
            (events["end_time"] >= alarm_time)
        ]

        if not active_matches.empty:

            classifications.append("during_event")
            associated_events.append(
                active_matches.iloc[0]["event_id"]
            )

            continue

        # Alarm starts within 60 minutes before an event.
        warning_matches = events[
            (events["start_time"] > alarm_time) &
            (
                events["start_time"] <=
                alarm_time + pd.Timedelta(
                    minutes=WARNING_WINDOW_MINUTES
                )
            )
        ]

        if not warning_matches.empty:

            next_event = warning_matches.sort_values(
                "start_time"
            ).iloc[0]

            classifications.append("early_warning")
            associated_events.append(next_event["event_id"])

        else:

            classifications.append("false_alarm")
            associated_events.append(None)

    episodes["alarm_type"] = classifications
    episodes["associated_event"] = associated_events

    return episodes


# ============================================================
# EVENT-LEVEL ANALYSIS
# ============================================================

def evaluate_events(events, episodes, dataset_name, threshold):

    results = []

    for _, event in events.iterrows():

        start = event["start_time"]
        peak = event["peak_time"]
        end = event["end_time"]

        warning_start = (
            start - pd.Timedelta(
                minutes=WARNING_WINDOW_MINUTES
            )
        )

        # First alarm in the 60-minute pre-onset window.
        pre_onset = episodes[
            (episodes["alarm_start"] >= warning_start) &
            (episodes["alarm_start"] < start)
        ]

        # First alarm during the GOES event.
        during_event = episodes[
            (episodes["alarm_start"] >= start) &
            (episodes["alarm_start"] <= end)
        ]

        pre_onset = pre_onset.sort_values("alarm_start")
        during_event = during_event.sort_values("alarm_start")

        has_warning = not pre_onset.empty
        detected_during = not during_event.empty

        first_warning_time = (
            pre_onset.iloc[0]["alarm_start"]
            if has_warning else pd.NaT
        )

        first_detection_time = (
            during_event.iloc[0]["alarm_start"]
            if detected_during else pd.NaT
        )

        # Earliest alarm associated with this event.
        candidate_times = []

        if has_warning:
            candidate_times.append(first_warning_time)

        if detected_during:
            candidate_times.append(first_detection_time)

        first_alarm_time = (
            min(candidate_times)
            if candidate_times else pd.NaT
        )

        lead_time = np.nan

        if pd.notna(first_alarm_time):
            lead_time = (
                start - first_alarm_time
            ).total_seconds() / 60

        detection_delay = np.nan

        if detected_during:
            detection_delay = (
                first_detection_time - start
            ).total_seconds() / 60

        # Did an alarm occur at least N minutes before peak?
        peak_warning_results = {}

        for minutes in PEAK_WARNING_MINUTES:

            deadline = (
                peak - pd.Timedelta(minutes=minutes)
            )

            qualifying = episodes[
                (episodes["alarm_start"] >= warning_start) &
                (episodes["alarm_start"] <= deadline)
            ]

            peak_warning_results[
                f"before_peak_{minutes}m"
            ] = int(not qualifying.empty)

        row = {
            "dataset": dataset_name,
            "threshold": threshold,
            "event_id": event["event_id"],
            "goes_class": event["goes_class"],
            "flare_type": event["flare_type"],
            "start_time": start,
            "peak_time": peak,
            "end_time": end,
            "duration_minutes": (
                end - start
            ).total_seconds() / 60,

            "pre_onset_warning": int(has_warning),
            "detected_during_event": int(detected_during),

            "first_warning_time": first_warning_time,
            "first_detection_time": first_detection_time,

            "lead_time_minutes": lead_time,
            "detection_delay_minutes": detection_delay,

            "detected_before_peak": int(
                pd.notna(first_alarm_time) and
                first_alarm_time < peak
            )
        }

        row.update(peak_warning_results)

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    event_results,
    episodes,
    predictions,
    dataset_name,
    threshold
):

    total_events = len(event_results)

    warned = int(
        event_results["pre_onset_warning"].sum()
    )

    detected = int(
        event_results["detected_during_event"].sum()
    )

    event_hits = int(
        (
            (event_results["pre_onset_warning"] == 1) |
            (event_results["detected_during_event"] == 1)
        ).sum()
    )

    total_alarm_episodes = len(episodes)

    false_alarms = int(
        (episodes["alarm_type"] == "false_alarm").sum()
    )

    early_warning_alarms = int(
        (episodes["alarm_type"] == "early_warning").sum()
    )

    during_event_alarms = int(
        (episodes["alarm_type"] == "during_event").sum()
    )

    observed_days = (
        predictions["timestamp"].max().date() -
        predictions["timestamp"].min().date()
    ).days + 1

    false_alarms_per_day = (
        false_alarms / observed_days
        if observed_days else 0
    )

    # Episode-level alarm precision.
    # This is the fraction of alarm episodes associated with
    # an active event or a flare beginning within 60 minutes.
    alarm_precision = (
        (early_warning_alarms + during_event_alarms) /
        total_alarm_episodes
        if total_alarm_episodes else 0
    )

    event_recall = (
        event_hits / total_events
        if total_events else 0
    )

    event_detection_recall = (
        detected / total_events
        if total_events else 0
    )

    warning_recall = (
        warned / total_events
        if total_events else 0
    )

    result = {
        "dataset": dataset_name,
        "threshold": threshold,
        "events": total_events,
        "events_with_warning_or_detection": event_hits,
        "pre_onset_events": warned,
        "events_detected_during_interval": detected,
        "event_recall": event_recall,
        "pre_onset_warning_recall": warning_recall,
        "in_event_detection_recall": event_detection_recall,
        "alarm_episodes": total_alarm_episodes,
        "early_warning_episodes": early_warning_alarms,
        "during_event_episodes": during_event_alarms,
        "false_alarm_episodes": false_alarms,
        "false_alarms_per_day": false_alarms_per_day,
        "alarm_episode_precision": alarm_precision,
        "observed_days": observed_days
    }

    for minutes in PEAK_WARNING_MINUTES:

        column = f"before_peak_{minutes}m"

        result[f"events_detected_{minutes}m_before_peak"] = int(
            event_results[column].sum()
        )

        result[f"recall_{minutes}m_before_peak"] = (
            event_results[column].mean()
            if total_events else 0
        )

    detected_delays = event_results[
        "detection_delay_minutes"
    ].dropna()

    result["mean_detection_delay_minutes"] = (
        detected_delays.mean()
        if not detected_delays.empty else np.nan
    )

    result["median_detection_delay_minutes"] = (
        detected_delays.median()
        if not detected_delays.empty else np.nan
    )

    lead_times = event_results[
        "lead_time_minutes"
    ].dropna()

    result["mean_lead_time_minutes"] = (
        lead_times.mean()
        if not lead_times.empty else np.nan
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    events = load_goes_events()

    all_metrics = []
    all_event_results = []
    all_alarm_results = []

    for dataset_name, dataset_path in DATASETS.items():

        predictions = generate_predictions(
            dataset_path,
            dataset_name
        )

        selected_events = select_events(
            events,
            predictions
        )

        print(
            f"\n{dataset_name}: "
            f"{len(selected_events)} overlapping GOES events"
        )

        for threshold in THRESHOLDS:

            print(
                f"\n{'=' * 70}\n"
                f"{dataset_name.upper()} | THRESHOLD {threshold:.2f}\n"
                f"{'=' * 70}"
            )

            episodes = extract_alarm_episodes(
                predictions,
                threshold
            )

            episodes = classify_alarm_episodes(
                episodes,
                selected_events
            )

            event_results = evaluate_events(
                selected_events,
                episodes,
                dataset_name,
                threshold
            )

            metrics = calculate_metrics(
                event_results,
                episodes,
                predictions,
                dataset_name,
                threshold
            )

            all_metrics.append(metrics)

            if not event_results.empty:
                all_event_results.append(event_results)

            if not episodes.empty:

                episode_copy = episodes.copy()
                episode_copy["dataset"] = dataset_name
                episode_copy["threshold"] = threshold

                all_alarm_results.append(episode_copy)

            print(f"GOES events: {metrics['events']}")
            print(
                "Events with warning or detection: "
                f"{metrics['events_with_warning_or_detection']}"
            )
            print(
                "Pre-onset warning events: "
                f"{metrics['pre_onset_events']}"
            )
            print(
                "Events detected during flare: "
                f"{metrics['events_detected_during_interval']}"
            )

            print(
                f"Event recall: {metrics['event_recall']:.4f}"
            )

            print(
                "Pre-onset warning recall: "
                f"{metrics['pre_onset_warning_recall']:.4f}"
            )

            print(
                "In-event detection recall: "
                f"{metrics['in_event_detection_recall']:.4f}"
            )

            print(
                f"Total alarm episodes: "
                f"{metrics['alarm_episodes']}"
            )

            print(
                f"Early-warning episodes: "
                f"{metrics['early_warning_episodes']}"
            )

            print(
                f"False alarm episodes: "
                f"{metrics['false_alarm_episodes']}"
            )

            print(
                f"False alarms/day: "
                f"{metrics['false_alarms_per_day']:.4f}"
            )

            print(
                "Alarm episode precision: "
                f"{metrics['alarm_episode_precision']:.4f}"
            )

            print(
                f"Mean detection delay: "
                f"{metrics['mean_detection_delay_minutes']:.2f} minutes"
                if pd.notna(metrics["mean_detection_delay_minutes"])
                else "Mean detection delay: N/A"
            )

            print(
                f"Median detection delay: "
                f"{metrics['median_detection_delay_minutes']:.2f} minutes"
                if pd.notna(metrics["median_detection_delay_minutes"])
                else "Median detection delay: N/A"
            )

            for minutes in PEAK_WARNING_MINUTES:

                print(
                    f"Events detected >= {minutes} min before peak: "
                    f"{metrics[f'events_detected_{minutes}m_before_peak']}/"
                    f"{metrics['events']}"
                )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = OUTPUT_DIR / "early_warning_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("\nSaved metrics:")
    print(metrics_path)

    if all_event_results:

        event_df = pd.concat(
            all_event_results,
            ignore_index=True
        )

        event_path = OUTPUT_DIR / "event_warning_details.csv"

        event_df.to_csv(
            event_path,
            index=False
        )

        print("Saved event details:")
        print(event_path)

    if all_alarm_results:

        alarm_df = pd.concat(
            all_alarm_results,
            ignore_index=True
        )

        alarm_path = OUTPUT_DIR / "alarm_episode_details.csv"

        alarm_df.to_csv(
            alarm_path,
            index=False
        )

        print("Saved alarm details:")
        print(alarm_path)

    print("\n" + "=" * 70)
    print("EARLY WARNING ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nFinal metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

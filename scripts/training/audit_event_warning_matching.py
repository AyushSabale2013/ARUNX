
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

OUTPUT_DIR = MODEL_DIR / "event_warning_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "June": DATA_DIR / "classification_validation.parquet",
    "July": DATA_DIR / "classification_test.parquet"
}

THRESHOLDS = [0.05, 0.10, 0.20]

WARNING_WINDOW_MINUTES = 60

LEAD_TIME_MINUTES = [5, 10, 15, 30, 60]

# Alarm episodes separated by more than one minute are distinct.
EPISODE_GAP_MINUTES = 1

# Persistence definition: at least 3 consecutive observations.
PERSISTENCE_MIN_OBSERVATIONS = 3


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

    print("\nLoading GOES events...")

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

    print(f"Valid M/X events: {len(events)}")
    print(events["flare_type"].value_counts())

    return events


# ============================================================
# GENERATE MODEL PREDICTIONS
# ============================================================

def generate_predictions(path, dataset_name):

    print(f"\nGenerating {dataset_name} predictions...")

    df = pd.read_parquet(path)

    if not {"timestamp", "target"}.issubset(df.columns):
        raise ValueError("Dataset must contain timestamp and target.")

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

    predictions = pd.DataFrame({
        "timestamp": timestamps,
        "true_class": df["target"].to_numpy(),
        "probability": probabilities
    })

    predictions = predictions.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Observations: {len(predictions)}")
    print(
        f"Observation period: "
        f"{predictions['timestamp'].min()} to "
        f"{predictions['timestamp'].max()}"
    )

    return predictions


# ============================================================
# SELECT EVENTS OVERLAPPING OBSERVATION PERIOD
# ============================================================

def select_events(events, predictions):

    data_start = predictions["timestamp"].min()
    data_end = predictions["timestamp"].max()

    selected = events[
        (events["end_time"] >= data_start) &
        (events["start_time"] <= data_end)
    ].copy()

    selected["warning_window_complete"] = (
        selected["start_time"] -
        pd.Timedelta(minutes=WARNING_WINDOW_MINUTES)
        >= data_start
    )

    return selected.reset_index(drop=True)


# ============================================================
# EXTRACT ALARM EPISODES
# ============================================================

def extract_alarm_episodes(predictions, threshold):

    df = predictions.copy()

    df["alarm"] = (
        df["probability"] >= threshold
    )

    positive = df[df["alarm"]].copy()

    if positive.empty:

        return pd.DataFrame(columns=[
            "alarm_id",
            "alarm_start",
            "alarm_end",
            "peak_probability",
            "mean_probability",
            "observations",
            "duration_minutes",
            "isolated_spike",
            "persistent"
        ])

    time_gap = positive["timestamp"].diff()

    new_episode = (
        time_gap.isna() |
        (time_gap > pd.Timedelta(
            minutes=EPISODE_GAP_MINUTES
        ))
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

    episodes.insert(
        0,
        "alarm_id",
        [f"ALARM_{i + 1:05d}" for i in range(len(episodes))]
    )

    # One-minute observations: add one minute to represent
    # the duration of a single-observation episode.
    episodes["duration_minutes"] = (
        episodes["alarm_end"] -
        episodes["alarm_start"]
    ).dt.total_seconds() / 60 + 1

    episodes["isolated_spike"] = (
        episodes["observations"] == 1
    ).astype(int)

    episodes["persistent"] = (
        episodes["observations"] >=
        PERSISTENCE_MIN_OBSERVATIONS
    ).astype(int)

    return episodes


# ============================================================
# ONE-TO-ONE ALARM TO EVENT MATCHING
# ============================================================

def match_alarms_to_events(events, episodes, dataset_name, threshold):

    print("\nPerforming one-to-one event matching...")

    event_results = []
    alarm_results = []

    used_alarm_ids = set()

    events = events.sort_values(
        "start_time"
    ).reset_index(drop=True)

    episodes = episodes.sort_values(
        "alarm_start"
    ).reset_index(drop=True)

    # Each event gets at most one alarm.
    # Each alarm can be assigned to at most one event.
    for _, event in events.iterrows():

        warning_start = (
            event["start_time"] -
            pd.Timedelta(minutes=WARNING_WINDOW_MINUTES)
        )

        eligible = episodes[
            (~episodes["alarm_id"].isin(used_alarm_ids)) &
            (episodes["alarm_start"] >= warning_start) &
            (episodes["alarm_start"] <= event["end_time"])
        ].copy()

        matched_alarm = None

        if not eligible.empty:

            # Choose the earliest eligible alarm.
            matched_alarm = eligible.sort_values(
                "alarm_start"
            ).iloc[0]

            used_alarm_ids.add(
                matched_alarm["alarm_id"]
            )

        if matched_alarm is None:

            alarm_id = None
            alarm_start = pd.NaT
            alarm_type = "missed"
            lead_time = np.nan
            detection_delay = np.nan
            time_to_peak = np.nan
            warning_detected = 0
            detected_during = 0
            event_detected = 0
            alarm_observations = 0
            alarm_duration = np.nan
            alarm_persistent = 0
            alarm_isolated = 0

        else:

            alarm_id = matched_alarm["alarm_id"]
            alarm_start = matched_alarm["alarm_start"]

            lead_time = (
                event["start_time"] - alarm_start
            ).total_seconds() / 60

            time_to_peak = (
                event["peak_time"] - alarm_start
            ).total_seconds() / 60

            if alarm_start < event["start_time"]:

                alarm_type = "pre_onset_warning"
                warning_detected = 1
                detected_during = 0
                detection_delay = np.nan

            else:

                alarm_type = "during_event_detection"
                warning_detected = 0
                detected_during = 1
                detection_delay = lead_time

            event_detected = 1

            alarm_observations = int(
                matched_alarm["observations"]
            )

            alarm_duration = float(
                matched_alarm["duration_minutes"]
            )

            alarm_persistent = int(
                matched_alarm["persistent"]
            )

            alarm_isolated = int(
                matched_alarm["isolated_spike"]
            )

        row = {
            "dataset": dataset_name,
            "threshold": threshold,
            "event_id": event["event_id"],
            "goes_class": event["goes_class"],
            "flare_type": event["flare_type"],
            "start_time": event["start_time"],
            "peak_time": event["peak_time"],
            "end_time": event["end_time"],
            "warning_window_complete": int(
                event["warning_window_complete"]
            ),
            "alarm_id": alarm_id,
            "alarm_time": alarm_start,
            "alarm_type": alarm_type,
            "event_detected": event_detected,
            "pre_onset_warning": warning_detected,
            "detected_during_event": detected_during,
            "lead_time_minutes": lead_time,
            "detection_delay_minutes": detection_delay,
            "time_before_peak_minutes": time_to_peak,
            "alarm_observations": alarm_observations,
            "alarm_duration_minutes": alarm_duration,
            "alarm_persistent": alarm_persistent,
            "alarm_isolated_spike": alarm_isolated
        }

        # Lead-time bins: only genuine pre-onset warnings count.
        for minutes in LEAD_TIME_MINUTES:

            row[f"warning_at_least_{minutes}m"] = int(
                warning_detected == 1 and
                lead_time >= minutes
            )

        event_results.append(row)

        if matched_alarm is not None:

            alarm_row = matched_alarm.to_dict()

            alarm_row.update({
                "dataset": dataset_name,
                "threshold": threshold,
                "matched_event_id": event["event_id"],
                "matched_flare_type": event["flare_type"],
                "alarm_type": alarm_type,
                "lead_time_minutes": lead_time
            })

            alarm_results.append(alarm_row)

    event_df = pd.DataFrame(event_results)

    matched_alarm_df = pd.DataFrame(alarm_results)

    # Unmatched alarms are false/unmatched episodes.
    unmatched = episodes[
        ~episodes["alarm_id"].isin(used_alarm_ids)
    ].copy()

    unmatched["dataset"] = dataset_name
    unmatched["threshold"] = threshold
    unmatched["matched_event_id"] = None
    unmatched["matched_flare_type"] = None
    unmatched["alarm_type"] = "unmatched_alarm"
    unmatched["lead_time_minutes"] = np.nan

    return event_df, matched_alarm_df, unmatched


# ============================================================
# CALCULATE EVENT METRICS
# ============================================================

def calculate_event_metrics(event_df):

    total = len(event_df)

    detected = int(
        event_df["event_detected"].sum()
    )

    warnings = int(
        event_df["pre_onset_warning"].sum()
    )

    during = int(
        event_df["detected_during_event"].sum()
    )

    complete = event_df[
        event_df["warning_window_complete"] == 1
    ]

    complete_warnings = int(
        complete["pre_onset_warning"].sum()
    )

    metrics = {
        "events": total,
        "events_detected": detected,
        "event_recall": detected / total if total else np.nan,
        "pre_onset_warnings": warnings,
        "pre_onset_warning_recall_all": (
            warnings / total if total else np.nan
        ),
        "complete_warning_window_events": len(complete),
        "pre_onset_warning_recall_complete": (
            complete_warnings / len(complete)
            if len(complete) else np.nan
        ),
        "detected_during_event": during,
        "during_event_detection_recall": (
            during / total if total else np.nan
        )
    }

    # Only genuine pre-onset warnings are used for lead-time stats.
    genuine_leads = event_df.loc[
        event_df["pre_onset_warning"] == 1,
        "lead_time_minutes"
    ].dropna()

    metrics["mean_pre_onset_lead_minutes"] = (
        genuine_leads.mean()
        if not genuine_leads.empty else np.nan
    )

    metrics["median_pre_onset_lead_minutes"] = (
        genuine_leads.median()
        if not genuine_leads.empty else np.nan
    )

    for minutes in LEAD_TIME_MINUTES:

        column = f"warning_at_least_{minutes}m"

        count = int(event_df[column].sum())

        metrics[f"warnings_{minutes}m_or_more"] = count

        metrics[f"warning_recall_{minutes}m"] = (
            count / total if total else np.nan
        )

    # Separate M and X metrics.
    for flare_type in ["M", "X"]:

        subset = event_df[
            event_df["flare_type"] == flare_type
        ]

        prefix = f"{flare_type}_"

        metrics[prefix + "events"] = len(subset)

        metrics[prefix + "detected"] = int(
            subset["event_detected"].sum()
        )

        metrics[prefix + "event_recall"] = (
            subset["event_detected"].mean()
            if len(subset) else np.nan
        )

        metrics[prefix + "pre_onset_warnings"] = int(
            subset["pre_onset_warning"].sum()
        )

        metrics[prefix + "warning_recall"] = (
            subset["pre_onset_warning"].mean()
            if len(subset) else np.nan
        )

    return metrics


# ============================================================
# CALCULATE ALARM AND PERSISTENCE METRICS
# ============================================================

def calculate_alarm_metrics(
    all_episodes,
    matched_episodes,
    unmatched_episodes,
    predictions
):

    total_episodes = len(all_episodes)
    matched_count = len(matched_episodes)
    unmatched_count = len(unmatched_episodes)

    observed_days = (
        predictions["timestamp"].max().date() -
        predictions["timestamp"].min().date()
    ).days + 1

    persistent_count = int(
        all_episodes["persistent"].sum()
    ) if total_episodes else 0

    isolated_count = int(
        all_episodes["isolated_spike"].sum()
    ) if total_episodes else 0

    return {
        "alarm_episodes": total_episodes,
        "one_to_one_matched_alarms": matched_count,
        "unmatched_alarm_episodes": unmatched_count,

        "one_to_one_alarm_precision": (
            matched_count / total_episodes
            if total_episodes else np.nan
        ),

        "unmatched_alarms_per_day": (
            unmatched_count / observed_days
            if observed_days else np.nan
        ),

        "observed_days": observed_days,

        "persistent_episodes": persistent_count,
        "persistent_episode_percentage": (
            persistent_count / total_episodes * 100
            if total_episodes else np.nan
        ),

        "isolated_spike_episodes": isolated_count,
        "isolated_spike_percentage": (
            isolated_count / total_episodes * 100
            if total_episodes else np.nan
        ),

        "mean_episode_duration_minutes": (
            all_episodes["duration_minutes"].mean()
            if total_episodes else np.nan
        )
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    dataset_name,
    threshold,
    event_metrics,
    alarm_metrics
):

    print("\n" + "=" * 75)
    print(f"{dataset_name.upper()} | THRESHOLD {threshold:.2f}")
    print("=" * 75)

    print(
        f"Events detected: "
        f"{event_metrics['events_detected']}/"
        f"{event_metrics['events']}"
    )

    print(
        f"Event recall: "
        f"{event_metrics['event_recall']:.4f}"
    )

    print(
        f"Pre-onset warnings: "
        f"{event_metrics['pre_onset_warnings']}/"
        f"{event_metrics['events']}"
    )

    print(
        f"Pre-onset warning recall: "
        f"{event_metrics['pre_onset_warning_recall_all']:.4f}"
    )

    print(
        f"Complete warning-window events: "
        f"{event_metrics['complete_warning_window_events']}"
    )

    print(
        f"Pre-onset recall (complete windows): "
        f"{event_metrics['pre_onset_warning_recall_complete']:.4f}"
    )

    print(
        f"Mean genuine lead time: "
        f"{event_metrics['mean_pre_onset_lead_minutes']:.2f} min"
        if pd.notna(event_metrics["mean_pre_onset_lead_minutes"])
        else "Mean genuine lead time: N/A"
    )

    print(
        f"Median genuine lead time: "
        f"{event_metrics['median_pre_onset_lead_minutes']:.2f} min"
        if pd.notna(event_metrics["median_pre_onset_lead_minutes"])
        else "Median genuine lead time: N/A"
    )

    print("\nLead-time breakdown:")

    for minutes in LEAD_TIME_MINUTES:

        print(
            f"At least {minutes:2d} minutes: "
            f"{event_metrics[f'warnings_{minutes}m_or_more']}/"
            f"{event_metrics['events']}"
        )

    print("\nClass-specific results:")

    for flare_type in ["M", "X"]:

        prefix = f"{flare_type}_"

        print(
            f"{flare_type}-class: "
            f"{event_metrics[prefix + 'detected']}/"
            f"{event_metrics[prefix + 'events']} detected | "
            f"{event_metrics[prefix + 'pre_onset_warnings']} "
            f"pre-onset warnings"
        )

    print("\nAlarm episode analysis:")

    print(
        f"Total alarm episodes: "
        f"{alarm_metrics['alarm_episodes']}"
    )

    print(
        f"One-to-one matched alarms: "
        f"{alarm_metrics['one_to_one_matched_alarms']}"
    )

    print(
        f"Unmatched alarms: "
        f"{alarm_metrics['unmatched_alarm_episodes']}"
    )

    print(
        f"One-to-one alarm precision: "
        f"{alarm_metrics['one_to_one_alarm_precision']:.4f}"
    )

    print(
        f"Unmatched alarms/day: "
        f"{alarm_metrics['unmatched_alarms_per_day']:.4f}"
    )

    print(
        f"Persistent episodes: "
        f"{alarm_metrics['persistent_episodes']}"
    )

    print(
        f"Isolated spike episodes: "
        f"{alarm_metrics['isolated_spike_episodes']}"
    )

    print(
        f"Mean episode duration: "
        f"{alarm_metrics['mean_episode_duration_minutes']:.2f} min"
        if pd.notna(alarm_metrics["mean_episode_duration_minutes"])
        else "Mean episode duration: N/A"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    events = load_goes_events()

    all_metrics = []
    all_event_results = []
    all_matched_alarms = []
    all_unmatched_alarms = []

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

            episodes = extract_alarm_episodes(
                predictions,
                threshold
            )

            event_df, matched_df, unmatched_df = (
                match_alarms_to_events(
                    selected_events,
                    episodes,
                    dataset_name,
                    threshold
                )
            )

            event_metrics = calculate_event_metrics(
                event_df
            )

            alarm_metrics = calculate_alarm_metrics(
                episodes,
                matched_df,
                unmatched_df,
                predictions
            )

            metrics = {
                "dataset": dataset_name,
                "threshold": threshold,
                **event_metrics,
                **alarm_metrics
            }

            all_metrics.append(metrics)

            event_df["dataset"] = dataset_name
            event_df["threshold"] = threshold

            all_event_results.append(event_df)

            if not matched_df.empty:
                all_matched_alarms.append(matched_df)

            if not unmatched_df.empty:
                all_unmatched_alarms.append(unmatched_df)

            print_summary(
                dataset_name,
                threshold,
                event_metrics,
                alarm_metrics
            )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = OUTPUT_DIR / "audit_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(f"\nSaved metrics: {metrics_path}")

    # ========================================================
    # SAVE EVENT DETAILS
    # ========================================================

    if all_event_results:

        event_df = pd.concat(
            all_event_results,
            ignore_index=True
        )

        event_path = OUTPUT_DIR / "event_matching_details.csv"

        event_df.to_csv(
            event_path,
            index=False
        )

        print(f"Saved event details: {event_path}")

    # ========================================================
    # SAVE MATCHED ALARMS
    # ========================================================

    if all_matched_alarms:

        matched_df = pd.concat(
            all_matched_alarms,
            ignore_index=True
        )

        matched_path = OUTPUT_DIR / "matched_alarm_details.csv"

        matched_df.to_csv(
            matched_path,
            index=False
        )

        print(f"Saved matched alarms: {matched_path}")

    # ========================================================
    # SAVE UNMATCHED ALARMS
    # ========================================================

    if all_unmatched_alarms:

        unmatched_df = pd.concat(
            all_unmatched_alarms,
            ignore_index=True
        )

        unmatched_path = OUTPUT_DIR / "unmatched_alarm_details.csv"

        unmatched_df.to_csv(
            unmatched_path,
            index=False
        )

        print(f"Saved unmatched alarms: {unmatched_path}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("EVENT WARNING AUDIT COMPLETE")
    print("=" * 75)

    display_columns = [
        "dataset",
        "threshold",
        "events",
        "event_recall",
        "pre_onset_warning_recall_all",
        "pre_onset_warning_recall_complete",
        "median_pre_onset_lead_minutes",
        "warnings_5m_or_more",
        "warnings_10m_or_more",
        "warnings_15m_or_more",
        "warnings_30m_or_more",
        "warnings_60m_or_more",
        "one_to_one_alarm_precision",
        "unmatched_alarms_per_day",
        "persistent_episode_percentage",
        "isolated_spike_percentage"
    ]

    print(
        metrics_df[display_columns].to_string(index=False)
    )

    print("\nAll results saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

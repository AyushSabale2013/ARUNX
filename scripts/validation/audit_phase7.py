
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("output/labeled_timeseries.parquet")
OUT = Path("output/phase7_audit")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SOURCE)

# 1. Audit every instrument signal
signal_cols = [
    c for c in df.columns
    if c.startswith(("hel1os_", "solexs_"))
    and not c.endswith(("_gti", "_missing", "_stat_err",
                        "_stat_err_missing"))
]

channel_results = []

for col in signal_cols:
    signal = df[col]
    prefix = col

    if prefix == "solexs_sdd2_counts":
        gti_col = "solexs_sdd2_gti"
        missing_col = "solexs_sdd2_missing"
    else:
        gti_col = prefix + "_gti"
        missing_col = prefix + "_missing"
    
    gti = df[gti_col]
    missing_flag = df[missing_col]
    actual_missing = signal.isna()

    result = {
        "channel": col,
        "total_rows": len(df),
        "actual_missing": int(actual_missing.sum()),
        "missing_flag_true": int(missing_flag.sum()),
        "missing_flag_mismatch": int(
            (actual_missing != missing_flag).sum()
        ),
        "gti_true": int(gti.sum()),
        "gti_true_but_missing": int((gti & actual_missing).sum()),
        "gti_false_but_has_value": int(
            ((~gti) & signal.notna()).sum()
        ),
        "non_finite": int(
            (~np.isfinite(signal.dropna())).sum()
        ),
        "negative_values": int((signal < 0).sum()),
    }

    err_col = prefix + "_stat_err"
    if err_col in df.columns:
        err = df[err_col]
        result["negative_stat_errors"] = int((err < 0).sum())
        result["missing_stat_errors"] = int(err.isna().sum())

    channel_results.append(result)

channel_df = pd.DataFrame(channel_results)
channel_df.to_csv(OUT / "channel_quality.csv", index=False)

# 2. Audit timestamps and daily continuity
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")

daily = (
    df.groupby("date")
    .agg(
        rows=("timestamp", "size"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
    .reset_index()
)

daily["missing_minutes"] = 1440 - daily["rows"]

gaps = []
for date, group in df.groupby("date", sort=True):
    ts = group["timestamp"].sort_values()
    delta = ts.diff()
    bad = delta[delta != pd.Timedelta(minutes=1)].dropna()

    for idx, value in bad.items():
        gaps.append({
            "date": date,
            "timestamp": ts.loc[idx],
            "gap_minutes": value.total_seconds() / 60,
        })

daily.to_csv(OUT / "daily_coverage.csv", index=False)
pd.DataFrame(gaps).to_csv(OUT / "timestamp_gaps.csv", index=False)

# 3. Save concise summary
summary = {
    "rows": len(df),
    "columns": len(df.columns),
    "observation_dates": df["date"].nunique(),
    "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
    "dates_not_1440_rows": int((daily["rows"] != 1440).sum()),
    "within_day_timestamp_gaps": len(gaps),
    "channels_with_missing_flag_mismatch": int(
        (channel_df["missing_flag_mismatch"] > 0).sum()
    ),
    "channels_with_gti_missing_conflicts": int(
        (channel_df["gti_true_but_missing"] > 0).sum()
    ),
    "channels_with_values_outside_gti": int(
        (channel_df["gti_false_but_has_value"] > 0).sum()
    ),
}

pd.Series(summary).to_csv(OUT / "audit_summary.csv", header=["value"])

print("\nPHASE 7 AUDIT SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nReports saved in:", OUT)
print("\nCHANNEL QUALITY:")
print(channel_df.to_string(index=False))


import re
import pandas as pd

GOES_FILE = "goes/goes_xray_event_list_2026.txt"

pattern = re.compile(
    r"^\s*(\d{1,2}-[A-Za-z]{3}-\d{4})\s+"
    r"(\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2})\s+"
    r"([A-Z]\d+(?:\.\d+)?)"
)

records = []

with open(GOES_FILE, "r") as file:
    for line_number, line in enumerate(file, start=1):
        match = pattern.match(line)

        if not match:
            continue

        date, start, peak, end, flare_class = match.groups()

        base = pd.to_datetime(
            date, format="%d-%b-%Y", utc=True
        )

        timestamps = []

        for clock in [start, peak, end]:
            timestamps.append(
                pd.to_datetime(
                    date + " " + clock,
                    format="%d-%b-%Y %H:%M",
                    utc=True
                )
            )

        records.append({
            "line": line_number,
            "date": date,
            "start": timestamps[0],
            "peak": timestamps[1],
            "end": timestamps[2],
            "class": flare_class,
            "original": line.strip()
        })

df = pd.DataFrame(records)

issues = []

for _, row in df.iterrows():
    start = row["start"]
    peak = row["peak"]
    end = row["end"]

    # Test same-day chronology first.
    if start <= peak <= end:
        continue

    # Test a normal midnight rollover.
    peak_next = peak + pd.Timedelta(days=1)
    end_next = end + pd.Timedelta(days=1)

    if start <= peak_next <= end_next:
        continue

    issues.append({
        "line": row["line"],
        "class": row["class"],
        "original": row["original"],
        "start": start,
        "peak": peak,
        "end": end
    })

issues_df = pd.DataFrame(issues)

print("Total parsed records:", len(df))
print("Chronology anomalies:", len(issues_df))

if not issues_df.empty:
    print("\nAnomalous records:")
    print(issues_df.to_string(index=False))

issues_df.to_csv(
    "output/goes_chronology_anomalies.csv",
    index=False
)

print("\nSaved: output/goes_chronology_anomalies.csv")

import re
import pandas as pd

GOES_FILE = "goes/goes_xray_event_list_2026.txt"

events = []

pattern = re.compile(
    r"^\s*(\d{1,2}-[A-Za-z]{3}-\d{4})\s+"
    r"(\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2})\s+"
    r"([A-Z]\d+(?:\.\d+)?)"
)

with open(GOES_FILE, "r") as file:
    for line in file:
        match = pattern.match(line)

        if not match:
            continue

        date, start, peak, end, flare_class = match.groups()

        events.append({
            "date": date,
            "start": start,
            "peak": peak,
            "end": end,
            "class": flare_class
        })

df = pd.DataFrame(events)

for col in ["start", "peak", "end"]:
    df[col] = pd.to_datetime(
        df["date"] + " " + df[col],
        format="%d-%b-%Y %H:%M",
        utc=True
    )

df["date"] = pd.to_datetime(
    df["date"],
    format="%d-%b-%Y"
).dt.date

df["class_type"] = df["class"].str[0]
df["intensity"] = df["class"].str[1:].astype(float)

print("Total parsed events:", len(df))

print("\nClass distribution:")
print(df["class_type"].value_counts().sort_index())

print("\n6 October 2026 events:")
print(
    df[df["date"] == pd.Timestamp("2026-10-06").date()]
    [["start", "peak", "end", "class"]]
    .to_string(index=False)
)

df.to_csv("output/goes_events_parsed.csv", index=False)

print("\nSaved: output/goes_events_parsed.csv")



import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

GOES_FILE = Path("goes/goes_xray_event_list_2026.txt")

OUTPUT_DIR = Path("output")
PARSED_OUTPUT = OUTPUT_DIR / "goes_events_parsed.csv"
REVIEW_OUTPUT = OUTPUT_DIR / "goes_events_manual_review.csv"

TIME_FORMAT = "%d-%b-%Y %H:%M"

# Matches:
# Date       Start  Peak   End    Class
# 11-Jan-2026 21:53 23:14 00:31  M3.3
PATTERN = re.compile(
    r"^\s*(\d{1,2}-[A-Za-z]{3}-\d{4})\s+"
    r"(\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2})\s+"
    r"([A-Z]\d+(?:\.\d+)?)"
)


# ============================================================
# PARSE ONE EVENT
# ============================================================

def parse_event(date_str, start_str, peak_str, end_str):
    """
    Parse event timestamps and handle midnight rollover.

    Rules:
    1. Start belongs to the catalogue date.
    2. If peak < start, peak belongs to the next day.
    3. If end < peak, end belongs to the next day.
    4. If chronology remains invalid, flag for review.
    """

    start = pd.to_datetime(
        f"{date_str} {start_str}",
        format=TIME_FORMAT,
        utc=True
    )

    peak = pd.to_datetime(
        f"{date_str} {peak_str}",
        format=TIME_FORMAT,
        utc=True
    )

    end = pd.to_datetime(
        f"{date_str} {end_str}",
        format=TIME_FORMAT,
        utc=True
    )

    # Peak crossed midnight.
    if peak < start:
        peak += pd.Timedelta(days=1)

    # End crossed midnight relative to peak.
    if end < peak:
        end += pd.Timedelta(days=1)

    # Final chronology validation.
    if not (start <= peak <= end):
        return None, (
            f"Invalid chronology: "
            f"start={start}, peak={peak}, end={end}"
        )

    return {
        "date": start.date(),
        "start": start,
        "peak": peak,
        "end": end
    }, None


# ============================================================
# MAIN
# ============================================================

def main():

    if not GOES_FILE.exists():
        raise FileNotFoundError(
            f"GOES catalogue not found: {GOES_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_events = []
    review_events = []

    total_parsed = 0
    unmatched_lines = 0

    with GOES_FILE.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            match = PATTERN.match(line)

            if not match:
                continue

            total_parsed += 1

            date_str, start_str, peak_str, end_str, flare_class = (
                match.groups()
            )

            event, error = parse_event(
                date_str,
                start_str,
                peak_str,
                end_str
            )

            if error:

                review_events.append({
                    "line": line_number,
                    "original": line.strip(),
                    "date": date_str,
                    "start_original": start_str,
                    "peak_original": peak_str,
                    "end_original": end_str,
                    "class": flare_class,
                    "reason": error
                })

                continue

            valid_events.append({
                "date": event["date"],
                "start": event["start"],
                "peak": event["peak"],
                "end": event["end"],
                "class": flare_class
            })

    # ========================================================
    # BUILD VALID DATASET
    # ========================================================

    df = pd.DataFrame(valid_events)

    if not df.empty:

        df["class_type"] = df["class"].str[0]

        df["intensity"] = (
            df["class"].str[1:].astype(float)
        )

        df = df.sort_values(
            ["start", "peak"]
        ).reset_index(drop=True)

    # ========================================================
    # BUILD MANUAL REVIEW DATASET
    # ========================================================

    review_df = pd.DataFrame(review_events)

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    df.to_csv(PARSED_OUTPUT, index=False)

    review_df.to_csv(REVIEW_OUTPUT, index=False)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========== GOES PARSER SUMMARY ==========")

    print(f"Total valid-format records: {total_parsed}")
    print(f"Successfully parsed:        {len(df)}")
    print(f"Manual review required:     {len(review_df)}")

    print("\nClass distribution:")

    if not df.empty:
        print(df["class_type"].value_counts().sort_index())

    print("\nOutput files:")
    print(f"Parsed events: {PARSED_OUTPUT}")
    print(f"Manual review: {REVIEW_OUTPUT}")

    if not review_df.empty:

        print("\n========== MANUAL REVIEW RECORDS ==========")

        print(
            review_df[
                ["line", "class", "original", "reason"]
            ].to_string(index=False)
        )

    print("\n===========================================")


if __name__ == "__main__":
    main()

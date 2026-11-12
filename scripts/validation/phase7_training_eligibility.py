
import pandas as pd

df = pd.read_parquet("output/labeled_timeseries.parquet")

hel_cols = [
    c for c in df.columns
    if c.startswith("hel1os_")
    and not c.endswith((
        "_gti", "_missing", "_stat_err",
        "_stat_err_missing"
    ))
]

has_hel = df[hel_cols].notna().any(axis=1)
has_sol = df["solexs_sdd2_counts"].notna()

usable = has_hel | has_sol

targets = [
    "flare_class",
    "flare_active",
    "flare_peak",
    "flare_15m",
    "flare_30m",
    "flare_60m",
]

print("TOTAL ROWS:", len(df))
print("ROWS WITH SENSOR DATA:", int(usable.sum()))
print("ROWS WITHOUT SENSOR DATA:", int((~usable).sum()))

for target in targets:
    valid = usable & df[target].notna()

    if target.startswith("flare_") and target in [
        "flare_15m", "flare_30m", "flare_60m"
    ]:
        valid &= df[target] != -1

    print(f"\n{target}")
    print("Training-eligible rows:", int(valid.sum()))
    print("Class distribution:")
    print(df.loc[valid, target].value_counts().sort_index().to_dict())

print("\nEligibility report complete.")

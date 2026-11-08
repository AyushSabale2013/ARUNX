import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("output/merged_20261006.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

channels = [
    "hel1os_cdte1_cdte1_lc_band_5.00kev_to_20.00kev",
    "hel1os_czt1_czt1_lc_band_20.00kev_to_40.00kev",
    "solexs_sdd2_counts"
]

events = [
    ("M1.4", "2026-10-06 05:00", "2026-10-06 07:15",
     "2026-10-06 05:38", "2026-10-06 06:10", "2026-10-06 06:51"),

    ("M1.8", "2026-10-06 07:15", "2026-10-06 09:00",
     "2026-10-06 07:47", "2026-10-06 08:05", "2026-10-06 08:13")
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

for i, event in enumerate(events):
    name, left, right, start, peak, end = event

    left = pd.Timestamp(left, tz="UTC")
    right = pd.Timestamp(right, tz="UTC")

    for j, channel in enumerate(channels):
        ax = axes[i, j]
        ax.plot(df["timestamp"], df[channel], linewidth=0.9)

        ax.set_xlim(left, right)
        ax.axvline(pd.Timestamp(start, tz="UTC"), linestyle=":", color="green")
        ax.axvline(pd.Timestamp(peak, tz="UTC"), linestyle="--", color="red")
        ax.axvline(pd.Timestamp(end, tz="UTC"), linestyle=":", color="green")

        ax.set_title(f"{name} - {channel}")
        ax.grid(alpha=0.3)

fig.tight_layout()
plt.savefig("output/flare_zoom.png", dpi=200)
plt.show()

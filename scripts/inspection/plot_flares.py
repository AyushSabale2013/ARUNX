import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("output/merged_20261006.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Representative channels
channels = [
    "hel1os_cdte1_cdte1_lc_band_5.00kev_to_20.00kev",
    "hel1os_czt1_czt1_lc_band_20.00kev_to_40.00kev",
    "solexs_sdd2_counts"
]

flares = [
    {
        "name": "M1.4",
        "start": "2026-10-06 05:38",
        "peak": "2026-10-06 06:10",
        "end": "2026-10-06 06:51"
    },
    {
        "name": "M1.8",
        "start": "2026-10-06 07:47",
        "peak": "2026-10-06 08:05",
        "end": "2026-10-06 08:13"
    }
]

fig, axes = plt.subplots(3, 1, figsize=(15, 11))

for ax, channel in zip(axes, channels):
    ax.plot(df["timestamp"], df[channel], linewidth=0.8)
    ax.set_title(channel)
    ax.set_ylabel("Count rate")
    ax.grid(alpha=0.3)

    for flare in flares:
        start = pd.Timestamp(flare["start"], tz="UTC")
        peak = pd.Timestamp(flare["peak"], tz="UTC")
        end = pd.Timestamp(flare["end"], tz="UTC")

        ax.axvspan(start, end, alpha=0.15)
        ax.axvline(peak, linestyle="--", linewidth=1)
        ax.text(peak, ax.get_ylim()[1] * 0.9, flare["name"])

axes[-1].set_xlabel("Time (UTC)")
fig.tight_layout()

plt.savefig("output/flare_inspection.png", dpi=200)
plt.show()

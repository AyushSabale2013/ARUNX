
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import pandas as pd
import numpy as np

HEL1OS_DIR = Path("raw_helios")

for detector in ["cdte1", "cdte2", "czt1", "czt2"]:

    files = sorted(
        p for p in HEL1OS_DIR.rglob(f"lightcurve_{detector}.fits")
        if "2026/10/06" in str(p)
    )

    print(f"\n===== {detector.upper()} =====")

    for path in files:
        with fits.open(path) as hdul:
            data = hdul[1].data

            times = pd.to_datetime(
                Time(
                    np.asarray(data["MJD"], dtype=float),
                    format="mjd"
                ).to_datetime(
                    timezone=__import__("datetime").timezone.utc
                ),
                utc=True
            )

            differences = times.to_series().diff().dt.total_seconds()

            print(f"\nFile: {path.parent.parent.name}")
            print("First timestamp:", times.min())
            print("Last timestamp: ", times.max())
            print("Rows:", len(times))
            print("Median cadence (seconds):", differences.median())
            print("Minimum interval (seconds):", differences.min())
            print("Maximum interval (seconds):", differences.max())

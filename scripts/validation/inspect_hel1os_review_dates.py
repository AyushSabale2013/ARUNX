
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "raw_hel1os"

TARGET_DATES = {
    "20260107",
    "20260130",
    "20260410",
    "20260710",
}


def get_col(data, name):
    for col in data.columns.names:
        if col.lower() == name.lower():
            return np.asarray(data[col], dtype=float)
    raise KeyError(name)


gti_files = sorted(DATA.rglob("gti*.fits"))

for gti_path in gti_files:

    date_match = next(
        (date for date in TARGET_DATES if date in gti_path.name
         or date in str(gti_path)),
        None
    )

    if date_match is None:
        continue

    detector = gti_path.stem.replace("gti", "").lower()

    folder = "cdte" if detector.startswith("cdte") else "czt"

    lc_path = (
        gti_path.parent.parent
        / folder
        / f"lightcurve_{detector}.fits"
    )

    print("\n" + "=" * 75)
    print(f"DATE: {date_match} | DETECTOR: {detector.upper()}")
    print("=" * 75)

    with fits.open(gti_path) as hdul:
        hdu = next(
            h for h in hdul
            if h.data is not None and h.columns is not None
        )

        starts = get_col(hdu.data, "tstart")
        stops = get_col(hdu.data, "tstop")

    print("GTI intervals:", len(starts))

    for i, (start, stop) in enumerate(zip(starts, stops)):
        print(
            f"GTI {i + 1}: "
            f"{start:.12f} -> {stop:.12f} "
            f"({(stop - start) * 86400:.2f} sec)"
        )

    with fits.open(lc_path) as hdul:

        for hdu in hdul[1:]:

            if hdu.data is None or hdu.columns is None:
                continue

            if "MJD" not in hdu.columns.names:
                continue

            times = get_col(hdu.data, "MJD")

            finite = np.isfinite(times)
            inside = np.zeros(len(times), dtype=bool)

            for start, stop in zip(starts, stops):
                inside |= (
                    finite
                    & (times >= start - 1e-12)
                    & (times <= stop + 1e-12)
                )

            outside_times = times[finite & ~inside]

            print(f"\nENERGY BAND: {hdu.name}")
            print("Total samples:", len(times))
            print("Inside GTI:", int(inside.sum()))
            print("Outside GTI:", len(outside_times))

            if len(outside_times):
                print(
                    "First outside MJD:",
                    f"{outside_times.min():.12f}"
                )
                print(
                    "Last outside MJD:",
                    f"{outside_times.max():.12f}"
                )

                print(
                    "Outside samples before first GTI:",
                    int(np.sum(outside_times < starts.min()))
                )

                print(
                    "Outside samples after last GTI:",
                    int(np.sum(outside_times > stops.max()))
                )

                internal = (
                    (outside_times >= starts.min())
                    & (outside_times <= stops.max())
                )

                print(
                    "Outside samples between GTI boundaries:",
                    int(internal.sum())
                )

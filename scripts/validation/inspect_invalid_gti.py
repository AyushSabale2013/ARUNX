from pathlib import Path
from astropy.io import fits
import numpy as np

dates = [
    "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09",
    "2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
    "2026-07-05", "2026-07-10"
]

root = Path("raw_solexs")

for date in dates:
    date_tag = date.replace("-", "")

    files = [
        p for p in root.rglob("*.gti.gz")
        if date_tag in p.name
    ]

    if not files:
        print(f"\n{date}: NO GTI FILE")
        continue

    path = files[0]

    with fits.open(path) as hdul:
        gti = next(
            h for h in hdul
            if h.data is not None
            and getattr(h.data, "names", None)
        )

        start = np.asarray(gti.data["START"], dtype=float)
        stop = np.asarray(gti.data["STOP"], dtype=float)

        invalid = (
            ~np.isfinite(start)
            | ~np.isfinite(stop)
            | (stop <= start)
        )

        bad_indices = np.where(invalid)[0]

        print(f"\nDate: {date}")
        print(f"File: {path}")
        print(f"Total intervals: {len(start)}")
        print(f"Invalid intervals: {len(bad_indices)}")

        for i in bad_indices:
            print(
                f"INVALID row {i}: "
                f"START={start[i]}, STOP={stop[i]}"
            )

        valid = ~invalid
        s = start[valid]
        e = stop[valid]

        order = np.argsort(s)
        s = s[order]
        e = e[order]

        overlaps = np.where(s[1:] < e[:-1])[0]

        print(f"Overlapping valid intervals: {len(overlaps)}")
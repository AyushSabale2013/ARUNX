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
    tag = date.replace("-", "")

    gti_file = next(
        (p for p in root.rglob("*.gti.gz") if tag in p.name), None
    )
    lc_file = next(
        (p for p in root.rglob("*.lc.gz") if tag in p.name), None
    )

    if gti_file is None or lc_file is None:
        continue

    with fits.open(gti_file) as h:
        gti = next(
            x for x in h
            if x.data is not None and getattr(x.data, "names", None)
        )
        start = np.asarray(gti.data["START"], dtype=float)
        stop = np.asarray(gti.data["STOP"], dtype=float)

    zero_times = start[
        np.isfinite(start) & np.isfinite(stop) & (start == stop)
    ]

    with fits.open(lc_file) as h:
        times = np.asarray(h["RATE"].data["TIME"], dtype=float)
        counts = np.asarray(h["RATE"].data["COUNTS"], dtype=float)

    print(f"\n{'=' * 55}\n{date}")

    for t in zero_times:
        # Exact timestamp match within 1 millisecond.
        matches = np.where(
            np.isclose(times, t, rtol=0, atol=0.001)
        )[0]

        if len(matches) == 0:
            print(f"GTI time={t}: NO matching lightcurve sample")
            continue

        i = matches[0]
        print(f"\nGTI time={t}: matching sample found")

        for j in range(max(0, i - 2), min(len(times), i + 3)):
            print(
                f"  TIME={times[j]:.3f} "
                f"COUNTS={counts[j]} "
                f"offset={times[j] - t:+.3f}s"
            )
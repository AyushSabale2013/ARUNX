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
    file = next(
        (p for p in root.rglob("*.gti.gz") if tag in p.name), None
    )

    if file is None:
        continue

    print(f"\n{'=' * 65}\n{date}: {file.name}")

    with fits.open(file) as hdul:
        gti = hdul["GTI"]

        print("\n--- COMPLETE GTI EXTENSION HEADER ---")
        print(gti.header)

        print("\n--- COLUMN DEFINITIONS ---")
        for col in gti.columns:
            print(
                f"{col.name}: format={col.format}, "
                f"unit={col.unit}, null={col.null}"
            )

        start = np.asarray(gti.data["START"], dtype=float)
        stop = np.asarray(gti.data["STOP"], dtype=float)

        zero_indices = np.where(
            np.isfinite(start) &
            np.isfinite(stop) &
            (start == stop)
        )[0]

        print(f"\nZero-duration rows: {len(zero_indices)}")

        for i in zero_indices:
            print(f"\nZero-duration row index: {i}")

            for j in range(max(0, i - 2), min(len(start), i + 3)):
                print(
                    f"  row={j}, START={start[j]}, STOP={stop[j]}, "
                    f"duration={stop[j] - start[j]:.3f}s"
                )
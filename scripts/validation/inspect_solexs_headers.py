from pathlib import Path
from astropy.io import fits

dates = [
    "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09",
    "2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
    "2026-07-05", "2026-07-10"
]

root = Path("raw_solexs")

keys = [
    "TELESCOP", "INSTRUME", "OBJECT",
    "TIMESYS", "TIMEUNIT", "MJDREF", "MJDREFI", "MJDREFF",
    "TIMEZERO", "TSTART", "TSTOP", "TIMEDEL", "TIMEPIXR",
    "DATE-OBS", "DATE-END", "HDUCLASS", "HDUCLAS1",
    "HDUCLAS2", "HDUCLAS3"
]

for date in dates:
    tag = date.replace("-", "")

    gti_file = next(
        (p for p in root.rglob("*.gti.gz") if tag in p.name), None
    )
    lc_file = next(
        (p for p in root.rglob("*.lc.gz") if tag in p.name), None
    )

    print(f"\n{'=' * 70}")
    print(date)

    for label, file in [("LIGHTCURVE", lc_file), ("GTI", gti_file)]:
        if file is None:
            print(f"\n{label}: FILE NOT FOUND")
            continue

        print(f"\n--- {label}: {file.name} ---")

        with fits.open(file) as hdul:
            for hdu in hdul:
                print(f"\nHDU: {hdu.name}")

                for key in keys:
                    if key in hdu.header:
                        print(f"  {key} = {hdu.header[key]}")

                if hdu.data is not None:
                    names = getattr(hdu.data, "names", None)
                    if names:
                        print(f"  COLUMNS = {names}")
                        print(f"  ROW COUNT = {len(hdu.data)}")
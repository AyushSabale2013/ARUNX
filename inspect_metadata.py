from pathlib import Path
from astropy.io import fits

FILES = (
    list(Path("raw_helios").rglob("*.fits"))
    + list(Path("raw_solexs").rglob("*.fits"))
    + list(Path("raw_solexs").rglob("*.fits.gz"))
    + list(Path("raw_solexs").rglob("*.lc.gz"))
)

for path in FILES:
    print("\n" + "=" * 80)
    print("FILE:", path)

    try:
        with fits.open(path) as hdul:
            for i, hdu in enumerate(hdul):
                print(f"\nHDU {i}: {hdu.name}")

                if hdu.data is not None and hasattr(hdu.data, "columns"):
                    for col in hdu.columns:
                        print(
                            f"  Column: {col.name:<15}"
                            f" Unit: {str(col.unit):<12}"
                            f" Format: {col.format}"
                        )

                for key in [
                    "TIMESYS", "TIMEUNIT", "MJDREF", "MJDREFI",
                    "MJDREFF", "TIMEZERO", "TSTART", "TSTOP",
                    "TIMEDEL", "EXPOSURE", "ONTIME"
                ]:
                    if key in hdu.header:
                        print(f"  {key}: {hdu.header[key]}")

    except Exception as e:
        print("ERROR:", e)

from astropy.io import fits
from pathlib import Path
import gzip

files = list(Path("raw_helios/2026/10/06").rglob("lightcurve*.fits"))

for file in sorted(files):
    print(f"\n{'='*60}")
    print(file)
    with fits.open(file) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"HDU {i}: {hdu.name}")
            if hdu.data is not None:
                print("Columns:", getattr(hdu.columns, "names", None))
                print("Rows:", len(hdu.data))
                print("Header time:", {
                    k: hdul[1].header.get(k)
                    for k in ["MJDREF", "TIMEZERO", "TIMEDEL", "TSTART", "TSTOP"]
                } if len(hdul) > 1 else {})

solexs = next(Path("raw_solexs/AL1_SLX_L1_20261006_v1.0").rglob("*.lc.gz"))
print(f"\n{'='*60}\nSoLEXS: {solexs}")

with gzip.open(solexs, "rb") as f:
    print("First 500 bytes:", f.read(500))

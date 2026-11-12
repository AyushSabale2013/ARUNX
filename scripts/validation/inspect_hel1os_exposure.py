
from pathlib import Path
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "raw_hel1os"

detectors = ["cdte1", "cdte2", "czt1", "czt2"]

for detector in detectors:
    print("\n" + "=" * 65)
    print(f"DETECTOR: {detector.upper()}")
    print("=" * 65)

    gti_files = sorted(
        DATA.rglob(f"gti{detector}.fits")
    )

    if not gti_files:
        print("No GTI file found.")
        continue

    gti_path = gti_files[0]
    segment_dir = gti_path.parent.parent

    lc_path = (
        segment_dir
        / ("cdte" if detector.startswith("cdte") else "czt")
        / f"lightcurve_{detector}.fits"
    )

    print("GTI:", gti_path)
    print("Lightcurve:", lc_path)

    with fits.open(gti_path) as hdul:
        print("\nGTI HDUs:")

        for i, hdu in enumerate(hdul):
            print(f"\nHDU {i}: {hdu.name}")
            print("Rows:", len(hdu.data) if hdu.data is not None else 0)

            for key in [
                "TELESCOP", "INSTRUME", "EXPOSURE",
                "TIMEUNIT", "TIMESYS", "MJDREFI",
                "MJDREFF", "TIMEDEL", "TSTART", "TSTOP"
            ]:
                if key in hdu.header:
                    print(f"{key}: {hdu.header[key]}")

            if hdu.data is not None and hdu.columns:
                print("Columns:", hdu.columns.names)

                if "tstart" in hdu.columns.names:
                    print("First GTI:", hdu.data["tstart"][0],
                          hdu.data["tstop"][0])
                    print("Last GTI:", hdu.data["tstart"][-1],
                          hdu.data["tstop"][-1])

    if not lc_path.exists():
        print("\nMatching lightcurve not found.")
        continue

    with fits.open(lc_path) as hdul:
        print("\nLIGHTCURVE HDUs:")

        for i, hdu in enumerate(hdul):
            print(f"\nHDU {i}: {hdu.name}")
            print("Rows:", len(hdu.data) if hdu.data is not None else 0)

            for key in [
                "TELESCOP", "INSTRUME", "EXPOSURE",
                "TIMEUNIT", "TIMESYS", "MJDREFI",
                "MJDREFF", "TIMEBIN", "TIMEDEL"
            ]:
                if key in hdu.header:
                    print(f"{key}: {hdu.header[key]}")

            if hdu.data is not None and hdu.columns:
                print("Columns:", hdu.columns.names)

                if "MJD" in hdu.columns.names:
                    print("First MJD:", hdu.data["MJD"][0])
                    print("Last MJD:", hdu.data["MJD"][-1])

from pathlib import Path
from astropy.io import fits

files = list(Path("raw_helios").rglob("gti*.fits"))

for path in files:
    print("\n" + "=" * 60)
    print("FILE:", path)

    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"\nHDU {i}: {hdu.name}")

            if i == 0:
                print("Primary header:")
                for key in hdu.header:
                    if key in [
                        "TIMESYS", "TIMEUNIT", "MJDREF",
                        "MJDREFI", "MJDREFF", "TIMEZERO"
                    ]:
                        print(key, "=", hdu.header[key])

            if hdu.data is not None:
                print("Rows:", len(hdu.data))
                print("Columns:", hdu.columns.names)

                if len(hdu.data) > 0:
                    print("First 5 intervals:")
                    for row in hdu.data[:5]:
                        print(row)

                print("HDU header:")
                for key in hdu.header:
                    if key in [
                        "TIMESYS", "TIMEUNIT", "MJDREF",
                        "MJDREFI", "MJDREFF", "TIMEZERO"
                    ]:
                        print(key, "=", hdu.header[key])


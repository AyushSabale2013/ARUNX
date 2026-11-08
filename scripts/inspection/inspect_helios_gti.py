from astropy.io import fits
from pathlib import Path
import pandas as pd

files = sorted(Path("raw_helios/2026/10/06").rglob("gtic*.fits"))

for file in files:
    print(f"\n{'='*65}")
    print(file)

    with fits.open(file) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue

            print("HDU:", hdu.name)
            print("Columns:", hdu.columns.names)

            for row in hdu.data:
                values = []
                for name in hdu.columns.names:
                    value = row[name]
                    if name.upper() in ["START", "STOP"]:
                        value = pd.to_datetime(
                            value, unit="s", origin="unix", utc=True
                        )
                    values.append(value)
                print(values)

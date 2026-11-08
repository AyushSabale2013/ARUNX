
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import astropy.units as u


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[2]

HEL1OS_DIR = BASE / "raw_helios"
SOLEXS_DIR = BASE / "raw_solexs"
OUTPUT_DIR = BASE / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. HELPER: READ ONE LIGHT-CURVE HDU
# --------------------------------------------------

def read_hdu(hdu, primary_header, instrument_name):

    if hdu.data is None or hdu.columns is None:
        return None

    columns = hdu.columns.names
    column_map = {c.upper(): c for c in columns}

    print(f"\nHDU: {hdu.name}")
    print("Columns:", columns)

    # --------------------------------------------------
    # Identify timestamp column
    # --------------------------------------------------

    if "MJD" in column_map:
        time_col = column_map["MJD"]
        time_values = np.asarray(
            hdu.data[time_col], dtype=float
        )

        # MJD is already an absolute timestamp.
        utc_times = Time(
            time_values,
            format="mjd",
            scale="utc"
        ).to_datetime()

    elif "TIME" in column_map:

        time_col = column_map["TIME"]
        time_values = np.asarray(
            hdu.data[time_col], dtype=float
        )

        header = hdu.header

        def get_header_value(key, default=None):
            return header.get(
                key,
                primary_header.get(key, default)
            )

        time_unit = get_header_value("TIMEUNIT", "s")
        time_system = str(
            get_header_value("TIMESYS", "UTC")
        ).lower()

        supported_scales = {
            "utc", "tai", "tt", "tdb", "tcg", "tcb"
        }

        if time_system not in supported_scales:
            raise ValueError(
                f"Unsupported TIMESYS: {time_system}"
            )

        mjdref = get_header_value("MJDREF")

        if mjdref is None:
            mjdrefi = get_header_value("MJDREFI", 0)
            mjdreff = get_header_value("MJDREFF", 0)
            mjdref = float(mjdrefi) + float(mjdreff)

        timezero = float(get_header_value("TIMEZERO", 0))

        seconds = (
            time_values + timezero
        ) * u.Unit(time_unit).to(u.s)

        mjd_values = mjdref + seconds / 86400.0

        utc_times = Time(
            mjd_values,
            format="mjd",
            scale=time_system
        ).utc.to_datetime()

    else:
        print("No supported timestamp column. Skipping HDU.")
        return None

    # --------------------------------------------------
    # Identify signal column
    # --------------------------------------------------

    excluded = {
        "TIME", "MJD", "ISOT", "TIMEDEL", "TIMEPIXR",
        "ERROR", "STAT_ERR", "SYS_ERR", "FRACEXP",
        "QUALITY", "CHANNEL"
    }

    signal_cols = [
        c for c in columns
        if c.upper() not in excluded
        and np.issubdtype(
            np.asarray(hdu.data[c]).dtype,
            np.number
        )
    ]

    if not signal_cols:
        print("No numeric signal column. Skipping HDU.")
        return None

    preferred = ["RATE", "COUNTS", "COUNT", "CTR"]

    signal_col = next(
        (
            column_map[name]
            for name in preferred
            if name in column_map and name not in excluded
        ),
        signal_cols[0]
    )

    signal = np.asarray(
        hdu.data[signal_col],
        dtype=float
    )

    if signal.ndim > 1:
        signal = signal[:, 0]

    # --------------------------------------------------
    # Clean invalid observations
    # --------------------------------------------------

    utc_times = np.asarray(utc_times)

    valid = np.isfinite(signal)

    utc_times = utc_times[valid]
    signal = signal[valid]

    if len(signal) == 0:
        print("No valid observations. Skipping HDU.")
        return None

    print("Timestamp column:", time_col)
    print("Signal column:", signal_col)
    print("Valid observations:", len(signal))
    print("First UTC time:", utc_times[0])
    print("Last UTC time:", utc_times[-1])

    return {
        "times": utc_times,
        "signal": signal,
        "signal_name": signal_col,
        "band": hdu.name
    }


# --------------------------------------------------
# 3. READ ALL LIGHT-CURVE HDUs IN A FITS FILE
# --------------------------------------------------

def read_lightcurve(file_path, instrument_name):

    print("\n" + "=" * 70)
    print("INSTRUMENT:", instrument_name)
    print("FILE:", file_path.name)
    print("=" * 70)

    results = []

    with fits.open(file_path) as hdul:

        hdul.info()

        primary_header = hdul[0].header

        for hdu in hdul[1:]:

            result = read_hdu(
                hdu,
                primary_header,
                instrument_name
            )

            if result is not None:
                results.append(result)

    if not results:
        raise ValueError(
            f"No usable light curves found in {file_path}"
        )

    return results


# --------------------------------------------------
# 4. FIND AND READ HEL1OS FILES
# --------------------------------------------------

helios_files = sorted(
    HEL1OS_DIR.glob(
        "2026/11/05/*/cdte/lightcurve_cdte*.fits"
    )
)

if not helios_files:
    raise FileNotFoundError(
        "No HEL1OS FITS files found."
    )

helios_data = []

for file_path in helios_files:

    results = read_lightcurve(
        file_path,
        "HEL1OS"
    )

    for result in results:

        helios_data.append(
            (file_path.name, result)
        )


# --------------------------------------------------
# 5. FIND AND READ SOLEXS FILE
# --------------------------------------------------

solexs_files = sorted(
    SOLEXS_DIR.rglob("*.lc.gz")
)

if not solexs_files:
    raise FileNotFoundError(
        "No SoLEXS .lc.gz file found."
    )

solexs_data = []

for file_path in solexs_files:

    results = read_lightcurve(
        file_path,
        "SoLEXS"
    )

    for result in results:
        solexs_data.append(result)


# --------------------------------------------------
# 6. PLOT BOTH INSTRUMENTS
# --------------------------------------------------

fig, axes = plt.subplots(
    2, 1,
    figsize=(16, 11),
    sharex=True
)


# HEL1OS: all available files and energy bands

for filename, result in helios_data:

    axes[0].plot(
        result["times"],
        result["signal"],
        label=f'{filename} | {result["band"]}',
        linewidth=0.7
    )

axes[0].set_title(
    "HEL1OS CdTe Light Curves - 5 November 2026"
)

axes[0].set_ylabel("Recorded counts (CTR)")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)


# SoLEXS: available SDD2 light curve

for result in solexs_data:

    axes[1].plot(
        result["times"],
        result["signal"],
        label=f'SoLEXS SDD2 | {result["signal_name"]}',
        linewidth=0.8
    )

axes[1].set_title(
    "SoLEXS SDD2 Light Curve - 5 November 2026"
)

axes[1].set_ylabel("Recorded signal")
axes[1].set_xlabel("Time (UTC)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)


fig.suptitle(
    "Aditya-L1: HEL1OS and SoLEXS Light Curve Inspection",
    fontsize=14
)

fig.tight_layout()

plot_path = OUTPUT_DIR / "lightcurve_comparison.png"

plt.savefig(
    plot_path,
    dpi=200,
    bbox_inches="tight"
)

print("\n" + "=" * 70)
print("HEL1OS light curves:", len(helios_data))
print("SoLEXS light curves:", len(solexs_data))
print("PLOT SAVED:", plot_path)
print("=" * 70)

plt.show()

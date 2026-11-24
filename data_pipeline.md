# Data pipeline and label reference

## Sources and current scope

| Source | Input | Use |
| --- | --- | --- |
| HEL1OS | Four detector FITS light curves, five bands per detector, detector GTIs | Hard X-ray channels, errors, and valid exposure windows |
| SoLEXS | SDD2 FITS light curve and paired GTI | Soft X-ray counts and valid exposure windows |
| GOES | X-ray event catalogue text | Independent event start/peak/end/class labels; never a model input |

The current dataset contract is M/X classification/nowcasting and 15-, 30-, and 60-minute M/X onset forecasting. The proposal PDFs also discuss C/M/X or four-state labels and 1/3/6/12/24-hour forecasts; these are alternate scientific definitions, not generated outputs in this code path.

## Current processing sequence

1. Inventory PRADAN ZIP files, compute SHA-256, check ZIP integrity, and record acquisition status. New remote URLs must come from an authenticated portal selection; credentials stay in `PRADAN_COOKIE`, outside source files and manifests.
2. Selectively extract HEL1OS light curves/GTIs and SoLEXS SDD2 light curves/GTIs. Validate archive member paths, file size, and CRC.
3. Parse GOES event rows into UTC start, peak, end, class letter and numeric intensity. Malformed date-like rows and chronology errors are written to review reports.
4. Read instrument time and signal columns. HEL1OS MJD/CTR/STAT_ERR and SoLEXS TIME/COUNTS are converted to UTC; non-finite timestamps/measurements and samples outside GTIs are masked. Actual GTI exposure remains separate from signal missingness.
5. Aggregate valid samples by the 1-minute UTC grid. Samples in `[10:00,10:01)` are assigned timestamp `10:01`. Each source date has 1,440 bins, with the last endpoint at 00:00 of the following date. The stored `date` names the represented bin's day.
6. Label M/X event overlap as `flare_active`; assign `flare_class` 0=no active M/X, 1=M, 2=X (maximum class retained on overlap); mark the minute-end containing each GOES peak as `flare_peak`. Event timing and associations are reported per event.
7. Create onset targets `flare_15m`, `flare_30m`, and `flare_60m`: 1 if an M/X event begins in `(timestamp, timestamp+horizon]`, 0 if no onset occurs in that period, and -1 if already inside an M/X event or the full future window is not observable in the same continuous segment.
8. Build availability, log-signal, missing/GTI, HEL1OS error, rolling mean/std/count/min/max (5/15/30/60 minutes), lag differences (1/5/15 minutes), and cyclical UTC time features. Rolling calculations use only current and earlier measurements and reset across gaps/unusable rows.
9. Split by the date represented by each minute-end row, using configured exclusive dates. Forecast rows near the next split boundary are purged by their prediction horizon. Task tables contain only known labels and usable rows. Save Parquet files, feature schema, split summary, reports and hashes per run.

## Important columns

- `timestamp`: right edge of a completed UTC minute.
- Signal columns: HEL1OS channel count rates and SoLEXS SDD2 counts, averaged per minute; these source values are not automatically interchangeable physical fluxes.
- `<signal>_gti`: some in-GTI source samples existed in the bin.
- `<signal>_missing`: no finite in-GTI signal survived aggregation.
- HEL1OS `<signal>_stat_err`: aggregate statistical error; `_stat_err_missing` records absent errors.
- `flare_active`, `flare_class`, `flare_peak`: GOES M/X interval, class, and peak-bin labels.
- `flare_15m`, `flare_30m`, `flare_60m`: onset forecast target with `-1` as unknown/excluded.
- Model-ready tables contain `timestamp`, `target`, and features only. They do not include GOES event metadata or target fields as inputs.

## Validation and retained gaps

The run validator reads every Parquet artifact and checks timestamp ordering/uniqueness/timezone, exact 1,440 rows for each represented UTC date, signal/GTI/missing consistency, finite values, target domains/distributions, feature schema and label exclusion, train/validation/test date boundaries, and readable task files. It writes `reports/preprocessing_report.json` and `reports/dataset_quality.csv`.

Raw FITS scientific audit reports include per-file cadence, duplicate timestamps, GTI validity, and in/out-of-GTI sample counts. A previous full audit on the staged Jan-Jul FITS found 8,708 detail rows passing; the run-specific report preserves that audit evidence when extracted FITS are reused.

The following require further science review before publication or an operational claim: time-system confirmation from Level-1 headers/manuals, detector/band rationale, GOES-to-bin boundary policy, missing coverage around events, class counts per split, and the validity of the proposed split period. SEP contamination, eclipse/pointing flags, saturation, and calibration status should be added only when trusted metadata is available. The raw HEL1OS/SoLEXS ratio, four-state targets, and longer horizons remain unimplemented until units and label definitions are agreed.

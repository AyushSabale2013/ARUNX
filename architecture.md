# ARUN-X data pipeline architecture

## Scope

This workspace implements the offline scientific data pipeline. It does not contain the React/FastAPI application described in some attached proposal and architecture PDFs. Those PDFs refer to another repository snapshot with synthetic demo data; the current workspace includes real Aditya-L1 archive files and multi-day FITS processing.

## Data flow

```text
PRADAN HEL1OS/SoLEXS archives -> inventory/checksum -> selective FITS extraction
                                                 -> UTC + GTI validation
GOES event catalogue -> parse/manual review ----> minute-end daily merges
                                                 -> active/class/peak labels
                                                 -> segment-aware forecast targets
                                                 -> past-only features + chronological splits
                                                 -> validated Parquet task datasets
```

## Components

- `scripts/pipeline/acquire_data.py`: archive manifest, local inventory, URL-list import from PRADAN selector scripts, optional resumable authenticated downloads.
- `scripts/coverage/extract_archives.py`: safe selective extraction of required FITS/GTI members.
- `scripts/validation/validate_extracted_data.py` and `validate_scientific_quality.py`: file/HDU/GTI and raw sampling audits.
- `scripts/pipeline/parse_goes.py`: UTC event parsing plus manual-review and unmatched-record reports.
- `scripts/pipeline/merge_lightcurves.py`: per-date HEL1OS/SoLEXS FITS readers, GTI masking, 1-minute means, checkpointing, and multi-day table assembly.
- `scripts/pipeline/label_flares.py`: GOES M/X active, class, and peak labels plus an event association report.
- `scripts/pipeline/create_forecast_labels.py`: 15/30/60-minute M/X onset targets with unknown labels across active events or incomplete observation segments.
- `scripts/pipeline/build_ml_dataset.py`: 592 past-only features and time-based task splits.
- `scripts/pipeline/run_preprocessing.py`: run-scoped orchestrator, logging, date inventory, and report assembly. Each invocation writes a fresh folder under `output/preprocessing_runs/`.
- `scripts/validation/validate_preprocessing_run.py`: end-to-end Parquet, target, feature leakage, timestamp, split, extraction, archive, and quality-report checks.

## Core contracts

- Inputs: Level-1 HEL1OS and SoLEXS FITS products/GTIs and a GOES event catalogue.
- Output cadence: UTC minute-end; a row at `10:01` represents `[10:00, 10:01)`.
- Signals: 20 HEL1OS detector-band count-rate channels and SoLEXS SDD2 counts, retaining missing/GTI flags and HEL1OS statistical errors.
- Targets: current M/X contract only; active/class/peak labels and 15/30/60-minute onset forecasts. GOES-derived data is never a model feature.
- ML row eligibility: config defaults to at least one valid signal from both instruments. Missing channel values remain NaN with flags; training code fits imputation/scaling from train only.
- Output directory: each run has independent `merged/`, `labels/`, `ml_ready/`, `reports/`, and `logs/` directories plus the effective config and run manifest.

## Important limitations

- The current merger supports the 1-minute minute-end grid only. Other cadences require changes to binning, target shifts, feature windows, and split validation together.
- A HEL1OS-to-SoLEXS ratio is not emitted because raw count rate and counts have not been shown to be calibrated into comparable physical units.
- GOES timestamps are interpreted as UTC by the parser; confirm this against source metadata before scientific claims. Level-1 detector selection and energy-channel interpretation should be documented with the instrument manuals.
- Per-date data gaps remain explicit and forecast horizons crossing a gap are unknown. The generated 1,440-row daily grid is not a claim of 24-hour exposure.
- A live inference API, alert system, complete model package registry, model evaluation, and application integration are outside this preprocessing workspace.

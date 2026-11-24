# ARUN-X project overview

## Goal

Build a trustworthy dataset from Aditya-L1 HEL1OS hard X-ray observations and SoLEXS soft X-ray observations, associate the measurements with GOES flare events, and investigate machine-learning forecasts for M/X-class flares.

## Prediction tasks

- **Active-flare detection / nowcasting:** determine whether an M/X flare is active at the current minute.
- **Flare class:** classify an active event as M or X (the current label table also uses class 0 for no active M/X event).
- **Forecasting:** predict whether an M/X event will start within a selected future window. The current example generates 15-, 30-, and 60-minute targets.

These tasks have different label populations and should be evaluated separately. A forecast target is not equivalent to active-flare detection.

## Current status

The repository now contains a configurable multi-day preprocessing run for 1 January–31 July 2026. Run `preproc_20261124_v1` generated a 207-date minute-end dataset, GOES labels, 592 past-only features, and chronological train/validation/test files for six tasks. Its report is `PASS_WITH_REVIEW`: five requested dates are missing and one malformed GOES row is under review. No model performance has been measured, and the test split contains only one X-class event.

## Main directories

| Path | Contents |
| --- | --- |
| `raw_hel1os/`, `raw_solexs/` | Downloaded source observations |
| `goes/` | GOES event catalogue input |
| `scripts/pipeline/` | Parsing, merging, and labeling scripts |
| `scripts/coverage/` | Instrument availability and overlap analysis |
| `scripts/inspection/` | Plotting and data inspection tools |
| `output/` | Derived tables, reports, and plots |
| `downloaders/`, `logs/` | Data acquisition utilities and logs |

## Recommended next milestones

1. Resolve the five missing dates and malformed GOES event, then approve the common-coverage eligibility rule.
2. Confirm time conventions and label policies against source documentation and representative events.
3. Finalize quality thresholds and chronological train/validation/test date ranges before model evaluation.
4. Train baselines and a gradient-boosted tree model; consider temporal deep learning after enough independent events are collected.
5. Compare models on held-out dates and document event-level performance before integrating predictions into an application.

See `data_pipeline.md` and `DATA_PREPROCESSING_STATUS.md` for the implemented pipeline and remaining work. `PREPROCESSING_AND_MODEL_GUIDE.md` is a historical one-day walkthrough.

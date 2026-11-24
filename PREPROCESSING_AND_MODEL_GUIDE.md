> **Historical walkthrough:** This document describes the earlier one-day example and is not the current pipeline status. For the verified multi-day workflow, see [README.md](README.md), [data_pipeline.md](data_pipeline.md), and [DATA_PREPROCESSING_STATUS.md](DATA_PREPROCESSING_STATUS.md).

# ARUN-X preprocessing and model guide

## What the pipeline does

This historical one-day walkthrough describes an earlier implementation for 6 October 2026. The current multi-day implementation, data outputs, and review status are documented in [README.md](README.md), [data_pipeline.md](data_pipeline.md), and [DATA_PREPROCESSING_STATUS.md](DATA_PREPROCESSING_STATUS.md).

### 1. Read the GOES event catalogue

[`scripts/pipeline/parse_goes.py`](scripts/pipeline/parse_goes.py) parses each catalogue row into an event date, start, peak, end, flare class, class letter, and intensity. Event timestamps are represented in UTC, and the parsed catalogue is saved as `output/goes_events_parsed.csv`.

### 2. Read HEL1OS light curves and good-time intervals

[`scripts/pipeline/merge_lightcurves.py`](scripts/pipeline/merge_lightcurves.py) finds HEL1OS FITS light curves for the configured date. It reads each channel's MJD timestamps and count-rate (`CTR`) values, converts the timestamps to UTC, and loads the detector's GTI intervals. A sample is retained only when its count rate is finite and its timestamp falls inside a GTI. The merge preserves per-channel signal, missing, and GTI indicators.

### 3. Read SoLEXS and its GTIs

The same merge script reads the SoLEXS SDD2 `RATE` light curve (`TIME` in Unix seconds and `COUNTS`) and its GTI file. It converts sample times to UTC and masks non-finite or out-of-GTI measurements. It currently expects exactly one light curve and one GTI file in the input tree, so the file-selection logic will need to be adapted for a multi-day batch.

### 4. Align the instruments on a common time grid

HEL1OS and SoLEXS rows are outer-joined by timestamp, then measurements are averaged into one-minute bins. GTI flags are true if at least one sample in that minute was in a valid interval. The results are reindexed to the full UTC day (1,440 minutes); absent measurements become NaN, missing flags become true, and absent GTI flags become false. Measurements outside GTIs are set to NaN. The output is `output/merged_20261006.parquet`.

The merge script checks that there are 1,440 unique minute timestamps, 21 signal channels, matching quality-flag columns, no non-missing value outside a GTI, and no inconsistent missing flags.

### 5. Attach flare labels

[`scripts/pipeline/label_flares.py`](scripts/pipeline/label_flares.py) loads the merged table and parsed GOES catalogue. For M- and X-class events, it labels minutes from event start through end as active, assigns class 1 to M and 2 to X (keeping the larger class if events overlap), and marks the event's floored-to-minute peak. It writes `output/labeled_20261006.parquet` with `flare_active`, `flare_class`, and `flare_peak` targets.

### 6. Create forecast targets

[`scripts/pipeline/create_forecast_labels.py`](scripts/pipeline/create_forecast_labels.py) creates 15-, 30-, and 60-minute targets. A row is positive when an M/X event starts strictly after that timestamp and within the selected horizon. Current M/X flare minutes and rows too close to the end of the available data to have a complete future window are marked `-1` (unknown/excluded). Other rows are negative. The result is `output/forecast_labeled_20261006.parquet`.

## Important preparation before model fitting

- Build and audit the complete multi-day dataset before treating these one-day files as training data. One day is a pipeline demonstration, not enough independent flare events to train or evaluate a reliable predictor.
- Split by time (train on earlier dates, validate on later dates, test on the latest held-out dates). Never randomly split neighboring minute rows: adjacent samples and windows overlap and would leak near-duplicate conditions across splits.
- Fit all transformations using training data only. For each signal, consider a log transform such as `log1p` when nonnegative, followed by a robust scaler fitted on training rows. Reuse those fitted parameters unchanged on validation, test, and inference data.
- Preserve missingness and GTI indicators as features. Do not replace missing measurements with zero without a separate missing flag; zero is a real measurement. Exclude or explicitly mark rows where required instruments are unavailable, and report how much coverage remains.
- For forecasting, make input features from timestamps at or before the row being scored. Rolling statistics must be trailing (for example, previous 5, 15, or 60 minutes), never centered or forward-looking. Exclude `flare_active`, `flare_class`, `flare_peak`, forecast target columns, and any other label-derived fields from model inputs.
- Forecast positives will be rare. Use class-weighted training or carefully chosen negative sampling on training data only. Keep the validation/test prevalence natural, and report precision-recall AUC, event-level recall, precision, false alarms per day, and lead time. Accuracy alone can look high for a model that always predicts no flare.
- Audit the GOES label boundaries, event overlaps, missing coverage around events, and the number of distinct events in every chronological split. The current code assigns the GOES event's M/X class across its full active interval; it does not yet create a complete end-to-end multi-day quality-filtering and split pipeline.

## Model recommendation

### Recommended first serious model: gradient-boosted decision trees

For the current scale and tabular minute-level features, start with **CatBoost or LightGBM** using engineered, past-only features. Include recent levels, changes, and trailing-window summaries (for example, means, maxima, and slopes over 5, 15, and 60 minutes), plus the missingness/GTI indicators. Train a separate binary model for each forecast horizon, or select one horizon as the primary objective before tuning.

This is a practical fit for a small number of labeled events and mixed signal/quality features. It is easier to validate than a deep sequence model and provides a meaningful benchmark before adding model complexity. Use class weighting, tune the decision threshold on validation data for the desired false-alarm/recall trade-off, and compare against both a no-skill/climatology baseline and logistic regression. If CatBoost/LightGBM is not available in the project environment, use scikit-learn's `HistGradientBoostingClassifier` or `RandomForestClassifier` as a readily available tree baseline.

### Later experiment: temporal CNN or CNN-LSTM

After collecting substantially more independent flare events across dates and cycles, compare the tree model with a small temporal CNN or CNN-LSTM trained on fixed-length, past-only multichannel windows. Keep the chronological holdout and event-level evaluation identical. Deep learning is not the best starting point for the present one-day sample; it can overfit and give misleadingly strong random-split results.

### Suggested training order

1. Climatology/no-skill baseline.
2. Logistic regression on standardized, engineered features.
3. CatBoost or LightGBM with class weighting and validation threshold selection.
4. Temporal CNN/CNN-LSTM once the multi-day event count supports it.

There is no evidence yet to claim one model is empirically best for ARUN-X: the project has not trained and evaluated models on a finalized multi-day dataset. The recommendation above is the best starting experiment, not a measured result.

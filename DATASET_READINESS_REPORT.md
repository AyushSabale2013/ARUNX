# Dataset Readiness Report

**Audit target:** `preproc_20261124_v1`  
**Audit date:** 24 November 2026  
**Scope:** Read-only audit of the 23 run-manifest outputs, saved audit reports/logs, GOES labels, split construction, feature-generation code, and training preprocessing code. The source Parquet outputs were not modified.

## Decision

All six task families are **READY_WITH_LIMITATIONS for exploratory model training**. The files are readable, their saved hashes match, stored targets match an independent reconstruction from the included GOES catalogue, and the current feature-generation code is past-only. This is not a clean confirmatory benchmark: one potentially real M2 event is unresolved and omitted, archive acquisition provenance cannot explain the five date gaps, and the test split contains only one X-class event. Do not make final test-performance or X-class claims from this run.

No task family is marked `READY_FOR_TRAINING` without qualification. The five missing dates do not appear to have been interpolated; forecast targets are unknown when a complete continuous future observation window is unavailable.

### Task decisions

| Task family | Decision | Main limitation |
|---|---|---|
| Classification (0 / M / X) | READY_WITH_LIMITATIONS | The test split has 55 valid M and 1 valid X event; the unresolved July 3 M2 record may make nearby test labels incomplete. C-class events are not a target class. |
| Nowcasting / active flare | READY_WITH_LIMITATIONS | Same M/X event-label limitations; minute rows from one event are correlated and are not independent event counts. |
| Peak detection | READY_WITH_LIMITATIONS | Only 15 validation and 56 test peak-positive rows; threshold selection is event-sparse, with one X event in test. |
| Forecast 15 minutes | READY_WITH_LIMITATIONS | 1,095 / 225 / 774 positive rows across train/validation/test; row positives are clustered around events, and the omitted M2 event may affect July labels. |
| Forecast 30 minutes | READY_WITH_LIMITATIONS | 2,081 / 446 / 1,503 positive rows; same provenance and omitted-event limitations. |
| Forecast 60 minutes | READY_WITH_LIMITATIONS | 3,862 / 890 / 2,695 positive rows; same provenance and omitted-event limitations. |

These decisions permit exploratory model fitting on the designated train split and model selection on validation, with the unresolved label and data-provenance limitations tracked. They do not certify that the science labels or the test set are final. An X-only task or X-specific test evaluation is **NOT_READY** because the test set has one X event.

## Audit method and scope

The audit verified the SHA-256 and byte size of every file listed in `run_manifest.json`, opened JSON and CSV outputs, and iterated every row group and column of each Parquet output. It independently reconstructed active/class/peak labels and 15/30/60-minute targets from the run’s saved GOES event catalogue and minute-end timestamps. It also reviewed the feature schema and implementation code for target leakage, rolling-window direction, split construction, model transforms, and event overlap.

The run used `--skip-extract`; its extraction and raw-sample quality reports were copied from prior evidence rather than freshly regenerated in this run. The saved audit reports record 8,708 of 8,708 raw FITS products passing, 370,885,639 raw samples, and 1,905,151 samples outside GTIs (masked by the merge). Those are inherited reports, not an independent re-audit of each FITS source in this review.

## Dataset inventory

The 23 manifest-listed outputs are grouped by role below. All 23 actual sizes and SHA-256 hashes matched the run manifest. All Parquets were fully readable. No exact duplicate SHA-256 values were found.

| Output path | Format | Rows / dimensions | Bytes | SHA-256 verification |
|---|---:|---:|---:|---|
| `merged/merged_timeseries_minute_end_v2.parquet` | Parquet | 298,080 × 105 | 17,631,228 | `eb3042a311c962a10a1f1a251ec3aa11094f9ec21be96177ddd528262c0f0231` verified |
| `labels/labeled_timeseries_minute_end_v2.parquet` | Parquet | 298,080 × 108 | 17,639,169 | `1ebea61e4cc05ebd4cd0959c63dae00234e874e02303dc69b9ebda7187694a72` verified |
| `labels/forecast_labeled_timeseries_minute_end_v2.parquet` | Parquet | 298,080 × 111 | 17,647,544 | `43ebad815bcca02d5a6d73c956be604650a936015ee310fc8d3eefc23afa0878` verified |
| `ml_ready/feature_schema.json` | JSON | 592 features | 26,495 | `4876d7c2e6efdabeaea1d4dd15ffdc0326d42bcbc16dc52c9caf323667b0d242` verified |
| `ml_ready/dataset_summary.csv` | CSV | 18 data rows × 7 columns | 2,752 | `59cb0aeee5816a67f97c3fbc22daea11dcef76bfe29e232386b9539c3c4410a0` verified |
| `ml_ready/classification_test.parquet` | Parquet | 41,047 × 594 | 77,136,304 | `0cc7632d7eb119e0ee6339bd95b5a93d3c1b69a48eb6075c836247eda7aa0878` verified |
| `ml_ready/classification_train.parquet` | Parquet | 183,280 × 594 | 316,295,537 | `a5825bdda6479fc10d5e44d0aefb1901932c536315fd3bcbe8eb906b5bab594e` verified |
| `ml_ready/classification_validation.parquet` | Parquet | 36,896 × 594 | 67,764,335 | `09d5010528ffbf5b30032776ad5607822531702992dbd5627dc041e00f7df69d` verified |
| `ml_ready/forecast_15m_test.parquet` | Parquet | 39,621 × 594 | 73,816,669 | `8157de877a56412e7b3e46eadb7702770bc20ee520895f3c416047afbe4885de` verified |
| `ml_ready/forecast_15m_train.parquet` | Parquet | 180,758 × 594 | 311,089,686 | `f219143859de45f39210834bb4f0b1932a894519c3eb3c72933272cdf08a6d73` verified |
| `ml_ready/forecast_15m_validation.parquet` | Parquet | 36,417 × 594 | 66,535,362 | `7b7e464a5b2d95eed86c82316c8165ba035ccd0e1f8b1fee7c52dbd5fc75b4a9` verified |
| `ml_ready/forecast_30m_test.parquet` | Parquet | 39,546 × 594 | 73,695,554 | `578f6a4d9291070529d90c86be0ee4ae3c895ae583b7bd3da8eefca8a16eec9a` verified |
| `ml_ready/forecast_30m_train.parquet` | Parquet | 180,469 × 594 | 310,574,043 | `4a6bd95213484bcbf25e2375e31a73d71b454efce06a8d1237154d97074afebc` verified |
| `ml_ready/forecast_30m_validation.parquet` | Parquet | 36,372 × 594 | 66,479,042 | `8f9ee1413b5c10bee579a3ab974ce63abb9c363322c5bb98f311aca2d6550b60` verified |
| `ml_ready/forecast_60m_test.parquet` | Parquet | 39,397 × 594 | 73,415,407 | `a3681c156f0f8fff4db2fe7a509f2f0953c22b2b50bd2b007f721f24abd071ee` verified |
| `ml_ready/forecast_60m_train.parquet` | Parquet | 179,898 × 594 | 309,763,511 | `bb286c245a5a32ddf094b311a81b726d78d9473beac93c07447b648d5c5ce975` verified |
| `ml_ready/forecast_60m_validation.parquet` | Parquet | 36,282 × 594 | 66,332,303 | `b6bce1917d90a03c4e6260757881faf51e385369c2085790ace91094bc854eb3` verified |
| `ml_ready/nowcasting_test.parquet` | Parquet | 41,047 × 594 | 77,136,287 | `96e44cdc10a4eaeefc605a0a922350219cb4e66eec86bf9529845be2e1b18aea` verified |
| `ml_ready/nowcasting_train.parquet` | Parquet | 183,280 × 594 | 316,295,521 | `26e0abbf72b6d0cbdd338965136f95b838c1adf2d2caba1df1be34146b32fe55` verified |
| `ml_ready/nowcasting_validation.parquet` | Parquet | 36,896 × 594 | 67,764,330 | `ad23e2977e24f6357b101c9c4b73449f963810669d813ab9c29c0d9872edd798` verified |
| `ml_ready/peak_detection_test.parquet` | Parquet | 41,047 × 594 | 77,136,263 | `986c1ad137e00a55fd4119d5c9a56cd080b1d30581b0723b389ec0bf1be4c6b2` verified |
| `ml_ready/peak_detection_train.parquet` | Parquet | 183,280 × 594 | 316,295,517 | `b2e2c144dba15258c890aa46df0549854fe0362d881c77415dc365442d0730ac` verified |
| `ml_ready/peak_detection_validation.parquet` | Parquet | 36,896 × 594 | 67,764,329 | `7e12a683f0dd5ae908792219bc87f4458fcadee451bd3085ec063b30528d1d85` verified |

### Schema and readability findings

- Merged master: 298,080 rows × 105 columns. Arrow types: 1 UTC timestamp (`timestamp[us, tz=UTC]`), 1 date string, 41 nullable `double` signal/error columns, and 62 nullable boolean quality flags.
- Active/class/peak table: 298,080 × 108; adds three integer labels (`flare_active`, `flare_class`, `flare_peak`).
- Forecast-labeled table: 298,080 × 111; adds three `int8` horizon targets.
- Each model-ready Parquet: 594 columns = UTC timestamp, `int8` target, and 592 features. Feature types are 528 `double` and 64 `int8`. Feature names are unique and no target/event-name patterns occur in the feature list.
- `feature_schema.json` parses and declares 592 features with `both_instruments` row policy. `dataset_summary.csv` parses and has 18 task/split records (six tasks × three splits).
- The 18 split Parquets share one schema. Classification, nowcasting, and peak detection intentionally reuse the same timestamps and feature rows within each split; only the target changes. Forecast horizons are intentional target-specific subsets. This cross-task reuse is not an accidental duplicate. No two of the 23 files are byte-identical.
- The Parquet full scan found no floating NaNs or infinities. Nullable cells remain in features (for rolling/difference edges and unavailable channel values); they are preserved for training-time imputation. The merged table has expected nulls where a channel lacks valid samples. A full scan of input channels found no negative signal values, so `log1p` did not create invalid values in this run.
- The manifest hashes the 23 deliverable outputs. It does not enumerate every log, report, or per-date checkpoint in the run directory; those supporting files are outside the 23-file hash inventory.

## Data quality and missing dates

The run covers 1 January through 31 July 2026. Its merge contains 207 dates × 1,440 minute-end rows = 298,080 rows. Five of the 212 requested calendar dates have no merged table:

| Missing date | HEL1OS local archives | SoLEXS local archives | Run inventory evidence |
|---|---:|---:|---|
| 2026-02-17 | 0 | 0 | No archive for either instrument in the in-run manifest. |
| 2026-04-16 | 0 | 0 | No archive for either instrument in the in-run manifest. |
| 2026-05-22 | 0 | 0 | No archive for either instrument in the in-run manifest. |
| 2026-06-04 | 0 | 1 | One SoLEXS archive exists; HEL1OS archive is absent, so both-instrument merge did not run. |
| 2026-06-09 | 2 | 0 | Two HEL1OS archives exist; SoLEXS archive is absent, so both-instrument merge did not run. |

### Download-versus-source-availability conclusion

The in-run acquisition manifest contains 633 entries, all marked `local_verified`; it records local names, sizes, and SHA-256 values. It has zero source URLs and zero expected remote checksums. The root `output/acquisition_manifest.json` is empty. The run has an `acquisition_inventory` log showing local inventory only; no remote download-stage log or HTTP/portal result exists. Therefore the evidence shows that the corresponding archive files were absent locally (or one instrument was absent locally), but **does not distinguish upstream unavailability from a selector omission, a download that was never attempted, or a failed download**. No network check was performed during this audit. Do not describe these gaps as confirmed source unavailability or confirmed download failures.

No missing observations were fabricated or interpolated. The forecast label generator requires continuous usable signal coverage through the full horizon; timestamp gaps and unusable intervals split observation segments, so windows spanning gaps become unknown (`-1`) and are excluded from supervised splits. Remaining limitations are reduced exposure/sample size and possible date-selection bias. The run does not provide a quantified per-date coverage threshold or a complete authoritative date eligibility decision.

## GOES event review

The manual-review file identifies source catalogue line 1642: `3-Jul-2026 23:50 12:03 00:05 M2.0`. With the parser’s midnight rules, start is 3 July 23:50 UTC, peak becomes 4 July 12:03 UTC, and end becomes 4 July 00:05 UTC. Since end precedes peak, chronology is invalid. A likely typo such as `00:03` is plausible but not proven; the parser correctly excluded the row instead of inventing a correction.

The exclusion removes this candidate M event from active/class/peak labels and from the 15/30/60-minute future-onset catalogue. Consequently, rows around its candidate interval have no label for it; depending on the other valid event starts and horizon, they may be treated as negative or labeled for a later valid onset. This can create false-negative or mis-timed labels if the row represents a real event. The precise target-count effect cannot be established without the authoritative corrected record. The row falls in the test period, so training and validation labels are not directly affected by this specific exclusion; July test evaluation is.

Valid catalogue events in the requested date range include 278 B, 1,354 C, 191 M, and 13 X events; one additional M2.0 row is excluded for review. Only M and X are used by this task contract. `flare_class=0` means no active M/X event, so B/C periods are included in the 0/background class; C is not a separate supervised class.

The saved parsed catalogue has 204 valid M/X events in the study window: train 123 M + 10 X, validation 13 M + 2 X, test 55 M + 1 X. The excluded M2.0 candidate would add one possible test M event if authoritatively confirmed. No valid M/X interval overlaps another in the study range.

## Target-generation audit

The implemented forecast rule is: a positive is an M/X onset strictly after the row timestamp and at or before the horizon end; a negative requires a usable row and continuous observation through the full future horizon; active M/X rows and incomplete coverage are unknown (`-1`). The timestamp is the right edge of the completed minute, and peak labels map to the next completed minute (an exact minute-boundary peak belongs to the next bin).

An independent recomputation from `forecast_labeled_timeseries_minute_end_v2.parquet` and `goes_events_parsed.csv` produced **zero mismatches** for the stored active, class, peak, 15m, 30m, and 60m labels. This establishes that the stored targets are consistent with the included catalogue and code rules; it does not validate the catalogue’s scientific correctness or the excluded event.

| Horizon | Unknown (`-1`) | Negative (`0`) | Positive (`1`) |
|---|---:|---:|---:|
| 15 min | 18,393 | 277,387 | 2,300 |
| 30 min | 18,862 | 274,779 | 4,439 |
| 60 min | 19,773 | 270,067 | 8,240 |

Positive counts grow with longer horizons, consistent with the wider future-onset window. The saved event catalogue contains B/C rows, but C-class events are intentionally not forecast positives; a C flare is treated as background unless an M/X onset falls in the horizon. M/X overlaps are absent in the study period, and the labeler retains the higher class if overlaps occur.

## Leakage and preprocessing audit

- **Rolling windows:** all 592 feature definitions were reviewed. Rolling mean/std/count/min/max use pandas rolling windows with default `center=False`, grouped by continuous usable segments; each row uses only the current and preceding observations. Lags use `grouped.diff(lag)`, which compares the current value to past values. No future/centered rolling feature was found.
- **Current-time inputs:** signal values, per-channel missing/GTI flags, and UTC hour/day-of-year cycles are features. The minute-end signal bin is the completed interval ending at the prediction timestamp. `timestamp` is stored separately and is not in the model feature list.
- **Target exclusion:** `feature_schema.json` contains no `flare_*`, target, event class/intensity, start, peak, or end feature. The only target column in model-ready files is the separate `target` column. Forecast future-event lookup is used only to create labels, not features.
- **Scaling/imputation:** preprocessing in the builder does not fit a global scaler. Baseline model code uses scikit-learn pipelines, so imputer/scaler fitting occurs inside `.fit(X_train, y_train)`; the forecasting XGBoost path uses native NaNs without learned scaling; the deep-learning code explicitly estimates medians, means, and standard deviations from `train_x` and applies those values unchanged to validation/test. Thus the inspected current code follows train-only fitting. No such transform was run or verified against this run in this audit; the resulting parameters must be saved with any new model.
- **Split overlap:** every one of the 18 model-ready Parquets has sorted, unique timestamps, no row-policy violations, and valid target values. Timestamp sets are disjoint across train/validation/test. Forecast train and validation rows are purged by the horizon so their target windows do not cross the next boundary. There are no valid M/X intervals crossing either boundary and no M/X start within 60 minutes of either boundary. No same valid GOES event is split across periods. Rolling history can include observations from the preceding period at a split edge, which is operationally reasonable for a continuing time series but should be recorded when interpreting strict independence.
- **Existing model artifacts:** `output/model_results/` already contains baseline, forecasting, event-level, and CNN-LSTM model outputs. The saved CNN-LSTM metrics declare 424 features and 1,958 positive 60m test rows, whereas this run has 592 features and 2,695 positive test rows. These artifacts are therefore from a different/unlinked dataset or pipeline state and cannot be treated as results for `preproc_20261124_v1`. The statement that model training has never been exercised is not true for the workspace as a whole; training provenance for those artifacts is missing.

## Chronological splits and class support

The configured strategy was preserved: training represented dates before 1 June; validation covers June; test covers July. Minute-end rows are assigned by the date of the minute they represent. The adjacent split timestamp ranges do not overlap.

| Task / split | Rows | Target distribution |
|---|---:|---|
| classification / train | 183,280 | 0=181,036; M=1,974; X=270 |
| classification / validation | 36,896 | 0=36,463; M=391; X=42 |
| classification / test | 41,047 | 0=39,694; M=1,335; X=18 |
| nowcasting / train | 183,280 | 0=181,036; active=2,244 |
| nowcasting / validation | 36,896 | 0=36,463; active=433 |
| nowcasting / test | 41,047 | 0=39,694; active=1,353 |
| peak_detection / train | 183,280 | 0=183,196; peak=84 |
| peak_detection / validation | 36,896 | 0=36,881; peak=15 |
| peak_detection / test | 41,047 | 0=40,991; peak=56 |
| forecast_15m / train | 180,758 | 0=179,663; 1=1,095 |
| forecast_15m / validation | 36,417 | 0=36,192; 1=225 |
| forecast_15m / test | 39,621 | 0=38,847; 1=774 |
| forecast_30m / train | 180,469 | 0=178,388; 1=2,081 |
| forecast_30m / validation | 36,372 | 0=35,926; 1=446 |
| forecast_30m / test | 39,546 | 0=38,043; 1=1,503 |
| forecast_60m / train | 179,898 | 0=176,036; 1=3,862 |
| forecast_60m / validation | 36,282 | 0=35,392; 1=890 |
| forecast_60m / test | 39,397 | 0=36,702; 1=2,695 |

The test split has 55 valid M events and only 1 X event (18 X-active minute rows). That is not statistically meaningful support for X-specific performance. Forecast positive rows are not independent events; several rows may refer to the same onset. Peak detection has only 15 positive validation rows, so validation threshold estimates will be noisy. Accuracy is not an adequate headline metric for these imbalanced targets.

## Known limitations

1. The five missing dates are not reconciled to source availability versus acquisition failure; the root acquisition manifest is empty and there are no remote download logs.
2. One candidate M2.0 event is excluded with a plausible but unverified peak-time typo; labels around the event are potentially incomplete, especially in the July test period.
3. Test X support is one event, insufficient for X-specific generalization or significance claims.
4. The study task is explicitly M/X. C flares are not a separate class and form part of class 0/background.
5. The run config’s `both_instruments` row policy is applied consistently, but a formal date-level coverage threshold and approved date inclusion policy are absent.
6. Instrument time conventions and minute alignment were not independently confirmed against authoritative mission documentation in this local audit.
7. Raw extraction/scientific reports were reused via `--skip-extract`; the 23 outputs were audited, but raw FITS were not reprocessed here.
8. Existing model results are unlinked and use a different feature count/data distribution. No result should be attributed to this run without a matching manifest.
9. The in-run SHA manifest covers 23 outputs but not all auxiliary reports, logs, and checkpoints.

## Recommended next steps

1. Obtain authoritative confirmation/correction for the 3 July M2.0 record. If corrected, create a **new versioned run** and regenerate labels, forecasts, splits, and hashes; do not edit this run in place.
2. Ask the data provider or authenticated selector export whether archives were available for all five gaps. Record source URLs/query, retrieval attempts, HTTP outcomes, remote checksums, and per-instrument status in a non-secret acquisition manifest.
3. Approve and encode date-level eligibility/coverage thresholds; keep missing intervals unknown, never interpolate observations, and report usable exposure by date.
4. Confirm HEL1OS/SoLEXS time reference and GOES timestamp convention/rounding with authoritative source documentation, including the minute-end interval mapping.
5. Start only exploratory training with the preserved train split; fit every learned transform on train, tune thresholds on validation only, and keep test untouched. Store run ID, feature list/hash, preprocessing parameters, package versions, and sample/time split IDs with models.
6. Do not report X-specific test performance; collect more independent X events or explicitly label that analysis as descriptive only. For peak detection, use event-level metrics and uncertainty intervals.
7. Reconcile or archive the old `output/model_results/` artifacts with their own source dataset and code-version manifests before citing them.

## Final readiness statement

The 23 files in `preproc_20261124_v1` are internally readable and structurally consistent, and no future-feature leakage was found in the current feature schema/code. This supports **limited exploratory model training**. It does **not** establish a final scientific dataset or reliable X-class evaluation. The unresolved M2.0 row and unexplained acquisition gaps must remain visible in any use of the July test period.

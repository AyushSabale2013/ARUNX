# ARUN-X --- End-to-End Data, Training, and Evaluation Procedure

**Project:** ARUN-X --- Automated Radiation Understanding Network for
X-ray Analysis\
**Purpose:** Solar-flare nowcasting and forecasting using Aditya-L1
X-ray observations, with GOES flare events used as labels/reference
events.\
**Document status:** Working technical procedure and experiment record\
**Last updated:** 12 November 2026

> **Important record-keeping note:** This document separates confirmed
> work and recorded settings from details that are not available in the
> conversation history. It does not invent shell commands, download
> parameters, preprocessing values, model hyperparameters, or results.
> Where an exact command or configuration was not preserved, that is
> stated explicitly. Keep this document updated when the original
> scripts/configuration are inspected.

------------------------------------------------------------------------

## 1. Project objective

The project aims to build machine-learning models that use solar X-ray
time-series data from Aditya-L1 to identify and forecast solar flares.

### 1.1 Input instruments

-   **SoLEXS (Solar Low Energy X-ray Spectrometer):** soft X-ray
    observations. The project description has used the approximate
    energy range **1--30 keV**.
-   **HEL1OS (High Energy L1 Orbiting X-ray Spectrometer):** hard X-ray
    observations. The project description has used the approximate
    energy range **8--150 keV**; individual inspected FITS products
    expose specific energy bands.

The intended approach is to align the two instruments' time series and
use their combined information.

### 1.2 Reference event catalogue

The project uses GOES flare-event information as the
reference/ground-truth event catalogue. The event-level evaluation
described below specifically reports GOES **M- and X-class** events.

GOES event labels are not a replacement for the Aditya-L1 input
measurements. The model input data are the Aditya-L1 instrument
observations; GOES is used for event timing/labels and evaluation.

### 1.3 Prediction horizons

The current experiments cover three forecast horizons:

-   **15 minutes**
-   **30 minutes**
-   **60 minutes**

The prediction target is a binary target for each horizon. The recorded
convention is minute-end indexing: a timestamp such as `00:01`
represents the interval `[00:00, 00:01)`. For a timestamp `t` and
horizon `H`, the target is positive when a qualifying flare onset occurs
in the next `H` minutes.

------------------------------------------------------------------------

## 2. Data acquisition

### 2.1 Data portal and source

Aditya-L1 data were downloaded through the **PRADAN portal**. The
project used bulk downloads of SoLEXS and HEL1OS data.

The download process involved portal session/cookie considerations.
Resume behaviour and session validity were discussed because the files
were large. Do not assume that every interrupted download resumed
successfully unless the downloaded archive was checked.

### 2.2 Initial test-day download and FITS inspection

A concrete inspection run was performed for **5 November 2026**. The
user reported that 4 November data were unavailable for the required
download, so 5 November data were used for both HEL1OS and SoLEXS in
that test.

The working test folder was created as:

``` text
aditya_test/
├── raw_solexs/
├── raw_helios/
├── goes/
└── output/
```

The folder was initially named `pradhan_test` and later renamed to
`aditya_test`.

The downloaded archives were extracted into the corresponding raw-data
folders. The FITS inspection script was named:

``` text
inspect_lightcurves.py
```

It used:

-   `astropy.io.fits`
-   `astropy.time.Time`
-   `astropy.units`

An initial `ModuleNotFoundError` for `matplotlib` and the
externally-managed Python environment restriction led to creating a
virtual environment and installing the required packages there. The FITS
inspection script subsequently ran successfully.

### 2.3 HEL1OS FITS inspection details recorded

One inspected file was:

``` text
raw_helios/2026/11/05/HLS_20261105_000005_43185sec_lev1_V111/cdte/lightcurve_cdte1.fits
```

A corresponding `lightcurve_cdte2.fits` file was also inspected.

The FITS files contain multiple HDUs. The inspected light-curve HDUs
included energy bands such as:

-   5--20 keV
-   20--30 keV
-   30--40 keV
-   40--60 keV
-   1.80--90.00 keV

The inspected table columns included:

-   `MJD`
-   `ISOT`
-   `CTR`
-   `STAT_ERR`

The recorded interpretation was:

-   `MJD` / `ISOT`: time information
-   `CTR`: measured count-rate signal
-   `STAT_ERR`: statistical error associated with the signal

For the inspected HEL1OS CDTE1 5--20 keV HDU:

-   Rows: **43,169**
-   First UTC time: **2026-11-05 00:00:06.027850**
-   Last UTC time: **2026-11-05 11:59:34.027850**

Other inspected bands had approximately 43,000 rows each. Exact counts
differ by HDU and detector.

**Interpretation:** An HDU is a Header/Data Unit inside a FITS file. HDU
0 is commonly the primary header/data unit; subsequent HDUs can contain
separate tables or data products. The inspected files use multiple HDUs
for different light-curve products/energy bands.

### 2.4 Bulk dataset used for model development

The later ML-ready dataset used for model experiments is located at:

``` text
output/ml_ready_minute_end_v2/
```

Its reported size was approximately **2.6 GB**.

The exact list of downloaded archive filenames, full date coverage,
portal query parameters, checksums, and extraction commands were not
preserved in the available project record. Recover these from the
download folder, shell history, PRADAN download records, or data
manifest before publishing a fully reproducible data-acquisition
appendix.

### 2.5 Data integrity checks to retain

For every bulk download, retain:

1.  Original archive filename and source portal.
2.  Instrument (SoLEXS or HEL1OS).
3.  Observation date/time coverage.
4.  Download timestamp and archive size.
5.  Checksum (for example, SHA-256).
6.  Extraction destination.
7.  FITS filename and HDU inventory.
8.  Any missing dates or failed downloads.

Do not silently substitute dates or assume complete temporal coverage.
The 5 November 2026 test-day substitution is a documented example of a
date not being available.

------------------------------------------------------------------------

## 3. Project directory and important artifacts

The active project root is:

``` text
/home/ayush-sabale/Desktop/BTP/ARUNX
```

The active Python virtual environment is:

``` text
/home/ayush-sabale/Desktop/BTP/ARUNX/venv
```

The correct Python interpreter used for the project is:

``` text
/home/ayush-sabale/Desktop/BTP/ARUNX/venv/bin/python
```

Important paths recorded in the project:

``` text
ARUNX/
├── venv/
├── scripts/
│   └── ml/
│       ├── evaluate_event_level.py
│       └── build_master_model_evaluation.py
└── output/
    ├── ml_ready_minute_end_v2/
    ├── goes_events_parsed.csv
    └── model_results/
        ├── baselines/
        │   ├── all_validation_metrics.csv
        │   ├── all_test_metrics.csv
        │   └── individual model prediction CSVs
        ├── forecasting/
        │   ├── all_validation_metrics.csv
        │   ├── all_test_metrics.csv
        │   └── xgboost prediction CSVs
        ├── deep_learning/
        │   ├── cnn/
        │   ├── cnn_avgmax/
        │   └── cnn_lstm/
        ├── comparisons/
        │   ├── all_models_validation_metrics.csv
        │   └── all_models_test_metrics.csv
        ├── event_level/
        │   └── event_level_summary.csv
        └── ARUNX_MASTER_MODEL_EVALUATION.csv
```

This is a logical inventory of known paths, not a complete listing of
every project file.

------------------------------------------------------------------------

## 4. Data preparation and feature construction

### 4.1 ML-ready minute-end dataset

The model training input is the prepared dataset:

``` text
output/ml_ready_minute_end_v2/
```

The dataset has:

-   A timestamp field.
-   A binary target field.
-   **424 features**.
-   A reported total size of approximately **2.6 GB**.

The precise filenames/partitioning inside this directory were not
recorded in the available notes.

### 4.2 Time alignment and sampling

The project uses a minute-level time convention. The two instrument
streams need a common time axis for fusion. The recorded
feature-generation rules require time continuity: rolling and lag
features reset across gaps or unusable segments and require consecutive
one-minute samples.

The exact resampling method, aggregation rule, tolerance for matching
timestamps, and handling of duplicate timestamps must be documented from
the actual preprocessing script before claiming a specific
implementation.

### 4.3 Recorded feature groups

The 424-feature input includes the following documented feature
families:

1.  Signal logarithms.
2.  Missing-value indicators.
3.  Good Time Interval (GTI) information.
4.  HEL1OS error-related features.
5.  Rolling-window statistics using windows of:
    -   5 minutes
    -   15 minutes
    -   30 minutes
    -   60 minutes
6.  Lag/difference features using lags of:
    -   1 minute
    -   5 minutes
    -   15 minutes
7.  Time-of-day cyclic features using sine and cosine.

The recorded design also discussed a **Hard-to-Soft Ratio (HSR)** as a
useful fusion feature. Confirm from the final feature-name list whether
it is present in the 424 features before stating that it was included in
the final training matrix.

### 4.4 Continuity and leakage safeguards

The documented feature-generation rules state:

-   Rolling and lag features reset across data gaps or unusable
    segments.
-   Rolling/lag features require consecutive one-minute samples.
-   The target at minute-end time `t` refers to flare onset in the
    future horizon, not the past.
-   Training/validation/test splits are chronological, not random.

These rules are important for avoiding accidental use of future
observations in predictors. The exact source code should be retained as
the authoritative implementation.

### 4.5 Preprocessing details that need to be recovered from code

The following values were not preserved in the available record and must
not be guessed:

-   Exact missing-value imputation method and fill values.
-   Exact outlier/spike handling rules.
-   Exact logarithm formula and offset, if any.
-   Exact scaler type and fitted parameters.
-   Whether scaling was fitted on training data only.
-   Exact SoLEXS/HEL1OS alignment and resampling implementation.
-   Exact target-generation code and flare-class qualification rule.
-   Exact list and order of the 424 feature names.

The deep-learning run folders do contain `preprocessing.npz` and
`feature_names.json`, which should be retained alongside model
checkpoints.

------------------------------------------------------------------------

## 5. Target definition and class imbalance

### 5.1 Forecast targets

Three separate binary targets were trained:

-   `15m`: flare onset within the next 15 minutes.
-   `30m`: flare onset within the next 30 minutes.
-   `60m`: flare onset within the next 60 minutes.

The target is defined at each minute-end timestamp. The recorded
convention is:

``` text
timestamp 00:01 represents interval [00:00, 00:01)
```

A positive label means a qualifying flare onset occurs in the future
horizon. The precise flare-class inclusion criteria for row-level target
generation should be confirmed from the label-generation code.
Event-level evaluation later uses GOES M/X events.

### 5.2 Class imbalance

The forecasting task is highly imbalanced. One recorded training split
for the 15-minute task had:

-   198,111 cleaned training rows
-   1,299 positive rows
-   196,812 negative rows
-   Positive prevalence: approximately 0.656%
-   Negative-to-positive ratio (`scale_pos_weight` / positive weighting
    basis): approximately 151.51

The deep-learning sequence dataset has fewer samples because sequences
require a complete 60-minute history and continuity.

For the CNN 15m run, the sequence counts were:

  Split          Sequences   Positive sequences   Negative sequences
  ------------ ----------- -------------------- --------------------
  Train            192,190                1,037              191,153
  Validation        38,696                  225               38,471
  Test              38,761                  542               38,219

The CNN 15m training `pos_weight` was approximately **184.33**,
calculated from the sequence training class distribution.

Do not confuse cleaned row counts with sequence counts. They represent
different stages of the data pipeline.

------------------------------------------------------------------------

## 6. Chronological dataset splitting

The project uses chronological partitions:

  -----------------------------------------------------------------------
  Split                   Date range              Role
  ----------------------- ----------------------- -----------------------
  Training                1 January--31 May 2026  Fit model parameters

  Validation              1 June--30 June 2026    Model selection, early
                                                  stopping, threshold
                                                  selection

  Test                    July 2026               Exploratory
                                                  final-period evaluation
  -----------------------------------------------------------------------

The split is time-based. It is not a random train/test split.

### 6.1 Training split

The model learns its parameters from January through May 2026.

### 6.2 Validation split

June 2026 is used for model selection and validation. Deep-learning
early stopping and the selected classification threshold are based on
validation data.

### 6.3 Test split

July 2026 is used for test metrics. However, July was already used in
earlier experiments and comparisons. Therefore, it is an **exploratory
benchmark**, not a pristine untouched holdout set. Do not describe it as
an untouched independent final test set.

### 6.4 Example row counts

For the 15-minute task, one recorded cleaned-row split was:

  Split             Rows   Positive rows   Positive rate
  ------------ --------- --------------- ---------------
  Train          198,111           1,299          0.656%
  Validation      39,817             225          0.565%
  Test            41,729             776          1.860%

For the CNN 15m run, after constructing valid 60-minute sequences:

  Split          Sequences
  ------------ -----------
  Train            192,190
  Validation        38,696
  Test              38,761

These counts illustrate why sequence counts differ from cleaned-row
counts. Other horizons can have different positive counts because their
forecast targets differ.

### 6.5 Leakage rules

-   Do not randomly shuffle observations across chronological splits.
-   Do not fit preprocessing/scaling parameters using validation or test
    data.
-   Do not use July test metrics to tune hyperparameters, select epochs,
    or choose thresholds.
-   Select thresholds using validation predictions only.
-   Preserve the existing split boundaries unless a documented new
    experiment explicitly changes them.

------------------------------------------------------------------------

## 7. Model training inventory

The current experiment contains **9 model families × 3 horizons = 27
model-horizon runs**.

### 7.1 Traditional machine-learning baselines

Five baseline families were trained for 15m, 30m, and 60m:

1.  Logistic Regression
2.  Random Forest
3.  Extra Trees
4.  Hist Gradient Boosting
5.  LightGBM

Total baseline runs: **15**.

Prediction files:

``` text
output/model_results/baselines/{model}_{15,30,60}m_predictions.csv
```

Aggregate metric files:

``` text
output/model_results/baselines/all_validation_metrics.csv
output/model_results/baselines/all_test_metrics.csv
```

### 7.2 XGBoost

XGBoost was trained for all three horizons.

Total XGBoost runs: **3**.

Prediction files:

``` text
output/model_results/forecasting/xgboost_15m_predictions.csv
output/model_results/forecasting/xgboost_30m_predictions.csv
output/model_results/forecasting/xgboost_60m_predictions.csv
```

Recorded best iterations:

  Horizon     Best iteration
  --------- ----------------
  15m                      9
  30m                     18
  60m                    104

Recorded validation PR-AUC:

  Horizon     Validation PR-AUC
  --------- -------------------
  15m                    0.0166
  30m                    0.0231
  60m                    0.0317

These values are experiment records, not universal performance claims.
The exact XGBoost hyperparameters and early-stopping configuration
should be recovered from the training script/configuration.

### 7.3 Deep-learning architectures

Three architectures were trained for each horizon:

1.  CNN
2.  CNN AvgMax
3.  CNN-LSTM

Total deep-learning runs: **9**.

Each run folder contains artifacts such as:

-   `training_history.csv`
-   `metrics.json`
-   `best_model.pt`
-   `experiment_config.json`
-   `feature_names.json`
-   `preprocessing.npz`
-   `forecast_{15,30,60}m_predictions.csv`

Directory layout:

``` text
output/model_results/deep_learning/
├── cnn/
│   ├── forecast_15m/
│   ├── forecast_30m/
│   └── forecast_60m/
├── cnn_avgmax/
│   ├── forecast_15m/
│   ├── forecast_30m/
│   └── forecast_60m/
└── cnn_lstm/
    ├── forecast_15m/
    ├── forecast_30m/
    └── forecast_60m/
```

The deep-learning models used:

-   Sequence length: **60**
-   Feature count: **424**
-   Device recorded in metrics: **CUDA**
-   Weighted binary cross-entropy training loss using a positive-class
    weight.
-   Validation loss recorded as unweighted BCE from logits.

The weighted training loss and unweighted validation loss are not
directly comparable because they use different weighting.

Recorded system context:

-   CPU: Intel Core i5-13450HX
-   GPU: NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM
-   System RAM: 8 GB
-   Python: 3.12.3
-   Project virtual environment: `venv`

Sequential training and limiting CPU threads were preferred to avoid
excessive memory pressure. The exact thread setting and batch sizes
should be taken from each `experiment_config.json`.

### 7.4 Deep-learning best-epoch records

  --------------------------------------------------------------------------
  Architecture          Horizon     Best epoch     Epochs run           Best
                                                                  validation
                                                                      PR-AUC
  -------------- -------------- -------------- -------------- --------------
  CNN                       15m              8             14       0.008529

  CNN                       30m              1              7       0.016554

  CNN                       60m              4             10       0.028829

  CNN AvgMax                15m              1              7       0.009546

  CNN AvgMax                30m              1              7       0.015743

  CNN AvgMax                60m              1              7       0.025699

  CNN-LSTM                  15m             11             17       0.015604

  CNN-LSTM                  30m              1              7       0.018721

  CNN-LSTM                  60m              1              7       0.031164
  --------------------------------------------------------------------------

The authoritative per-run records are each run's `metrics.json` and
`experiment_config.json`.

------------------------------------------------------------------------

## 8. Prediction files and sample-level evaluation

Each trained model produced prediction files. The comparison stage
consolidated model metrics.

### 8.1 Comparison outputs

The consolidated comparison files are:

``` text
output/model_results/comparisons/all_models_validation_metrics.csv
output/model_results/comparisons/all_models_test_metrics.csv
```

The recorded comparison files contain 18 rows each:

-   5 baseline families × 3 horizons = 15
-   XGBoost × 3 horizons = 3

They do not include the 9 deep-learning rows. Deep-learning metrics are
read from each architecture/horizon `metrics.json`.

### 8.2 Metrics recorded

The comparison/evaluation records include:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   PR-AUC
-   ROC-AUC
-   True negatives (TN)
-   False positives (FP)
-   False negatives (FN)
-   True positives (TP)
-   Sample count
-   Positive sample count
-   Specificity
-   False-positive rate
-   False alarms per day
-   Observed/evaluated days
-   Prevalence
-   Threshold
-   Training time, where available

For imbalanced flare forecasting, PR-AUC, precision, recall, F1, and
false alarms are important. Accuracy alone can be misleading because
negative samples dominate.

### 8.3 Threshold selection

The deep-learning `metrics.json` files record a validation-selected
threshold. The same threshold is then applied to the test predictions.

Do not choose a threshold from the July test set and report that as an
unbiased test result. Threshold selection should be done using
validation data.

------------------------------------------------------------------------

## 9. Event-level evaluation

Sample-level classification asks whether individual minute samples are
positive. Event-level evaluation asks whether the model generated an
alert episode associated with a real GOES M/X event.

The event-level evaluator used was:

``` text
scripts/ml/evaluate_event_level.py
```

The command executed was:

``` bash
python scripts/ml/evaluate_event_level.py \
  --pred-dir output/model_results \
  --goes-file output/goes_events_parsed.csv \
  --output-dir output/model_results/event_level
```

Recorded evaluator settings/results:

-   GOES M/X catalogue events: **231**
-   Prediction files found: **27**
-   Alert-episode gap: **1 minute**
-   July prediction period contained **56 GOES M/X events** in the event
    summary.

### 9.1 Event-level outputs

The summary file is:

``` text
output/model_results/event_level/event_level_summary.csv
```

The summary contains 27 rows, one per prediction file/horizon.

The event-level fields include:

-   GOES M/X events in period
-   GOES events in association window
-   Detected events
-   Missed events
-   Event detection rate
-   Alert episode precision
-   Alert episodes
-   Episodes associated with events
-   False-alarm episodes
-   Redundant associated episodes
-   False alarms per day
-   Mean, median, minimum, and maximum lead time
-   Row-level TP, FP, and FN
-   Observed coverage and missing timestamp minutes

### 9.2 Event-level evaluator limitation

The event-level evaluator currently has a model-name collision for
deep-learning prediction files. It labels all nine deep-learning runs
with the generic name:

``` text
forecast
```

Consequences:

1.  Deep-learning event metrics cannot safely be joined to `cnn`,
    `cnn_avgmax`, or `cnn_lstm`.
2.  Detailed event-match output filenames can collide/overwrite because
    the generic model prefix is reused.
3.  The current event evaluator counts M/X events in the period even
    when the prediction series has gaps or does not provide full
    preceding horizon coverage.

The current event-level results are therefore **preliminary**. The
coverage issue was identified, but the decision at the time was to leave
the evaluator unchanged and inspect the current output first. Do not
describe these event metrics as fully validated.

### 9.3 Example event-match inspection

The following file was inspected:

``` text
output/model_results/event_level/random_forest_15m_event_matches.csv
```

It contained 41 matched events.

Recorded class distribution:

-   M-class: 40
-   X-class: 1

Lead-time statistics:

-   Mean: 13.68 minutes
-   Median: 15 minutes
-   Exactly 15-minute lead: 33 of 41 detected events (about 80.5%)

This is a small, specific sample. The one detected X-class event is not
enough to claim reliable X-class performance. The concentration of
15-minute lead times also needs to be interpreted in light of the
matching rule: it selects the first qualifying alert in the forecast
window. It is not, by itself, proof that the model predicted every flare
exactly 15 minutes in advance.

------------------------------------------------------------------------

## 10. Master model evaluation CSV

The central model-evaluation record was created at:

``` text
output/model_results/ARUNX_MASTER_MODEL_EVALUATION.csv
```

The build script is:

``` text
scripts/ml/build_master_model_evaluation.py
```

The command executed was:

``` bash
python scripts/ml/build_master_model_evaluation.py
```

The script combines:

-   Classical model and XGBoost validation metrics from
    `all_models_validation_metrics.csv`.
-   Classical model and XGBoost test metrics from
    `all_models_test_metrics.csv`.
-   Deep-learning validation/test metrics from the nine `metrics.json`
    files.
-   Event-level metrics for uniquely named classical/XGBoost
    model-horizon rows.
-   Model metadata and source paths where available.

Deep-learning event-level fields are deliberately left blank because the
current event summary uses the ambiguous generic model name `forecast`.

### 10.1 Verified master-file structure

The user ran a verification script and confirmed:

``` text
Shape: (27, 69)
Duplicate model-horizon pairs: 0
```

Records per horizon:

  Horizon       Records
  ----------- ---------
  15m                 9
  30m                 9
  60m                 9
  **Total**      **27**

Family distribution:

  Family                    Records
  ----------------------- ---------
  baselines                      15
  forecasting (XGBoost)           3
  deep_learning                   9
  **Total**                  **27**

The master CSV was further checked to confirm:

-   Missing validation PR-AUC values: **0**
-   Missing test PR-AUC values: **0**

Some other missing values are expected because fields differ by model
type. For example, deep-learning-only fields such as `best_epoch` do not
apply to classical models. Event-level fields for deep-learning rows
remain blank pending the evaluator naming fix.

------------------------------------------------------------------------

## 11. Recorded sample-level test PR-AUC results

The following table is taken from the verified master CSV output.

  Model                        15m PR-AUC   30m PR-AUC     60m PR-AUC
  ------------------------ -------------- ------------ --------------
  Extra Trees                    0.063976     0.102744       0.199055
  Hist Gradient Boosting         0.065753     0.113837       0.240954
  LightGBM                       0.052383     0.085457       0.197876
  Logistic Regression            0.065112     0.122951       0.218312
  Random Forest              **0.075860**     0.120793   **0.250289**
  CNN                            0.053221     0.082741       0.127108
  CNN AvgMax                     0.035275     0.082399       0.169229
  CNN-LSTM                       0.032647     0.092383       0.177737
  XGBoost                        0.023912     0.071911       0.149778

Best sample-level test PR-AUC among the nine model families in the
current July benchmark:

  Horizon   Model                   Test PR-AUC
  --------- --------------------- -------------
  15m       Random Forest              0.075860
  30m       Logistic Regression        0.122951
  60m       Random Forest              0.250289

**Interpretation caution:** July has a higher positive prevalence than
June in the recorded 15-minute split. PR-AUC depends on class
prevalence. These July results are exploratory and should not be
presented as a final unbiased estimate of future operational
performance.

------------------------------------------------------------------------

## 12. Reproducibility: commands and checks currently confirmed

### 12.1 Inspect FITS files

The inspection script used was:

``` bash
python inspect_lightcurves.py
```

It used Astropy FITS/time functionality. The exact original invocation
path and full script content should be retained in the repository.

### 12.2 Run event-level evaluation

``` bash
python scripts/ml/evaluate_event_level.py \
  --pred-dir output/model_results \
  --goes-file output/goes_events_parsed.csv \
  --output-dir output/model_results/event_level
```

### 12.3 Build the master evaluation CSV

``` bash
python scripts/ml/build_master_model_evaluation.py
```

### 12.4 Verify the master CSV

``` bash
python - <<'PY'
import pandas as pd

path = "output/model_results/ARUNX_MASTER_MODEL_EVALUATION.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("Duplicate model-horizon pairs:",
      df.duplicated(["model", "horizon"]).sum())

print(df[["family", "model", "horizon"]].to_string(index=False))

print("\nMissing PR-AUC values:")
print("Validation:", df["val_pr_auc"].isna().sum())
print("Test:", df["test_pr_auc"].isna().sum())
PY
```

Expected checks based on the completed run:

``` text
Shape: (27, 69)
Duplicate model-horizon pairs: 0
Validation missing PR-AUC: 0
Test missing PR-AUC: 0
```

------------------------------------------------------------------------

## 13. Important limitations and items to verify before a professor-facing final report

The current master CSV is a valid central experiment record, but it is
not by itself proof that the full scientific pipeline is completely
validated.

Before writing final claims, recover or verify:

1.  **Download manifest:** exact bulk archive names, download date,
    observation coverage, file sizes, checksums, and portal query
    details.
2.  **Data inventory:** exact SoLEXS and HEL1OS date coverage and
    missing dates.
3.  **Preprocessing source:** exact timestamp alignment, resampling,
    cleaning, imputation, scaling, and transformation rules.
4.  **Feature manifest:** exact 424 feature names and definitions.
5.  **Target-generation code:** exact qualifying flare classes,
    onset-time definition, and boundary handling.
6.  **Split implementation:** verify the date filters and whether any
    rows at split boundaries are excluded to prevent target-horizon
    leakage.
7.  **Training configurations:** exact hyperparameters, batch sizes,
    optimizer, learning rate, scheduler, early stopping, random seeds,
    and thread settings.
8.  **Threshold policy:** verify that all reported thresholds were
    selected on validation only.
9.  **Event-level naming:** fix the generic `forecast` architecture name
    and rerun event evaluation so all nine deep-learning architectures
    have correctly named outputs.
10. **Event coverage:** revise event eligibility to require adequate
    prediction coverage for the relevant horizon, then rerun event-level
    evaluation.
11. **Independent test:** because July has already been used in
    experiments, reserve a later untouched period or use a carefully
    designed time-based evaluation for a stronger final performance
    claim.

Do not overwrite existing results when fixing the event evaluator. Save
a new version of the event-level outputs or back up the current
preliminary results first.

------------------------------------------------------------------------

## 14. Recommended experiment record policy

For every new experiment, retain:

-   Dataset version and checksum/manifest.
-   Git commit or source-code version.
-   Configuration file.
-   Feature-name file.
-   Preprocessing/scaler artifacts.
-   Model checkpoint.
-   Training history.
-   Validation metrics.
-   Test metrics.
-   Prediction CSV.
-   Event-level summary and event-match files.
-   Date/time of run and software versions.

Use distinct architecture-specific output names. Avoid generic names
such as `forecast` when multiple architectures are evaluated.

Keep raw downloaded data immutable. Store prepared data in a separate
output directory. Do not silently replace existing predictions or
metrics.

------------------------------------------------------------------------

## 15. Current project status

  -----------------------------------------------------------------------
  Stage                               Current status
  ----------------------------------- -----------------------------------
  PRADAN Aditya-L1 data               Performed; exact bulk manifest
  access/download                     needs recovery

  FITS extraction and inspection      Performed for the documented 5 Nov
                                      2026 test data

  ML-ready minute-end dataset         Available at
                                      `output/ml_ready_minute_end_v2/`

  Feature set                         424 features recorded

  Forecast horizons                   15m, 30m, 60m

  Chronological splits                Jan--May train, June validation,
                                      July test

  Traditional baselines               5 families × 3 horizons trained

  XGBoost                             3 horizons trained

  Deep learning                       CNN, CNN AvgMax, CNN-LSTM × 3
                                      horizons trained

  Sample-level comparison             Consolidated and verified

  Event-level evaluation              Run; preliminary coverage and
                                      naming issues remain

  Master evaluation CSV               Created and structurally verified
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 16. Final artifact reference

**Central evaluation file:**

``` text
output/model_results/ARUNX_MASTER_MODEL_EVALUATION.csv
```

**Build script:**

``` text
scripts/ml/build_master_model_evaluation.py
```

**Event-level evaluator:**

``` text
scripts/ml/evaluate_event_level.py
```

The master CSV is the current single-file index of model/horizon
validation and test metrics. Use this document together with the actual
preprocessing/training scripts and per-run configuration files when
preparing a formal research report.

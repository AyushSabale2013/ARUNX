# Model training and evaluation plan

## Recommended starting point

Start with **CatBoost or LightGBM gradient-boosted trees** on engineered tabular features. The current data is one-minute instrument signals with quality flags, while the number of independent flare events is still limited. A tree model is a practical first serious model and is easier to audit than a deep sequence model at this stage.

Build features only from the current and earlier observations: recent signal levels, changes/slopes, and trailing summaries over windows such as 5, 15, and 60 minutes. Include missingness and GTI flags. Consider log-transforming nonnegative count signals, then fit any scaling/normalization parameters on training dates only. Keep the exact feature list and transform with the fitted model.

The recommendation is a proposed first experiment, not a measured winner. There are no finalized multi-day splits or trained results yet.

## Training sequence

1. **Climatology/no-skill baseline:** estimate event frequency on training data and establish expected performance under the natural prevalence.
2. **Logistic regression:** a simple, interpretable benchmark using training-fitted transforms.
3. **CatBoost/LightGBM:** class-weighted binary model, separately for each forecast horizon or with one clearly selected primary horizon.
4. **Temporal CNN or CNN-LSTM:** test only after more independent events and dates are available; use fixed-length past-only sequences.

If CatBoost/LightGBM is unavailable, use scikit-learn tree baselines such as `HistGradientBoostingClassifier` or `RandomForestClassifier`.

## Split and leakage rules

- Split on chronological date boundaries: earlier dates for training, later dates for validation, latest held-out dates for testing.
- Keep the test period untouched until model and threshold choices are fixed.
- Do not randomly split minute rows. Adjacent observations and overlapping sequence windows are highly related.
- Compute lagged and rolling features without looking beyond the prediction time. Do not use centered windows, future fills, labels, event class, or any GOES-derived target as input.
- Fit imputation, scaling, feature selection, and sampling only on training data. Validation and test retain natural class prevalence.
- Exclude forecast target values of `-1` from supervised loss and metrics.

## Imbalance and metrics

Forecast positives are expected to be rare. Use class weights or training-only negative sampling; never rebalance validation or test. Choose a probability threshold on validation data to reflect the operational trade-off between missed flares and false alarms.

Report at minimum:

- Event counts and positive prevalence for each split and horizon.
- Precision-recall AUC and the selected operating threshold.
- Event-level recall (did the model alert before each event?), precision, and false alarms per day.
- Lead time distribution for detected events.
- Confusion matrix at the chosen threshold, with per-date performance where sample size permits.

Accuracy alone is inadequate for rare events. Include uncertainty or caution when there are few independent test events. Compare each model on the same held-out dates and report seeds/configuration; never substitute illustrative numbers for measured results.

## Readiness gate

Begin model comparison only after multi-day processing is reproducible, labels have been audited against GOES, usable coverage and event counts are known, and chronological splits include enough independent events to make test results informative. Save model weights, preprocessing state, feature schema, target horizon, split dates, configuration, and evaluation outputs together so a result can be reproduced.

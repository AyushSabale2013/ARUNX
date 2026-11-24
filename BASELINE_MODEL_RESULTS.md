# Baseline Model Results

**Dataset run:** `preproc_20261124_v1`  
**Generated UTC:** 2026-11-24T14:24:03.577821+00:00  
**Model run directory:** `output/model_results/xgboost_baselines_preproc_20261124_v1`  
**Model:** XGBoost binary classifier, separately trained for each forecast horizon.

## Dataset and safeguards

- Dataset has 592 approved features and uses `both_instruments` rows.
- The six prepared task/target mappings are classification → `flare_class`, nowcasting → `flare_active`, peak detection → `flare_peak`, and forecasting → `flare_15m` / `flare_30m` / `flare_60m`. This training run uses the three forecast target Parquets only.
- The source feature order is loaded from `feature_schema.json`; each split is checked against the exact ordered columns `timestamp`, `target`, then the approved feature list.
- Model inputs exclude the `timestamp` and `target` columns and are exactly the 592 schema features. Learned global scaling is not used; XGBoost handles missing values natively.
- `scale_pos_weight` is computed only from training class counts. Early stopping uses validation PR-AUC; the decision threshold maximizes validation F1. Test file hashes are verified as byte-integrity metadata before fitting, but test feature/target rows are loaded only after model selection and final evaluation; they are then used only to verify serialized-model prediction reproducibility.
- Chronological boundaries are preserved from the preprocessing config: training before 2026-06-01, validation in June, test in July. Forecast horizon purging is already applied in the input files.
- Preprocessing validation status: `PASS_WITH_REVIEW`.

## Training configuration

```json
{
  "seed": 42,
  "shared_xgboost_parameters": {
    "colsample_bytree": 0.8,
    "device": "cpu",
    "early_stopping_rounds": 75,
    "eval_metric": "aucpr",
    "gamma": 0.0,
    "learning_rate": 0.04,
    "max_bin": 256,
    "max_depth": 5,
    "min_child_weight": 5,
    "n_estimators": 1500,
    "n_jobs": 4,
    "objective": "binary:logistic",
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "tree_method": "hist"
  },
  "class_weight": "negative_train_count / positive_train_count, recomputed per horizon"
}
```

## Dataset distributions and evaluation

All performance rows below are from the untouched test split after validation-based model selection. Precision, recall, and F1 use the validation-selected threshold. ROC-AUC and PR-AUC use continuous probability scores; PR-AUC is average precision.

| Horizon | Split | Rows | Negatives | Positives |
|---|---|---:|---:|---:|
| 15 min | train | 180,758 | 179,663 | 1,095 |
| 15 min | validation | 36,417 | 36,192 | 225 |
| 15 min | test | 39,621 | 38,847 | 774 |
| 30 min | train | 180,469 | 178,388 | 2,081 |
| 30 min | validation | 36,372 | 35,926 | 446 |
| 30 min | test | 39,546 | 38,043 | 1,503 |
| 60 min | train | 179,898 | 176,036 | 3,862 |
| 60 min | validation | 36,282 | 35,392 | 890 |
| 60 min | test | 39,397 | 36,702 | 2,695 |

### Test metrics compared with majority-class baseline

| Horizon | Best iteration (rounds) | Model threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Majority PR-AUC | Majority ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 min | 0 (1) | 0.510871 | 0.0586 | 0.6705 | 0.1077 | 0.7198 | 0.0601 | 0.2148 | 0.3295 | 0.0195 | 0.5000 |
| 30 min | 57 (58) | 0.230501 | 0.1494 | 0.3673 | 0.2124 | 0.7898 | 0.1081 | 0.0826 | 0.6327 | 0.0380 | 0.5000 |
| 60 min | 9 (10) | 0.429528 | 0.1892 | 0.6660 | 0.2947 | 0.7737 | 0.2066 | 0.2095 | 0.3340 | 0.0684 | 0.5000 |

Best iteration is zero-based; rounds shows the number of retained trees. The 15-minute model early-stopped after its first tree, so its validation-selected fit is especially simple and may warrant validation-only tuning in a later model run. The majority classifier predicts the majority class in training (class 0) for every test row. Its precision/recall/F1, confusion matrix, FPR/FNR, ROC-AUC, and PR-AUC are saved in each horizon’s `metrics.json`. Because it assigns a constant score, ROC-AUC is 0.5; its average precision equals the test positive prevalence.

### Confusion matrices

Rows are true labels `[0, 1]`; columns are predicted labels `[0, 1]`.

| Horizon | XGBoost `[[TN, FP], [FN, TP]]` | Majority baseline `[[TN, FP], [FN, TP]]` |
|---|---|---|
| 15 min | `[[30502, 8345], [255, 519]]` | `[[38847, 0], [774, 0]]` |
| 30 min | `[[34900, 3143], [951, 552]]` | `[[38043, 0], [1503, 0]]` |
| 60 min | `[[29012, 7690], [900, 1795]]` | `[[36702, 0], [2695, 0]]` |

### Feature importance

Gain-based importance is saved for all 592 features, including zero-importance features. Top ten features per horizon:

| Horizon | Rank | Feature | Gain | Split count |
|---|---:|---|---:|---:|
| 15 min | 1 | `hel1os_cdte1_band1_roll30_min` | 23906.3 | 19 |
| 15 min | 2 | `hel1os_cdte2_band5_roll30_mean` | 19550 | 9 |
| 15 min | 3 | `hel1os_cdte1_band5_roll60_mean` | 7285.52 | 51 |
| 15 min | 4 | `solexs_sdd2_counts_roll15_std` | 6253.93 | 1 |
| 15 min | 5 | `hel1os_cdte2_band1_roll30_min` | 5114.93 | 1 |
| 15 min | 6 | `hel1os_cdte1_band1_roll60_std` | 4882.77 | 28 |
| 15 min | 7 | `hel1os_czt1_band3_roll15_max` | 4611.21 | 3 |
| 15 min | 8 | `solexs_sdd2_counts_roll5_min` | 4378.2 | 6 |
| 15 min | 9 | `solexs_sdd2_counts_roll5_max` | 4062.54 | 12 |
| 15 min | 10 | `hel1os_cdte1_band5_roll60_min` | 4032.29 | 32 |
| 30 min | 1 | `hel1os_cdte2_band5_roll30_mean` | 17783.4 | 6 |
| 30 min | 2 | `hel1os_cdte1_band1_roll30_min` | 16513.5 | 18 |
| 30 min | 3 | `hel1os_cdte1_band5_roll30_mean` | 14868.4 | 28 |
| 30 min | 4 | `hel1os_cdte1_band5_roll60_min` | 3817.63 | 74 |
| 30 min | 5 | `hel1os_cdte2_band1_roll15_max` | 2743.19 | 6 |
| 30 min | 6 | `hel1os_czt1_band3_roll60_max` | 2393.5 | 49 |
| 30 min | 7 | `hel1os_cdte1_band3_roll30_std` | 2148.54 | 6 |
| 30 min | 8 | `hel1os_cdte1_band1_roll60_max` | 2132.79 | 73 |
| 30 min | 9 | `hel1os_czt1_band1_roll30_max` | 2127.41 | 23 |
| 30 min | 10 | `hel1os_czt1_band5_roll30_max` | 2087.44 | 10 |
| 60 min | 1 | `hel1os_cdte1_band5_roll30_mean` | 15338.4 | 22 |
| 60 min | 2 | `hel1os_cdte1_band5_roll30_max` | 9547.58 | 7 |
| 60 min | 3 | `hel1os_cdte1_band1_roll30_max` | 7791.28 | 4 |
| 60 min | 4 | `hel1os_cdte1_band5_roll60_mean` | 7271.51 | 8 |
| 60 min | 5 | `hel1os_cdte1_band1_roll30_min` | 6927.71 | 5 |
| 60 min | 6 | `hel1os_cdte1_band5_roll60_min` | 6904.33 | 56 |
| 60 min | 7 | `hel1os_cdte1_band5_log1p` | 5057.85 | 1 |
| 60 min | 8 | `hel1os_cdte1_band5_roll60_max` | 3897.75 | 18 |
| 60 min | 9 | `hel1os_cdte1_band1_roll15_std` | 3649.87 | 3 |
| 60 min | 10 | `hel1os_cdte1_band1_roll60_max` | 3528.06 | 29 |

### Precision-recall curves and artifacts

Test and validation precision-recall curve points are saved as CSV files per horizon. Test predictions are saved with UTC timestamps, actual labels, probabilities, and predicted classes. Each saved model was reloaded and produced probabilities matching the original in-memory model within absolute tolerance `1e-7`.

## Limitations

- The readiness report identifies five requested dates without both-instrument coverage and one malformed July 3 M2.0 GOES event that was excluded. The July test labels may therefore be incomplete near that event.
- There is only one valid X-class event in the test period. These binary forecast models predict M/X onset together; results do not establish X-specific skill.
- All three horizons share one July test period. This is one temporal holdout, not evidence of generalization across multiple solar cycles or independent test periods.
- Minute-level positives are clustered around individual events; event-level recall, false alarms per day, and lead-time evaluation remain necessary before scientific claims.
- XGBoost gains are descriptive, not causal. Feature importance should be checked for stability across additional chronological periods.
- Existing artifacts under `output/model_results/` use older 424-feature datasets and are not included in these results.

## Recommended next experiments

1. Resolve the excluded GOES event from an authoritative catalogue and rebuild a new versioned dataset before final July test reporting.
2. Verify archive availability for the five missing dates and preserve the download/source provenance.
3. Add a second or rolling-origin temporal evaluation period while keeping the current July test results as a single holdout reference.
4. Report event-level detection, false alarms per day, lead time, and bootstrap uncertainty by event; do not claim X-specific performance from one X event.
5. Compare class weighting against unweighted XGBoost and a simple regularized logistic model using the same chronological splits and validation-only selection.

## Reproducibility and artifact mapping

- Dataset run ID: `preproc_20261124_v1`.
- Feature schema SHA-256: `4876d7c2e6efdabeaea1d4dd15ffdc0326d42bcbc16dc52c9caf323667b0d242`.
- Each horizon’s `training_metadata.json` contains train/validation/test file hashes, class counts, time ranges, and the dataset run ID.
- Each horizon directory contains the XGBoost JSON model, model config, ordered feature names, metrics and confusion matrix, PR curves, test predictions, and complete feature importance.
- To repeat training without overwriting these outputs, choose a new output directory and report path, for example: `venv\Scripts\python.exe scripts\ml\train_xgboost_baselines.py --dataset-run preproc_20261124_v1 --output-dir output\model_results\xgboost_baselines_preproc_20261124_v1_rerun --report BASELINE_MODEL_RESULTS_rerun.md --seed 42`.
- `artifact_manifest.json` records SHA-256 hashes for every model-run artifact and this report; as a self-referential manifest, it does not list its own hash.

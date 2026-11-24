# ARUN-X Git collaboration and data policy

This repository is one shared codebase for the preprocessing and modeling work. Use separate topic branches and pull requests; keep large scientific inputs and run products on local or approved shared storage.

## What belongs in Git

Commit source and reproducibility material:

- Python preprocessing, training, evaluation, and inspection scripts.
- Tests and workflow/build files, including future `.github/workflows/`, Docker, and project configuration.
- `config/` templates and safe configuration examples. Do not put credentials in these files.
- Requirements and environment setup instructions.
- README and technical documentation, including the preprocessing PDF.
- Feature schemas, JSON schemas, small dataset manifests, source run IDs, hashes, extraction inventories, and small QA/status reports.
- Small sample fixtures only when intentionally checked into `tests/fixtures/` and clearly identified as redistributable test data.

The ignore rules target generated data and model output binaries by path or extension. They do not ignore JSON/CSV/Markdown globally, so schemas and small provenance records remain visible to Git.

## What stays local

Do not commit:

- Raw HEL1OS/SoLEXS archives, FITS lightcurves, GTI sidecars, extracted observations, or bulk GOES downloads.
- Merged, labeled, feature, split, or canonical Parquet tables and other generated matrix binaries.
- Model weights/checkpoints/serialized estimators, experiment-run directories, predictions, and large evaluation output.
- Local logs, caches, temporary extraction folders, virtual environments, secrets, portal cookies, and credentials.

These artifacts are reproducible or too large/sensitive for ordinary Git history. The `.gitignore` prevents new untracked files of these types from being added by accident; it does not remove anything already tracked.

## Sharing raw observations

Use the authorized PRADAN/ISSDC portal to retrieve source products. Each developer should configure their own portal session locally and never share or commit session cookies, passwords, bearer tokens, or authenticated URLs. Store archives under the repository’s ignored local download tree (`downloaders/pradan1.issdc.gov.in/al1/protected/downloadData/`) or in the configured archive cache. Store extracted HEL1OS and SoLEXS products in the ignored `raw_hel1os/` and `raw_solexs/` trees.

For team transfer, use an organization-approved storage location with access controls. Share a run ID and its small manifest, which should contain source filenames, dates, archive SHA-256 hashes, extraction statuses, config hash, and schema hash. A manifest identifies the expected inputs; the binary observations must be transferred separately. Do not invent a URL or share another developer’s authenticated portal session.

## Canonical master dataset

Current canonical dataset run: `preproc_20261124_v1`.

- Local Parquet: `output/master_datasets/preproc_20261124_v1/canonical_master_features_targets.parquet` (ignored; 298,080 rows × 705 columns, 592 features and six targets).
- Tracked schema: `output/master_datasets/preproc_20261124_v1/canonical_master_schema.json`.
- Tracked hashes and lineage: `output/master_datasets/preproc_20261124_v1/manifest.json`.
- Source run: `output/preprocessing_runs/preproc_20261124_v1/`; its run manifest, configuration, feature schema, and reports are small reproducibility artifacts. Parquet products remain ignored.
- Generator: `scripts/pipeline/build_canonical_master.py`.

After obtaining the exact source run’s local data, regenerate the canonical master from the repository root:

```powershell
python -m venv venv  # once per clone, if no project environment exists
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-baseline-ml.txt
python scripts\pipeline\build_canonical_master.py
```

The builder refuses to replace a populated version directory and checks the generated feature names/order against the frozen schema. Compare `manifest.json` hashes and date coverage with the committed values. Byte-for-byte identity requires the same source tables and dependency/runtime behavior; if any input, transformation, or schema changes, create a new run ID and a new master directory rather than overwriting this version.

The current `requirements-baseline-ml.txt` is a pinned baseline-training environment, not a complete pinned dependency list for archive acquisition, FITS extraction, and the entire preprocessing pipeline. The repository does not yet contain a dedicated locked preprocessing environment file; add one before relying on fresh-clone end-to-end regeneration.

To recreate source preprocessing products from a fresh clone, first obtain the exact archive set and GOES catalogue, then use the configured pipeline and a new run ID:

```powershell
python scripts\pipeline\run_preprocessing.py --config config\preprocessing.json --run-id <new_run_id>
```

Use `--skip-extract` only when the expected extracted FITS products already exist and have been verified. The current run has five missing dates and its remote authenticated download path was not tested; the existing manifest cannot determine whether those products were unavailable or never retrieved. Consequently, the exact master must be copied from approved shared storage or regenerated after the complete source inputs are made available. Do not interpolate missing observations.

## Model artifacts and experiments

Keep model code, configurations, dependency pins, and reusable evaluation logic in Git. Keep trained weights, optimizer state, checkpoints, fold outputs, prediction arrays, and experiment logs under ignored local directories such as `output/model_results/`, `output/models/`, `output/experiments/`, or an approved external experiment store.

For results shared in a pull request, include a concise Markdown summary with dataset run ID, feature-schema hash, split period, commands/configuration, metrics, and limitations. Do not attach large per-row prediction exports or weights. Keep lightweight, non-sensitive metadata only when the team needs it for reproducibility; do not put secrets, local paths, or authenticated URLs in metadata.

## Two-developer workflow

1. Start from an up-to-date `main` branch and create a focused branch, for example `preprocess/add-gti-checks` or `model/xgb-threshold-study`.
2. Keep preprocessing and modeling changes in the same repository. Coordinate shared interfaces through committed schemas/configuration and stable run IDs.
3. Make code changes and add/update focused tests or validation checks. Keep local datasets and model artifacts outside commits.
4. Before staging, review `git status --short` and `git diff --cached --stat`; explicitly stage intended source/docs rather than `git add .` when working around large local data.
5. Open a pull request to `main`, describe the run/schema impact, and request review from the other developer. Resolve conflicts together; do not force-push shared branches or rewrite shared history.
6. Merge only after review and the relevant checks pass. Pull the merged `main` before starting the next task branch.

## Ignore-policy summary

| Category | Git policy |
|---|---|
| Source code | Track |
| Preprocessing scripts | Track |
| Training and evaluation scripts | Track |
| Documentation and PDF | Track |
| Feature schemas and configuration templates | Track |
| Small manifests, hashes, and QA reports | Track |
| Raw FITS and ZIP data | Ignore |
| Intermediate datasets and generated Parquet | Ignore |
| Canonical master binaries | Ignore; schema and manifest stay tracked |
| Model weights and checkpoints | Ignore |
| Predictions and experiment/evaluation outputs | Ignore |
| Secrets, credentials, cookies, and local environments | Ignore |

## Existing tracked artifacts to review separately

At the time this policy was added, Git already tracks several artifacts this policy classifies as local-only: 74 files in `output/model_results/` (about 44.1 MB total), 19 generated files in `output/models/` (about 1.15 MB total), three legacy Parquet files (`output/merged_20261006.parquet`, `output/labeled_20261006.parquet`, and `output/forecast_labeled_20261006.parquet`, about 0.11 MB each), three XGBoost model JSON exports under `models/forecasting/`, and the GOES event catalogue `goes/goes_xray_event_list_2026.txt`. The catalogue is a small but necessary input to reproduce existing labels. These paths now match ignore rules, but because they are already tracked, Git will continue tracking changes to them. They remain in Git history. Coordinate any removal from the current index as a separate change; removing tracked paths or rewriting history is outside this ignore-policy update.

The repository audit found no tracked files over 5 MiB and no tracked filenames matching common secret/cookie/credential patterns. No notebooks or `.github/` workflow directory were present in the audited tree; the ignore rules leave notebook source and workflows trackable if added later. This is a filename/size audit, not a content scan; do not treat it as proof that files contain no secrets. A pre-existing Git status entry shows a deleted tracked path named `0`; it was left untouched.

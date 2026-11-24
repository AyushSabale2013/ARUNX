# Agent instructions for ARUN-X

## Project context

This repository builds a solar-flare research dataset from HEL1OS, SoLEXS, and GOES observations. Read `architecture.md`, `project_overview.md`, `data_pipeline.md`, and `training_plan.md` before making changes that affect data meaning or model evaluation. `z_plan.md` is the project progress note; confirm actual scripts and outputs because plans may lag implementation.

## Data and scientific integrity

- Keep all event and observation timestamps explicitly UTC.
- Keep raw source files immutable. Write derived data under `output/` or another clearly named generated-data directory.
- Respect instrument GTIs. Do not silently turn missing/out-of-GTI signal into a physical zero.
- Preserve quality flags and document any filtering or imputation.
- Do not put GOES-derived labels, future observations, or target columns into model input features.
- For forecast features, use only data available at or before the prediction timestamp. Rolling windows must be trailing.
- Split by chronological date, not randomly by minute row. Keep neighboring windows and each test period isolated from training.
- Do not report accuracy as the only metric for rare flare events. Report event counts and useful event-level/rare-event metrics.
- Never fabricate model scores. Label one-day experiments as exploratory and distinguish measured results from recommendations.

## Code-change conventions

- Make file paths relative to the repository root or derive them from `__file__`; avoid assumptions about the caller's working directory.
- Make date range, source locations, and output paths explicit configuration rather than hidden constants when extending a one-day script to batch use.
- Add validation for row count, timestamp uniqueness/order, expected columns, GTI consistency, and label counts at pipeline boundaries.
- Keep intermediate outputs reproducible and describe the inputs/configuration that generated them.
- Update the relevant documentation when the data schema, label semantics, or pipeline behavior changes.
- Do not change source data or delete generated/raw data unless explicitly requested.

## Before declaring a pipeline or model ready

Confirm that multiple dates can be processed, event/coverage audits pass, labels match the GOES event definitions, chronological splits contain adequate independent events, and preprocessing is fitted only on training data. For model claims, compare with a baseline on untouched held-out dates and record the split, seed, features, target horizon, threshold, and metrics.

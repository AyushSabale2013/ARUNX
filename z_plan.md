Let's pause and look at the bigger picture. Our current phase is about building a reliable real-world dataset for solar flare prediction. We are not training models yet.

I'll explain the plan in simple language, including what we have completed and what comes next.

# ARUN-X: Current phase plan

# Our main objective

Collect real solar observations from HEL1OS and SoLEXS, use GOES to identify when M/X-class flares occurred, combine everything into one clean dataset, and use that dataset to train machine learning and deep learning models.

## Phase 1 — Data collection

Initial testing complete

We need data from three sources.

| Instrument | What it gives us                                  |
| ---------- | ------------------------------------------------- |
| HEL1OS     | Hard X-ray observations                           |
| SoLEXS     | Soft X-ray observations                           |
| GOES       | Solar flare event information and classifications |

What we will do:

1. Select a suitable date range.
2. Download HEL1OS data for those dates.
3. Download SoLEXS data for those dates.
4. Download the corresponding GOES flare catalogue.

We already downloaded and tested one day of HEL1OS and SoLEXS data and parsed the GOES catalogue.

## Phase 2 — Data preprocessing

One-day pipeline complete

Raw satellite data is not immediately suitable for ML models.

We need to:

1. Extract FITS and compressed files.
2. Read the observations.
3. Convert timestamps into UTC.
4. Handle missing and invalid measurements.
5. Check instrument observation periods using GTI information.
6. Resample both instruments to a common one-minute interval.
7. Align their observations by timestamp.

Example:

| Time (UTC) | HEL1OS | SoLEXS |
| ---------- | ------ | ------ |
| 05:30      | 120.5  | 450.2  |
| 05:31      | 122.1  | 455.8  |
| 05:32      | 130.4  | 470.1  |

The values above are illustrative, not actual measurements.

We have already implemented and tested this process for 6 October.

## Phase 3 — Create the master dataset

One-day version complete

After preprocessing, we combine everything into a single Parquet dataset.

Each row represents one minute of solar observations.

It will contain:

- HEL1OS measurements.
- SoLEXS measurements.
- GOES-based flare labels.
- Missing-data indicators.
- GTI and quality indicators.
- Forecasting targets.

Think of this Parquet file as our central dataset. Every model will use data derived from it.

## Phase 4 — Generate labels

Partially complete

The satellite instruments provide measurements, but they don't directly tell us whether an M/X flare is occurring.

GOES provides the event information needed to generate labels.

We plan to build three prediction tasks:

A. Classification

Identify the flare class:

- 0 = No active M/X flare
- 1 = M-class flare
- 2 = X-class flare

B. Nowcasting

Determine whether an M/X flare is currently happening.

C. Forecasting

Predict whether an M/X flare will start within a future time window.

We have tested classification and 15-, 30-, and 60-minute forecasting labels on one day.

We still need to implement all final horizons, quality filters and the full multi-day labeling pipeline.

## Phase 5 — Choose the dataset date range

Current stage

We analysed the GOES catalogue and found 232 M/X events.

Our proposed date range is:

# February – July 2026

Proposed period, subject to satellite data availability

| Dataset split | Months  | M/X events |
| ------------- | ------- | ---------- |
| Training      | Feb–Apr | 116        |
| Validation    | May–Jun | 24         |
| Testing       | July    | 57         |

Total: 197 M/X events

Why this period?

Because it has substantially more M/X events than the other months in our catalogue.

One limitation: it contains only 12 X-class events. We should be careful about making claims specifically about X-class prediction.

## Phase 6 — Bulk download

Not started

Once we confirm the date range is available on PRADAN:

1. Download HEL1OS data in batches.
2. Download SoLEXS data in batches.
3. Extract the files.
4. Check which dates have both instruments.
5. Generate a list of successfully downloaded dates.

We will automate the downloads where practical and respect portal rate limits.

## Phase 7 — Build the final training dataset

Not started

This is an important phase.

We will:

- Merge all downloaded days.
- Generate all classification, nowcasting and forecasting labels.
- Identify bad-quality periods.
- Flag SEP-contaminated intervals where reliable flags or data are available.
- Handle missing observations.
- Split the data chronologically into training, validation and testing.

Important: We must avoid data leakage. Future observations must never accidentally become model inputs.

Done when all three splits have valid data and sufficient M/X events.

## Phase 8 — Connect the dataset to ARUN-X

Not started

We will integrate the real-data pipeline into your existing project.

We'll also:

- Fix the Train button.
- Remove manually entered performance scores.
- Load the actual Parquet dataset.
- Save preprocessing scalers with their trained models.
- Connect model predictions to the frontend.

## Phase 9 — Train and evaluate models

Not started

We'll train models in this order:

| Order | Model                        |
| ----- | ---------------------------- |
| 1     | Climatology baseline         |
| 2     | Logistic Regression          |
| 3     | Random Forest                |
| 4     | CNN-LSTM                     |
| 5     | HEL1OS + SoLEXS fusion model |

We'll run experiments with five random seeds where appropriate and evaluate using actual test data.

Our final result will be a comparison table containing measured performance metrics, not fabricated scores.

# Where exactly are we right now?

# ARUN-X progress

# 4 / 9

Fully completed roadmap steps

1

Environment setup

Complete

2

One-day data download

Complete

3

Read and plot

Complete

4

One-day merge pipeline

Complete

5

Select date range

In progress

6

Bulk download

Pending

7

Final labels and splits

Pending

8

Project integration

Pending

9

Model training

Pending

## What we do next

We are currently at Step 5: selecting the final date range.

Before downloading anything, we need to answer one question:

Does PRADAN have both HEL1OS and SoLEXS data available from February through July 2026?

We'll verify that first. If coverage is incomplete, we'll adjust the date range based on actual availability and GOES flare counts.

One rule for the rest of this project: we won't move to the next step until the current step's validation criteria pass.
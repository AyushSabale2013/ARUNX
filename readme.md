# ARUN-X Explained Simply

### Forecasting solar flares from X-ray observations of the Aditya-L1 satellite

**ARUN-X** stands for *Automated Radiation Understanding Network for X-ray Analysis*.

> **In one sentence:** We teach computers to read the X-rays coming from the Sun and to warn us **15, 30 or 60 minutes before** a solar flare starts.

| | |
|---|---|
| **Project record dated** | 12 November 2026 |
| **Satellite** | Aditya-L1 (ISRO) |
| **Instruments used** | SoLEXS (soft X-rays) and HEL1OS (hard X-rays) |
| **Answer key** | GOES flare catalogue (M and X class flares) |
| **Models compared** | 9 model types x 3 forecast horizons = 27 runs |

---

## Contents

1. [The Big Idea](#1-the-big-idea)
2. [Where the Data Comes From](#2-where-the-data-comes-from)
3. [Getting the Data Ready](#3-getting-the-data-ready)
4. [What We Are Predicting](#4-what-we-are-predicting)
5. [The Rare-Event Problem](#5-the-rare-event-problem)
6. [How the Data Was Split](#6-how-the-data-was-split)
7. [The Nine Models](#7-the-nine-models)
8. [How Success Was Measured](#8-how-success-was-measured)
9. [Results](#9-results)
10. [Limitations](#10-limitations)
11. [Project Status and Next Steps](#11-project-status-and-next-steps)
12. [Mini Dictionary](#12-mini-dictionary)

---

## 1. The Big Idea

A **solar flare** is a sudden burst of energy on the Sun. It shows up as a quick jump in X-ray light. Large flares can disturb satellites and communication systems, so an early warning is valuable.

Our approach has three simple steps:

| Step | What happens |
|---|---|
| **1. Observe** | Collect X-ray measurements from two instruments on Aditya-L1 |
| **2. Learn** | Train machine-learning models to recognise the pattern that appears just before a flare |
| **3. Warn** | Each minute, estimate whether a flare will start within the next 15, 30 or 60 minutes |

### The complete workflow

![ARUN-X workflow](figures/fig1_pipeline.png)

*Figure 1. Every stage writes its results to files, so each step can be checked later.*

---

## 2. Where the Data Comes From

### The two instruments

| Instrument | What it measures | Approximate energy range | Role in the story |
|---|---|---|---|
| **SoLEXS** | Soft X-rays | 1 to 30 keV | The thermal, "heating up" part of a flare |
| **HEL1OS** | Hard X-rays | 8 to 150 keV | The sudden, high-energy part of a flare |

Soft and hard X-rays describe **different parts of the same event**. Using both gives the models a fuller picture than either alone.

### The answer key: GOES

GOES is a separate satellite system that keeps an official list of flares. In this project it is used **only** to know when real flares happened and how strong they were.

> **Important:** GOES is never given to the models as input. Only Aditya-L1 data goes in. GOES is used for labels and for checking the results.

The evaluation focuses on the two strongest flare classes, **M-class** and **X-class**. In the evaluation catalogue there are **231** such events, and **56** of them fall in July 2026.

### How the files look

- Data was downloaded in bulk from the **PRADAN portal**, ISRO's data portal.
- The files are in **FITS** format, the standard file type in astronomy.
- A FITS file has several sections called **HDUs**, and each section can hold a different energy band.

**Worked example (5 November 2026):** data for 4 November was not available, so 5 November was inspected instead. One HEL1OS file had **43,169 rows** covering 00:00:06 to 11:59:34 UTC. Each row has a time (`MJD`, `ISOT`), a count rate (`CTR`) and its uncertainty (`STAT_ERR`).

---

## 3. Getting the Data Ready

Raw satellite files cannot go straight into a model. They are converted into one clean table with:

- **one row per minute,**
- **424 features** (the clues the model learns from), and
- **a target column** that says whether a flare began soon after.

The finished dataset is about **2.6 GB**.

### The 424 clues, in seven groups

| Group | In plain words |
|---|---|
| Signal logarithms | Compresses very large numbers so small and large changes can be compared |
| Missing-value flags | Marks the minutes where data was missing |
| GTI information | Shows when the instrument data is valid, so a gap is not mistaken for a quiet Sun |
| HEL1OS error features | How uncertain each hard X-ray measurement is |
| Rolling statistics | Averages and trends over the last 5, 15, 30 and 60 minutes |
| Lag and difference | How much the signal changed 1, 5 and 15 minutes earlier |
| Time of day | The hour of the day, written as sine and cosine |

### Safety rules against "cheating"

Machine-learning results can look better than they really are if the model accidentally sees the future. This is called **data leakage**. The project avoids it with these rules:

| Rule | Why it matters |
|---|---|
| Rolling and lag clues restart after a data gap | A gap is never treated as real data |
| They need consecutive one-minute samples | Keeps the clues honest |
| Clues use only the past, the target uses only the future | No peeking at the answer |
| Data is split by time, never mixed randomly | Mimics real forecasting |

---

## 4. What We Are Predicting

For every minute `t` and each horizon `H` (15, 30 or 60 minutes), the question is:

> **Will a qualifying flare start within the next H minutes?**
> The answer is **1 (yes)** or **0 (no)**.

This gives **three separate prediction tasks**, one for each horizon.

---

## 5. The Rare-Event Problem

Flares are rare. Most minutes have no flare coming.

![Class imbalance](figures/fig3_class_imbalance.png)

*Figure 2. Left: in the 15-minute training data only 0.656% of rows are flare rows. Right: the share of flare rows grows with the horizon and is higher in July than in June.*

**15-minute training data**

| | Count |
|---|---|
| Total rows | 198,111 |
| Flare-coming rows (positive) | 1,299 |
| No-flare rows (negative) | 196,812 |
| Positive share | about 0.656% |

**Why this matters:** a lazy model that always answers "no flare" would be right about 99% of the time and yet be completely useless. For that reason the project uses metrics designed for rare events, and gives flare examples extra weight during training (about **184 times** the weight of a normal example for the 15-minute deep-learning task).

---

## 6. How the Data Was Split

The data is split **by time**, like studying from old exam papers and then sitting a new one.

![Chronological split](figures/fig2_split_timeline.png)

*Figure 3. Training uses January to May, validation uses June, and testing uses July 2026.*

| Part | Dates (2026) | Purpose |
|---|---|---|
| **Training** | 1 January to 31 May | The model learns here |
| **Validation** | 1 June to 30 June | Choose settings, decide when to stop training, pick the alert threshold |
| **Test** | July | Final check |

> **Honest note:** July had already been used in earlier experiments. It is therefore an **exploratory benchmark** and not a perfectly fresh, untouched test set.

---

## 7. The Nine Models

Each model was trained for all three horizons, giving **9 x 3 = 27 runs**.

| Type | Models | Idea |
|---|---|---|
| **Classical (5)** | Logistic Regression, Random Forest, Extra Trees, Hist Gradient Boosting, LightGBM | Well-known, fast methods that work on the table of clues |
| **Boosting (1)** | XGBoost | Builds many small decision trees, each fixing the previous one's mistakes |
| **Deep learning (3)** | CNN, CNN AvgMax, CNN-LSTM | Neural networks that read the last 60 minutes as a sequence |

**About the deep models**

| Model | Trainable parameters | What makes it different |
|---|---|---|
| CNN | 315,009 | Finds local patterns across the 60-minute window |
| CNN AvgMax | 317,057 | A CNN variant that uses average and maximum pooling |
| CNN-LSTM | 331,905 | A CNN followed by an LSTM that remembers how the signal evolved |

All three read **60 consecutive minutes** with **424 features per minute** and were trained on a GPU (NVIDIA RTX 3050 Laptop, 6 GB).

---

## 8. How Success Was Measured

### Minute-by-minute checks (sample level)

| Measure | Question it answers |
|---|---|
| **Precision** | When the model warns, how often is it right? |
| **Recall** | Of all real flare minutes, how many did it catch? |
| **F1** | A single balance of precision and recall |
| **PR-AUC** (main score) | How good is the model overall on rare events? Higher is better |
| **False alarms per day** | How often does it warn when nothing happens? |

> Accuracy alone is misleading here because "no flare" dominates the data.

### Whole-flare checks (event level)

Event-level checking asks a more practical question: **did the model raise an alert before a real GOES M/X flare?** It reports events detected and missed, the **lead time** (how many minutes early the alert came) and the false-alarm episodes per day.

---

## 9. Results

### 9.1 Best model for each task

| Horizon | Best model | Test PR-AUC |
|---|---|---|
| **15 minutes** | Random Forest | **0.0759** |
| **30 minutes** | Logistic Regression | **0.1230** |
| **60 minutes** | Random Forest | **0.2503** |

### 9.2 All nine models side by side

![PR-AUC comparison](figures/fig4_pr_auc_comparison.png)

*Figure 4. Test PR-AUC on July 2026. The dashed red line marks what random guessing would score, which equals the share of flare rows in the test month.*

<details>
<summary>Show the same numbers as a table</summary>

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Extra Trees | 0.0640 | 0.1027 | 0.1991 |
| Hist Gradient Boosting | 0.0658 | 0.1138 | 0.2410 |
| LightGBM | 0.0524 | 0.0855 | 0.1979 |
| Logistic Regression | 0.0651 | **0.1230** | 0.2183 |
| Random Forest | **0.0759** | 0.1208 | **0.2503** |
| CNN | 0.0532 | 0.0827 | 0.1271 |
| CNN AvgMax | 0.0353 | 0.0824 | 0.1692 |
| CNN-LSTM | 0.0326 | 0.0924 | 0.1777 |
| XGBoost | 0.0239 | 0.0719 | 0.1498 |

</details>

### 9.3 How good is that, really?

A model guessing at random would score about the share of flare rows in the test month. Comparing the best model with that level gives a fair sense of real skill.

![Lift over random](figures/fig5_lift_over_random.png)

*Figure 5. The best model is roughly 3.4 to 4.1 times better than random guessing at every horizon.*

| Horizon | Random-guess level | Best model | Improvement |
|---|---|---|---|
| 15 minutes | 0.0186 | 0.0759 | about 4.1 times |
| 30 minutes | 0.0361 | 0.1230 | about 3.4 times |
| 60 minutes | 0.0652 | 0.2503 | about 3.8 times |

### 9.4 Key takeaways

- **Simple models won.** Random Forest and Logistic Regression beat the more complex deep-learning models on this benchmark.
- **Higher scores at longer horizons are not automatically better skill.** The share of flare rows also grows with the horizon, which lifts the score.
- **Deep models peaked very early.** Most reached their best validation score at epoch 1, which suggests they over-fit the training months quickly.

![Deep learning epochs](figures/fig7_deep_epochs.png)

*Figure 6. Best epoch versus epochs run for each deep model. Seven of the nine runs peaked by epoch 4, and six of them at epoch 1.*

### 9.5 Event-level example: Random Forest, 15 minutes

| Quantity | Value |
|---|---|
| GOES M/X events in July | 56 |
| Events detected | **41** (about 73%) |
| M-class and X-class matched | 40 and 1 |
| Average lead time | **13.68 minutes** |
| Median lead time | **15 minutes** |
| Alerts with exactly 15 minutes of lead | 33 of 41 (about 80.5%) |

> **Read with care:** only **one** X-class flare was matched, so no claim about X-class performance can be made. Lead times also bunch at 15 minutes partly because of the matching rule, which picks the *first* qualifying alert in the window. It does not prove every flare was predicted exactly 15 minutes ahead.

### 9.6 Detection versus false alarms

![Event trade-off](figures/fig6_event_tradeoff.png)

*Figure 7. Each point is one classical or XGBoost model at one horizon. The best spot is the top left: many flares caught with few false alarms.*

| Model and horizon | Events detected (of 56) | False-alarm episodes per day |
|---|---|---|
| LightGBM, 60 min | 52 | 14.5 |
| Random Forest, 60 min | 50 | 26.7 |
| Random Forest, 30 min | 45 | 10.3 |
| Random Forest, 15 min | 41 | 8.8 |

Catching more flares often comes with more false alarms. The best balance depends on how costly a false alarm is in practice.

> Event-level results for the three deep-learning models are not available yet because of a naming problem in the evaluation script (see Section 10).

---

## 10. Limitations

| Issue | What it means |
|---|---|
| **July is not untouched** | It was used in earlier experiments, so the scores are exploratory and not final proof |
| **Naming problem in the event evaluator** | All nine deep-learning runs were given the generic label `forecast`, so their event results cannot safely be matched to a model |
| **Coverage gaps** | The evaluator counts events even where the prediction data has gaps, so detection may be underestimated |
| **Very few large flares** | Only one X-class event was matched in the example, so no strong X-class claim is possible |
| **One test month, one run** | There are no repeated runs or confidence intervals yet |
| **Threshold policy** | Deep-model thresholds were chosen on validation data. For the classical models this still needs confirming |
| **Narrow scope** | One mission and one time span in 2026 |

### Details still to be recovered from the original code

These were not saved in the project notes, and **none of them has been guessed**:

- the exact list of downloaded files, their dates and checksums,
- how the two instruments were aligned and resampled,
- how missing values and outliers were handled,
- which scaler was used, and whether it was fitted on training data only,
- the exact rule for which flare classes count as a positive label,
- the full list of the 424 feature names, and whether the hard-to-soft ratio is included,
- the hyper-parameters of each model, plus optimizer, learning rate and batch size.

---

## 11. Project Status and Next Steps

| Stage | Status |
|---|---|
| Data download from PRADAN | Done (full file list still to be recovered) |
| FITS inspection | Done for the 5 November 2026 test day |
| Minute-level dataset with 424 features | Ready |
| 15, 30 and 60 minute targets | Ready |
| Time-based split | Done |
| 5 classical models x 3 horizons | Trained |
| XGBoost x 3 horizons | Trained |
| 3 deep models x 3 horizons | Trained |
| Sample-level comparison | Done and checked |
| Event-level evaluation | Run, but **preliminary** |
| Master results file (27 rows, 69 columns) | Created and checked |

### Next steps

1. Fix the event evaluator (model names and coverage) and re-run it. Save a **new** version of the outputs instead of overwriting the old ones.
2. Test on a **later, untouched** period.
3. Add repeated runs and confidence intervals.
4. Compare SoLEXS-only, HEL1OS-only and combined inputs.

---

## 12. Mini Dictionary

| Term | Easy meaning |
|---|---|
| **Solar flare** | A sudden burst of energy from the Sun |
| **X-ray** | A kind of light that our eyes cannot see |
| **Aditya-L1** | India's solar observatory satellite |
| **SoLEXS and HEL1OS** | The two X-ray instruments on Aditya-L1 |
| **GOES** | US satellites whose flare list is used as the answer key |
| **FITS and HDU** | An astronomy file format, and one section inside such a file |
| **GTI** | Time periods when the instrument data is valid |
| **Feature** | One input clue for the model |
| **Target** | The answer the model should predict |
| **Horizon** | How far ahead we predict (15, 30 or 60 minutes) |
| **Class imbalance** | One answer, here "no flare", is far more common than the other |
| **Chronological split** | Splitting the data by time and not at random |
| **Data leakage** | Letting the model accidentally see the answer |
| **Threshold** | The probability level above which an alert is raised |
| **Precision, recall, F1** | Alert accuracy, catch rate, and a balance of the two |
| **PR-AUC** | One score for rare-event quality; higher is better |
| **False alarm** | An alert that is not followed by a real flare |
| **Lead time** | How many minutes early the alert came |
| **CNN and LSTM** | Neural networks: one finds patterns, the other remembers sequences |

---

## Key Files in the Project

| File | What it holds |
|---|---|
| `output/ml_ready_minute_end_v2/` | The ready-to-use dataset (about 2.6 GB) |
| `output/goes_events_parsed.csv` | The GOES flare list |
| `output/model_results/ARUNX_MASTER_MODEL_EVALUATION.csv` | One file with all 27 model results |
| `scripts/ml/evaluate_event_level.py` | The event-level evaluator |
| `scripts/ml/build_master_model_evaluation.py` | Builds the master results file |

---

<p align="center"><b>ARUN-X</b> | Forecasting solar flares from space X-rays</p>

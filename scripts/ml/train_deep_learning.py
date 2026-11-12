"""
ARUN-X PyTorch deep-learning trainer for M/X flare forecasting.

Reads the existing chronological split files:
  output/ml_ready_minute_end_v2/forecast_{15m,30m,60m}_{train,validation,test}.parquet

The files contain timestamp, target, and the 424 features listed in feature_schema.json.
Only complete 60-row windows with exactly one-minute timestamp spacing are used.
Feature imputation/scaling is fitted on TRAIN only.

Examples:
  python scripts/ml/train_deep_learning.py --model cnn_lstm
  python scripts/ml/train_deep_learning.py --model cnn
  python scripts/ml/train_deep_learning.py --model both

Outputs are written under output/model_results/deep_learning/.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "output" / "ml_ready_minute_end_v2"
OUTPUT_DIR = ROOT / "output" / "model_results" / "deep_learning"

HORIZONS = (15, 30, 60)
SEQ_LEN = 60
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_names() -> list[str]:
    schema_path = DATA_DIR / "feature_schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema["features"]
    if len(features) != 424:
        raise ValueError(f"Expected 424 features in schema, found {len(features)}")
    return features


def load_split(horizon: int, split: str, feature_names: list[str]):
    path = DATA_DIR / f"forecast_{horizon}m_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path, columns=["timestamp", "target", *feature_names])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "target"]).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    # The prepared split files should contain known binary labels only.
    labels = df["target"].astype(np.int8).to_numpy()
    if not np.isin(labels, [0, 1]).all():
        bad = sorted(set(labels.tolist()) - {0, 1})
        raise ValueError(f"{path.name} contains unexpected target values: {bad}")

    features = (
        df[feature_names]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32, copy=True)
    )
    return df["timestamp"], features, labels


def fit_scaler(train_x: np.ndarray):
    # Fit median imputation and standardization on training rows only.
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

    imputed = np.where(np.isnan(train_x), medians[None, :], train_x)
    means = imputed.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = imputed.std(axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    return medians, means, stds


def transform_features(x, medians, means, stds):
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.where(np.isnan(x), medians[None, :], x)
    x = (x - means[None, :]) / stds[None, :]
    return np.clip(x, -10.0, 10.0).astype(np.float32, copy=False)


def valid_sequence_endpoints(timestamps: pd.Series, seq_len: int) -> np.ndarray:
    """Return endpoint row indices whose full input window is minute-contiguous."""
    if len(timestamps) < seq_len:
        return np.empty(0, dtype=np.int64)

    # Explicitly convert timestamps to nanoseconds before comparing.
    minute_ns = timestamps.dt.as_unit("ns").astype("int64").to_numpy()

    one_minute_ns = 60 * 1_000_000_000
    consecutive = np.diff(minute_ns) == one_minute_ns

    bad_gap_prefix = np.concatenate(
        ([0], np.cumsum(~consecutive, dtype=np.int64))
    )

    endpoints = np.arange(seq_len - 1, len(timestamps), dtype=np.int64)
    starts = endpoints - seq_len + 1
    gap_count = bad_gap_prefix[endpoints] - bad_gap_prefix[starts]

    return endpoints[gap_count == 0]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, endpoints: np.ndarray):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.endpoints = endpoints

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = int(self.endpoints[index])
        start = end - SEQ_LEN + 1
        # Shape returned: [time, features] = [60, 424].
        return self.x[start:end + 1], self.y[end], end


class CNNModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # [batch, time, features] -> [batch, features, time]
        return self.network(x.transpose(1, 2)).squeeze(1)


class CNNLSTMModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(z)
        return self.head(sequence[:, -1, :]).squeeze(1)


def make_loader(dataset, shuffle: bool, device):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities, actuals, endpoints = [], [], []
    for xb, yb, eb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        actuals.extend(yb.numpy().tolist())
        endpoints.extend(eb.numpy().tolist())
    return np.asarray(actuals, dtype=np.int8), np.asarray(probabilities), np.asarray(endpoints)


def choose_f1_threshold(y_true, probabilities):
    # Evaluate all unique validation probabilities plus sensible endpoints.
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    best = (0.0, 0.5)
    for threshold in candidates:
        pred = probabilities >= threshold
        score = f1_score(y_true, pred, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1], best[0]


def binary_metrics(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(np.int8)
    result = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2 else None
        ),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel().tolist(),
        "positive_samples": int(y_true.sum()),
        "samples": int(len(y_true)),
    }
    return result, predicted


def train_one(model_name, horizon, device, feature_names):
    print("\n" + "=" * 76)
    print(f"TRAINING {model_name.upper()} | FORECAST {horizon} MINUTES")
    print("=" * 76)

    ts_train, x_train_raw, y_train = load_split(horizon, "train", feature_names)
    ts_val, x_val_raw, y_val = load_split(horizon, "validation", feature_names)
    ts_test, x_test_raw, y_test = load_split(horizon, "test", feature_names)

    medians, means, stds = fit_scaler(x_train_raw)
    x_train = transform_features(x_train_raw, medians, means, stds)
    x_val = transform_features(x_val_raw, medians, means, stds)
    x_test = transform_features(x_test_raw, medians, means, stds)

    train_ends = valid_sequence_endpoints(ts_train, SEQ_LEN)
    val_ends = valid_sequence_endpoints(ts_val, SEQ_LEN)
    test_ends = valid_sequence_endpoints(ts_test, SEQ_LEN)

    if min(len(train_ends), len(val_ends), len(test_ends)) == 0:
        raise RuntimeError(
            "At least one split has no valid contiguous 60-minute sequences. "
            "Check timestamps and split files."
        )

    print(f"Features: {len(feature_names)} | Sequence: {SEQ_LEN} minutes")
    print(
        f"Valid sequences | train={len(train_ends):,}, "
        f"validation={len(val_ends):,}, test={len(test_ends):,}"
    )
    print(
        f"Positive labels | train={int(y_train[train_ends].sum()):,}, "
        f"validation={int(y_val[val_ends].sum()):,}, "
        f"test={int(y_test[test_ends].sum()):,}"
    )

    train_ds = SequenceDataset(x_train, y_train, train_ends)
    val_ds = SequenceDataset(x_val, y_val, val_ends)
    test_ds = SequenceDataset(x_test, y_test, test_ends)
    train_loader = make_loader(train_ds, True, device)
    train_eval_loader = make_loader(train_ds, False, device)
    val_loader = make_loader(val_ds, False, device)
    test_loader = make_loader(test_ds, False, device)

    positives = float(y_train[train_ends].sum())
    negatives = float(len(train_ends) - positives)
    if positives <= 0:
        raise RuntimeError("Training split has no positive forecast examples.")
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    model = CNNModel(len(feature_names)) if model_name == "cnn" else CNNLSTMModel(len(feature_names))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    run_dir = OUTPUT_DIR / model_name / f"forecast_{horizon}m"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"

    best_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        seen = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            seen += len(yb)

        val_actual, val_prob, _ = predict(model, val_loader, device)
        val_pr_auc = float(average_precision_score(val_actual, val_prob))
        val_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(np.log(np.clip(val_prob, 1e-7, 1 - 1e-7) /
                                    (1 - np.clip(val_prob, 1e-7, 1 - 1e-7))),
                             dtype=torch.float32),
                torch.tensor(val_actual, dtype=torch.float32),
            ).item()
        )
        scheduler.step(val_pr_auc)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "validation_pr_auc": val_pr_auc,
            "validation_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_PR_AUC={val_pr_auc:.5f} | "
            f"lr={row['learning_rate']:.2e} | "
            f"{time.time() - start_time:.1f}s"
        )

        if val_pr_auc > best_pr_auc + 1e-6:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "horizon_minutes": horizon,
                    "sequence_length": SEQ_LEN,
                    "feature_names": feature_names,
                    "model_state_dict": model.state_dict(),
                    "best_validation_pr_auc": best_pr_auc,
                    "best_epoch": best_epoch,
                    "pos_weight": float(pos_weight.item()),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping: no validation PR-AUC improvement for {PATIENCE} epochs.")
            break

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_actual, val_prob, val_endpoint_indices = predict(model, val_loader, device)
    threshold, val_best_f1 = choose_f1_threshold(val_actual, val_prob)

    test_actual, test_prob, test_endpoint_indices = predict(model, test_loader, device)
    test_metrics, test_predicted = binary_metrics(test_actual, test_prob, threshold)
    val_metrics, _ = binary_metrics(val_actual, val_prob, threshold)

    # Persist train-only preprocessing parameters for reproducible inference.
    np.savez_compressed(
        run_dir / "preprocessing.npz",
        medians=medians,
        means=means,
        stds=stds,
    )
    with (run_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    test_timestamps = ts_test.iloc[test_endpoint_indices].reset_index(drop=True)
    predictions = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual": test_actual,
        "probability": test_prob,
        "predicted": test_predicted,
    })
    predictions.to_csv(run_dir / f"forecast_{horizon}m_predictions.csv", index=False)

    summary = {
        "model": model_name,
        "horizon_minutes": horizon,
        "sequence_length": SEQ_LEN,
        "feature_count": len(feature_names),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "validation_best_f1_threshold": threshold,
        "validation_f1_at_selected_threshold": val_best_f1,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sequence_counts": {
            "train": int(len(train_ends)),
            "validation": int(len(val_ends)),
            "test": int(len(test_ends)),
        },
        "class_counts": {
            "train_positive": int(positives),
            "train_negative": int(negatives),
            "validation_positive": int(val_actual.sum()),
            "test_positive": int(test_actual.sum()),
        },
        "device": str(device),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBEST CHECKPOINT:", checkpoint_path)
    print("VALIDATION threshold:", f"{threshold:.6f}")
    print("TEST METRICS:")
    print(json.dumps(test_metrics, indent=2))
    print("Predictions:", run_dir / f"forecast_{horizon}m_predictions.csv")


def main():
    global EPOCHS, BATCH_SIZE, SEQ_LEN
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "cnn_lstm", "both"],
        default="cnn_lstm",
        help="Architecture to train (default: cnn_lstm)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=SEQ_LEN)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SEQ_LEN = args.sequence_length

    if any(h not in HORIZONS for h in args.horizons):
        raise ValueError(f"Horizons must be selected from {HORIZONS}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ARUN-X PyTorch Forecast Training")
    print("Project root:", ROOT)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    feature_names = load_feature_names()
    model_names = ["cnn", "cnn_lstm"] if args.model == "both" else [args.model]

    for model_name in model_names:
        for horizon in args.horizons:
            set_seed(SEED)
            train_one(model_name, horizon, device, feature_names)

    print("\nAll requested deep-learning runs completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

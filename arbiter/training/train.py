"""LightGBM model training script.

Usage:
    python -m arbiter.training.train [--dry-run]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from arbiter.models.calibration import PlattCalibrator
from arbiter.models.features import SPEC
from arbiter.training.dataset import temporal_split
from arbiter.training.evaluate import brier_score, expected_calibration_error

logger = logging.getLogger(__name__)

_LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
}
_NUM_ROUNDS = 300
_EARLY_STOPPING_ROUNDS = 30


def train_model(
    data_path: str | None = None,
    output_path: str = "models/arbiter_lgbm.pkl",
    dry_run: bool = False,
    *,
    features: npt.NDArray[np.float64] | None = None,
    labels: npt.NDArray[np.float64] | None = None,
    timestamps: npt.NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Train a LightGBM model on historical market snapshot data.

    Steps:
    1. Load dataset (from args, CSV, or DB)
    2. Temporal train/val/test split
    3. Train LightGBM with binary logloss
    4. Fit Platt scaling (logistic regression) on validation set
    5. Evaluate: Brier score, ECE
    6. Save model + calibrator to output_path (unless dry_run)

    Args:
        data_path: Optional CSV path. If None and features not provided, loads from DB.
        output_path: Where to save the trained model.
        dry_run: If True, trains on a small slice and skips saving.
        features: Pre-loaded feature array (overrides data_path and DB loading).
        labels: Pre-loaded label array.
        timestamps: Pre-loaded timestamp array for temporal splitting.

    Returns:
        Dict with val_brier, test_brier, val_ece, test_ece metrics.
    """
    if features is None or labels is None or timestamps is None:
        features, labels, timestamps = _load_data(data_path)

    if dry_run:
        n = min(len(features), 100)
        features = features[:n]
        labels = labels[:n]
        timestamps = timestamps[:n]

    splits = temporal_split(features, labels, timestamps)
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    booster = lgb.train(
        _LGBM_PARAMS,
        train_data,
        num_boost_round=_NUM_ROUNDS,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    # Fit Platt scaling (logistic regression) on validation predictions
    val_raw = np.asarray(booster.predict(x_val), dtype=np.float64)
    lr = LogisticRegression()
    lr.fit(val_raw.reshape(-1, 1), y_val)
    calibrator = PlattCalibrator(lr)

    # Evaluate: val metrics use RAW (uncalibrated) predictions to avoid circular
    # evaluation — the calibrator was fitted on val data, so evaluating calibrated
    # val predictions would overfit (val_ece ≈ 0 always).
    # Test metrics use calibrated predictions for true out-of-sample evaluation.
    test_raw = np.asarray(booster.predict(x_test), dtype=np.float64)
    test_cal = calibrator.predict(test_raw)

    metrics = {
        "val_brier": brier_score(val_raw, y_val),
        "test_brier": brier_score(test_cal, y_test),
        "val_ece": expected_calibration_error(val_raw, y_val),
        "test_ece": expected_calibration_error(test_cal, y_test),
    }

    logger.info(
        "Training complete — val_brier=%.4f  test_brier=%.4f  val_ece=%.4f  test_ece=%.4f",
        metrics["val_brier"],
        metrics["test_brier"],
        metrics["val_ece"],
        metrics["test_ece"],
    )

    if not dry_run:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({"model": booster, "calibrator": calibrator}, f)
        logger.info("Model saved to %s", output_path)

    return metrics


def _load_data(
    data_path: str | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load features, labels, and timestamps from CSV or raise if no source available."""
    if data_path is not None:
        import csv

        rows: list[list[float]] = []
        lab: list[float] = []
        ts: list[float] = []
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append([float(row.get(name, "nan")) for name in SPEC.names])
                lab.append(float(row["outcome"]))
                ts.append(float(row.get("timestamp", i)))
        return (
            np.array(rows, dtype=np.float64),
            np.array(lab, dtype=np.float64),
            np.array(ts, dtype=np.float64),
        )

    raise ValueError("No data source provided. Pass features/labels/timestamps or data_path.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM probability model")
    parser.add_argument("--data-path", type=str, default=None, help="CSV training data path")
    parser.add_argument(
        "--output", type=str, default="models/arbiter_lgbm.pkl", help="Model output"
    )
    parser.add_argument("--dry-run", action="store_true", help="Train on small slice, skip saving")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = train_model(data_path=args.data_path, output_path=args.output, dry_run=args.dry_run)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

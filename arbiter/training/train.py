"""LightGBM model training script.

Usage:
    python -m arbiter.training.train [--dry-run]
"""

from __future__ import annotations


def train_model(
    data_path: str | None = None,
    output_path: str = "models/arbiter_lgbm.pkl",
    dry_run: bool = False,
) -> None:
    """Train a LightGBM model on historical market snapshot data.

    Steps:
    1. Load dataset from DB (or CSV if data_path provided)
    2. Temporal train/val/test split
    3. Train LightGBM with binary logloss
    4. Fit isotonic calibration on validation set
    5. Evaluate: log-loss, Brier score, ECE
    6. Save model + calibrator to output_path

    Args:
        data_path: Optional CSV path. If None, loads from database.
        output_path: Where to save the trained model.
        dry_run: If True, runs one training step and exits.
    """
    raise NotImplementedError  # Phase 4


if __name__ == "__main__":
    import sys

    train_model(dry_run="--dry-run" in sys.argv)

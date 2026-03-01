"""Training pipeline for the probability estimation model."""

from arbiter.training.collector import (
    backfill_outcomes,
    collect_resolved_markets,
    snapshot_live_markets,
)
from arbiter.training.dataset import load_dataset, temporal_split
from arbiter.training.evaluate import brier_score, calibration_curve, expected_calibration_error
from arbiter.training.train import train_model

__all__ = [
    "backfill_outcomes",
    "brier_score",
    "calibration_curve",
    "collect_resolved_markets",
    "expected_calibration_error",
    "load_dataset",
    "snapshot_live_markets",
    "temporal_split",
    "train_model",
]

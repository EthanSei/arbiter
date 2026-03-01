"""Tests for the LightGBM training pipeline."""

import pickle
import tempfile
from pathlib import Path

import numpy as np

from arbiter.training.train import train_model


def _synthetic_dataset(n: int = 200) -> tuple:
    """Generate synthetic features and labels that LightGBM can learn from."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((n, 16))
    # Deterministic labels correlated with first feature
    labels = (features[:, 0] > 0).astype(float)
    timestamps = np.arange(n, dtype=np.float64)
    return features, labels, timestamps


class TestTrainModel:
    def test_dry_run_exits_without_saving(self):
        """dry_run=True completes without writing an output file."""
        features, labels, timestamps = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pkl")
            train_model(
                features=features,
                labels=labels,
                timestamps=timestamps,
                output_path=output_path,
                dry_run=True,
            )
            assert not Path(output_path).exists()

    def test_saves_pickle_with_model_and_calibrator(self):
        """Output pickle contains 'model' and 'calibrator' keys."""
        features, labels, timestamps = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pkl")
            train_model(
                features=features,
                labels=labels,
                timestamps=timestamps,
                output_path=output_path,
            )
            assert Path(output_path).exists()
            with open(output_path, "rb") as f:
                artifact = pickle.load(f)
            assert "model" in artifact
            assert "calibrator" in artifact

    def test_model_can_predict(self):
        """Loaded model produces predictions in (0, 1) range."""
        features, labels, timestamps = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pkl")
            train_model(
                features=features,
                labels=labels,
                timestamps=timestamps,
                output_path=output_path,
            )
            with open(output_path, "rb") as f:
                artifact = pickle.load(f)
        preds = artifact["model"].predict(features[:5])
        assert preds.shape == (5,)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_returns_metrics_dict(self):
        """train_model returns a dict with val and test metrics."""
        features, labels, timestamps = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pkl")
            metrics = train_model(
                features=features,
                labels=labels,
                timestamps=timestamps,
                output_path=output_path,
            )
        assert isinstance(metrics, dict)
        assert "val_brier" in metrics
        assert "test_brier" in metrics
        assert "val_ece" in metrics
        assert "test_ece" in metrics

    def test_metrics_are_valid_floats(self):
        """All returned metric values are finite floats."""
        features, labels, timestamps = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pkl")
            metrics = train_model(
                features=features,
                labels=labels,
                timestamps=timestamps,
                output_path=output_path,
            )
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is not float"
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_dry_run_still_returns_metrics(self):
        """dry_run mode trains on a slice and still returns metrics."""
        features, labels, timestamps = _synthetic_dataset()
        metrics = train_model(features=features, labels=labels, timestamps=timestamps, dry_run=True)
        assert isinstance(metrics, dict)
        assert "val_brier" in metrics

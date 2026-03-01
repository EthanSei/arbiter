"""Tests for model evaluation utilities — Brier score, ECE, calibration curve."""

import numpy as np
import pytest

from arbiter.training.evaluate import (
    brier_score,
    calibration_curve,
    expected_calibration_error,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        """Perfect binary predictions score 0."""
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(probs, outcomes) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Completely wrong binary predictions score 1."""
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(probs, outcomes) == pytest.approx(1.0)

    def test_uniform_half_predictions(self):
        """All-0.5 predictions on balanced outcomes → 0.25."""
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(probs, outcomes) == pytest.approx(0.25)

    def test_single_sample(self):
        """Single prediction works."""
        probs = np.array([0.8])
        outcomes = np.array([1.0])
        # (0.8 - 1.0)^2 = 0.04
        assert brier_score(probs, outcomes) == pytest.approx(0.04)

    def test_asymmetric_predictions(self):
        """Non-trivial mixed probabilities."""
        probs = np.array([0.9, 0.1])
        outcomes = np.array([1.0, 0.0])
        # ((0.9-1)^2 + (0.1-0)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
        assert brier_score(probs, outcomes) == pytest.approx(0.01)


class TestExpectedCalibrationError:
    def test_perfectly_calibrated(self):
        """Perfectly calibrated model has ECE ≈ 0."""
        # 100 samples in [0,1] where outcome rate matches predicted prob
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.0, 1.0, 1000)
        outcomes = (rng.uniform(0, 1, 1000) < probs).astype(float)
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        # With 1000 samples, ECE should be small (< 0.05)
        assert ece < 0.05

    def test_maximally_miscalibrated(self):
        """All predictions 0.9 but all outcomes 0 → large ECE."""
        probs = np.full(100, 0.9)
        outcomes = np.zeros(100)
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        # Should be close to 0.9 (predicted 0.9, actual 0.0)
        assert ece == pytest.approx(0.9, abs=0.01)

    def test_empty_bins_ignored(self):
        """Bins with no samples don't affect ECE."""
        # All predictions clustered in one bin
        probs = np.array([0.51, 0.52, 0.53, 0.54])
        outcomes = np.array([1.0, 1.0, 0.0, 0.0])
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        # One non-empty bin: |mean(probs) - mean(outcomes)| = |0.525 - 0.5| = 0.025
        assert ece == pytest.approx(0.025, abs=0.01)

    def test_ece_range(self):
        """ECE is always in [0, 1]."""
        rng = np.random.default_rng(99)
        probs = rng.uniform(0, 1, 200)
        outcomes = rng.choice([0.0, 1.0], 200)
        ece = expected_calibration_error(probs, outcomes)
        assert 0.0 <= ece <= 1.0


class TestCalibrationCurve:
    def test_returns_two_arrays(self):
        """Returns (mean_predicted, fraction_positive) arrays."""
        probs = np.linspace(0.05, 0.95, 100)
        outcomes = (probs > 0.5).astype(float)
        mean_pred, frac_pos = calibration_curve(probs, outcomes, n_bins=10)
        assert isinstance(mean_pred, np.ndarray)
        assert isinstance(frac_pos, np.ndarray)
        assert len(mean_pred) == len(frac_pos)

    def test_non_empty_bins_only(self):
        """Only returns data for non-empty bins."""
        # All predictions in [0.4, 0.6] — only a few bins populated
        probs = np.array([0.45, 0.46, 0.55, 0.56])
        outcomes = np.array([0.0, 1.0, 1.0, 0.0])
        mean_pred, frac_pos = calibration_curve(probs, outcomes, n_bins=10)
        assert len(mean_pred) <= 10
        assert len(mean_pred) > 0

    def test_perfect_calibration_diagonal(self):
        """Perfectly calibrated predictions lie on the diagonal."""
        rng = np.random.default_rng(123)
        probs = rng.uniform(0, 1, 5000)
        outcomes = (rng.uniform(0, 1, 5000) < probs).astype(float)
        mean_pred, frac_pos = calibration_curve(probs, outcomes, n_bins=10)
        # Each bin should be close to diagonal
        np.testing.assert_allclose(mean_pred, frac_pos, atol=0.08)

    def test_mean_predicted_in_range(self):
        """Mean predicted values are in [0, 1]."""
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        mean_pred, frac_pos = calibration_curve(probs, outcomes, n_bins=5)
        assert np.all(mean_pred >= 0.0)
        assert np.all(mean_pred <= 1.0)
        assert np.all(frac_pos >= 0.0)
        assert np.all(frac_pos <= 1.0)

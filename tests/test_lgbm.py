"""Tests for the LightGBM probability estimator."""

from __future__ import annotations

import pickle
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arbiter.ingestion.base import Contract
from arbiter.models.lgbm import LGBMEstimator


@pytest.fixture
def contract() -> Contract:
    """A typical open contract."""
    return Contract(
        source="kalshi",
        contract_id="TICKER-YES",
        title="Will X happen?",
        category="politics",
        yes_price=0.60,
        no_price=0.40,
        yes_bid=0.58,
        yes_ask=0.62,
        last_price=0.59,
        volume_24h=50_000.0,
        open_interest=120_000.0,
        expires_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        url="https://kalshi.com/markets/TICKER",
        status="open",
    )


def _make_mock_model(raw_output: float = 0.7) -> MagicMock:
    """Create a mock LightGBM booster that returns a fixed raw output."""
    model = MagicMock()
    model.predict.return_value = np.array([raw_output])
    return model


def _make_mock_calibrator(calibrated_output: float = 0.65) -> MagicMock:
    """Create a mock isotonic calibrator that returns a fixed calibrated output."""
    cal = MagicMock()
    cal.predict.return_value = np.array([calibrated_output])
    return cal


def _build_estimator(
    model: object | None = None,
    calibrator: object | None = None,
) -> LGBMEstimator:
    """Build an estimator and inject mock model/calibrator directly."""
    estimator = LGBMEstimator(model_path=None)
    if model is not None:
        estimator._model = model  # type: ignore[attr-defined]
        estimator._calibrator = calibrator  # type: ignore[attr-defined]
    return estimator


class TestLGBMEstimatorNoModel:
    """Fallback behavior when no model is loaded."""

    async def test_fallback_returns_market_midpoint(self, contract: Contract) -> None:
        estimator = LGBMEstimator(model_path=None)
        prob = await estimator.estimate(contract)
        assert prob == pytest.approx(0.60)

    async def test_fallback_clamps_to_valid_range(self) -> None:
        c = Contract(
            source="kalshi",
            contract_id="T",
            title="T",
            category="c",
            yes_price=0.0,
            no_price=1.0,
            yes_bid=0.0,
            yes_ask=0.0,
            last_price=None,
            volume_24h=0.0,
            open_interest=0.0,
            expires_at=None,
            url="",
            status="open",
        )
        estimator = LGBMEstimator(model_path=None)
        prob = await estimator.estimate(c)
        assert 0.0 < prob < 1.0

    def test_model_loaded_is_false_without_path(self) -> None:
        estimator = LGBMEstimator(model_path=None)
        assert estimator.model_loaded is False

    def test_calibrated_is_false_without_model(self) -> None:
        estimator = LGBMEstimator(model_path=None)
        assert estimator.calibrated is False


class TestLGBMEstimatorWithModel:
    """Behavior with a loaded model."""

    async def test_returns_calibrated_probability(self, contract: Contract) -> None:
        model = _make_mock_model(raw_output=0.7)
        calibrator = _make_mock_calibrator(calibrated_output=0.65)
        estimator = _build_estimator(model, calibrator)

        prob = await estimator.estimate(contract)
        assert prob == pytest.approx(0.65)
        model.predict.assert_called_once()
        calibrator.predict.assert_called_once()

    async def test_returns_raw_probability_without_calibrator(self, contract: Contract) -> None:
        model = _make_mock_model(raw_output=0.72)
        estimator = _build_estimator(model, calibrator=None)

        prob = await estimator.estimate(contract)
        assert prob == pytest.approx(0.72)

    def test_model_loaded_is_true_when_injected(self) -> None:
        estimator = _build_estimator(_make_mock_model())
        assert estimator.model_loaded is True

    def test_calibrated_is_false_without_calibrator(self) -> None:
        estimator = _build_estimator(_make_mock_model(), calibrator=None)
        assert estimator.calibrated is False

    def test_calibrated_is_true_with_calibrator(self) -> None:
        estimator = _build_estimator(_make_mock_model(), _make_mock_calibrator())
        assert estimator.calibrated is True

    async def test_clamps_output_to_valid_range(self, contract: Contract) -> None:
        model = _make_mock_model(raw_output=1.1)
        estimator = _build_estimator(model)

        prob = await estimator.estimate(contract)
        assert 0.0 < prob < 1.0

    async def test_nan_output_falls_back_to_midpoint(self, contract: Contract) -> None:
        model = _make_mock_model(raw_output=float("nan"))
        estimator = _build_estimator(model)

        prob = await estimator.estimate(contract)
        assert prob == pytest.approx(0.60)

    async def test_clamps_below_zero(self, contract: Contract) -> None:
        model = _make_mock_model(raw_output=-0.1)
        estimator = _build_estimator(model)

        prob = await estimator.estimate(contract)
        assert 0.0 < prob < 1.0


class TestLGBMEstimatorLoadFromDisk:
    """Loading model weights from pickle files."""

    def test_raises_on_invalid_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            LGBMEstimator(model_path="/nonexistent/model.pkl")

    def test_loads_model_from_pickle(self) -> None:
        # Use simple objects that can be pickled (not MagicMock)
        model_data = {"model": "fake_model", "calibrator": "fake_calibrator"}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model_data, f)
            path = f.name

        estimator = LGBMEstimator(model_path=path)
        assert estimator.model_loaded is True
        assert estimator._model == "fake_model"  # type: ignore[attr-defined]
        assert estimator._calibrator == "fake_calibrator"  # type: ignore[attr-defined]
        Path(path).unlink()


class TestLGBMEstimatorFeatureIntegration:
    """Verify features are extracted and passed to the model correctly."""

    async def test_model_receives_correct_feature_count(self, contract: Contract) -> None:
        from arbiter.models.features import SPEC

        model = _make_mock_model()
        estimator = _build_estimator(model)

        await estimator.estimate(contract)
        call_args = model.predict.call_args[0][0]
        assert call_args.shape == (1, len(SPEC.names))


class TestLGBMEstimatorPredictionErrors:
    """Graceful fallback when the model raises during prediction."""

    async def test_prediction_exception_falls_back_to_midpoint(self, contract: Contract) -> None:
        model = MagicMock()
        model.predict.side_effect = Exception(
            "The number of features in data (13) is not the same as it was in training data (16)."
        )
        estimator = _build_estimator(model)

        prob = await estimator.estimate(contract)
        # Falls back to market midpoint (yes_price=0.60)
        assert prob == pytest.approx(0.60)

    async def test_prediction_exception_does_not_propagate(self, contract: Contract) -> None:
        model = MagicMock()
        model.predict.side_effect = ValueError("feature mismatch")
        estimator = _build_estimator(model)

        # Should not raise
        prob = await estimator.estimate(contract)
        assert 0.0 < prob < 1.0

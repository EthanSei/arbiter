"""Tests for training data generation from candlestick history."""

from __future__ import annotations

import csv
import math
import pickle
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arbiter.models.features import SPEC
from arbiter.training.historical import backtest_from_csv, build_training_samples, candle_to_sample

# ---------------------------------------------------------------------------
# Fixtures — realistic Kalshi candlestick + market data
# ---------------------------------------------------------------------------

# A settled market dict (as returned by KalshiClient.fetch_settled)
SETTLED_MARKET = {
    "ticker": "KXCPI-26JAN-T0.003",
    "event_ticker": "KXCPI-26JAN",
    "title": "Will CPI rise more than 0.3% in January 2026?",
    "category": "Economics",
    "last_price": 25,
    "result": "no",
    "volume": 108385,
    "close_time": "2026-01-15T21:00:00Z",
    "status": "finalized",
}

SETTLED_MARKET_YES = {
    "ticker": "KXCPI-26JAN-T0.001",
    "event_ticker": "KXCPI-26JAN",
    "title": "Will CPI rise more than 0.1% in January 2026?",
    "category": "Economics",
    "last_price": 92,
    "result": "yes",
    "volume": 184087,
    "close_time": "2026-01-15T21:00:00Z",
    "status": "finalized",
}

# close_time as Unix timestamp
CLOSE_TS = int(
    datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC).timestamp()
)

# Candles: hourly intervals. end_period_ts is Unix seconds.
# These are ~48h before close_time.
CANDLE_48H_BEFORE = {
    "end_period_ts": CLOSE_TS - 48 * 3600,
    "yes_price": {"open": 0.20, "high": 0.25, "low": 0.18, "close": 0.22},
    "no_price": {"open": 0.80, "high": 0.82, "low": 0.75, "close": 0.78},
    "volume": 500,
}

CANDLE_30H_BEFORE = {
    "end_period_ts": CLOSE_TS - 30 * 3600,
    "yes_price": {"open": 0.22, "high": 0.28, "low": 0.21, "close": 0.26},
    "no_price": {"open": 0.78, "high": 0.79, "low": 0.72, "close": 0.74},
    "volume": 800,
}

# This candle is only 12h before close — inside the 24h exclusion window
CANDLE_12H_BEFORE = {
    "end_period_ts": CLOSE_TS - 12 * 3600,
    "yes_price": {"open": 0.24, "high": 0.30, "low": 0.23, "close": 0.28},
    "no_price": {"open": 0.76, "high": 0.77, "low": 0.70, "close": 0.72},
    "volume": 1200,
}

# This candle is 1h before close — deep inside exclusion window
CANDLE_1H_BEFORE = {
    "end_period_ts": CLOSE_TS - 1 * 3600,
    "yes_price": {"open": 0.25, "high": 0.26, "low": 0.24, "close": 0.25},
    "no_price": {"open": 0.75, "high": 0.76, "low": 0.74, "close": 0.75},
    "volume": 2000,
}


# ===========================================================================
# TestCandleToSample
# ===========================================================================


class TestCandleToSample:
    def test_basic_features(self) -> None:
        """Should extract yes_price, no_price, last_price from candle close."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert sample["yes_price"] == pytest.approx(0.22)
        assert sample["no_price"] == pytest.approx(1.0 - 0.22)
        assert sample["last_price"] == pytest.approx(0.22)

    def test_log_volume(self) -> None:
        """log_volume_24h should be log1p of candle volume."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert sample["log_volume_24h"] == pytest.approx(math.log1p(500))

    def test_time_to_expiry(self) -> None:
        """time_to_expiry_hours should be (close_time - candle_ts) / 3600."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert sample["time_to_expiry_hours"] == pytest.approx(48.0)

    def test_overround_is_zero_for_synthetic(self) -> None:
        """Overround = yes + no - 1, which is 0 when no_price = 1 - yes_price."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert sample["overround"] == pytest.approx(0.0)

    def test_day_of_week_and_hour(self) -> None:
        """day_of_week and hour_of_day should come from candle timestamp."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        ts = CANDLE_48H_BEFORE["end_period_ts"]
        dt = datetime.fromtimestamp(ts, tz=UTC)
        assert sample["day_of_week"] == float(dt.weekday())
        assert sample["hour_of_day"] == float(dt.hour)

    def test_outcome_no(self) -> None:
        """Outcome should be 0.0 for result='no'."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert sample["outcome"] == 0.0

    def test_outcome_yes(self) -> None:
        """Outcome should be 1.0 for result='yes'."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="yes",
        )
        assert sample["outcome"] == 1.0

    def test_timestamp_is_candle_ts(self) -> None:
        """timestamp should be the candle's end_period_ts."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert sample["timestamp"] == float(CANDLE_48H_BEFORE["end_period_ts"])

    def test_nan_features(self) -> None:
        """Features unavailable from candle data should be NaN."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert math.isnan(sample["bid_ask_spread"])
        assert math.isnan(sample["log_open_interest"])
        assert math.isnan(sample["price_discrepancy"])
        assert math.isnan(sample["volume_ratio"])

    def test_lag_features_without_history(self) -> None:
        """Without price_history, lag features should be NaN."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        assert math.isnan(sample["price_delta_1h"])
        assert math.isnan(sample["price_delta_24h"])
        assert math.isnan(sample["price_volatility_24h"])
        assert math.isnan(sample["volume_ratio_24h"])

    def test_lag_features_with_history(self) -> None:
        """With price_history, lag features should be computed."""
        history = [0.18, 0.20, 0.22]  # 3 prior close prices
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
            price_history=history,
        )
        # delta_1h = current - previous
        assert sample["price_delta_1h"] == pytest.approx(0.22 - 0.20)
        # delta_24h = current - first
        assert sample["price_delta_24h"] == pytest.approx(0.22 - 0.18)
        # volatility = std of history
        import numpy as np

        assert sample["price_volatility_24h"] == pytest.approx(float(np.std(history)))

    def test_all_spec_features_present(self) -> None:
        """Sample dict should contain all SPEC feature names + outcome + timestamp."""
        from arbiter.models.features import SPEC

        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=CLOSE_TS,
            result="no",
        )
        for name in SPEC.names:
            assert name in sample, f"Missing feature: {name}"
        assert "outcome" in sample
        assert "timestamp" in sample

    def test_no_close_time(self) -> None:
        """When close_time_ts is None, time_to_expiry should be NaN."""
        sample = candle_to_sample(
            CANDLE_48H_BEFORE,
            close_time_ts=None,
            result="no",
        )
        assert math.isnan(sample["time_to_expiry_hours"])


# ===========================================================================
# TestBuildTrainingSamples
# ===========================================================================


class TestBuildTrainingSamples:
    def test_excludes_candles_within_24h_of_close(self) -> None:
        """Candles within 24h of close_time should be excluded to prevent leakage."""
        candles = [CANDLE_48H_BEFORE, CANDLE_30H_BEFORE, CANDLE_12H_BEFORE, CANDLE_1H_BEFORE]
        samples = build_training_samples(
            [SETTLED_MARKET],
            {"KXCPI-26JAN-T0.003": candles},
        )
        # Only 48h and 30h candles should survive
        assert len(samples) == 2
        timestamps = [s["timestamp"] for s in samples]
        assert float(CANDLE_48H_BEFORE["end_period_ts"]) in timestamps
        assert float(CANDLE_30H_BEFORE["end_period_ts"]) in timestamps

    def test_computes_lag_features_from_history(self) -> None:
        """Earlier candles should be used as price_history for later candles."""
        candles = [CANDLE_48H_BEFORE, CANDLE_30H_BEFORE]
        samples = build_training_samples(
            [SETTLED_MARKET],
            {"KXCPI-26JAN-T0.003": candles},
        )
        # First sample (48h) has no history → NaN lag features
        ts_48h = float(CANDLE_48H_BEFORE["end_period_ts"])
        first = [s for s in samples if s["timestamp"] == ts_48h][0]
        assert math.isnan(first["price_delta_1h"])

        # Second sample (30h) has history from 48h candle
        ts_30h = float(CANDLE_30H_BEFORE["end_period_ts"])
        second = [s for s in samples if s["timestamp"] == ts_30h][0]
        assert not math.isnan(second["price_delta_24h"])

    def test_skips_market_with_no_candles(self) -> None:
        """Markets with no matching candle data should produce zero samples."""
        samples = build_training_samples(
            [SETTLED_MARKET],
            {},  # no candles
        )
        assert len(samples) == 0

    def test_skips_market_with_no_result(self) -> None:
        """Markets without a result field should be skipped."""
        market_no_result = {**SETTLED_MARKET}
        del market_no_result["result"]
        samples = build_training_samples(
            [market_no_result],
            {"KXCPI-26JAN-T0.003": [CANDLE_48H_BEFORE]},
        )
        assert len(samples) == 0

    def test_multiple_markets(self) -> None:
        """Should process multiple markets, each with their own candles."""
        candles_market1 = [CANDLE_48H_BEFORE]
        candles_market2 = [CANDLE_30H_BEFORE]
        samples = build_training_samples(
            [SETTLED_MARKET, SETTLED_MARKET_YES],
            {
                "KXCPI-26JAN-T0.003": candles_market1,
                "KXCPI-26JAN-T0.001": candles_market2,
            },
        )
        assert len(samples) == 2

    def test_outcome_matches_market_result(self) -> None:
        """Each sample's outcome should match its parent market's result."""
        samples = build_training_samples(
            [SETTLED_MARKET_YES],
            {"KXCPI-26JAN-T0.001": [CANDLE_48H_BEFORE]},
        )
        assert len(samples) == 1
        assert samples[0]["outcome"] == 1.0

    def test_market_without_close_time(self) -> None:
        """Markets without close_time should still generate samples (no time filter)."""
        market = {**SETTLED_MARKET, "close_time": None}
        samples = build_training_samples(
            [market],
            {"KXCPI-26JAN-T0.003": [CANDLE_48H_BEFORE, CANDLE_12H_BEFORE]},
        )
        # Without close_time, can't filter by 24h window — include all candles
        assert len(samples) == 2

    def test_custom_exclude_hours(self) -> None:
        """Custom exclude_hours should change the exclusion window."""
        candles = [CANDLE_48H_BEFORE, CANDLE_30H_BEFORE, CANDLE_12H_BEFORE]
        # Exclude candles within 36h of close → only 48h survives
        samples = build_training_samples(
            [SETTLED_MARKET],
            {"KXCPI-26JAN-T0.003": candles},
            exclude_hours=36,
        )
        assert len(samples) == 1
        assert samples[0]["timestamp"] == float(CANDLE_48H_BEFORE["end_period_ts"])

    def test_empty_inputs(self) -> None:
        """Empty markets list should return empty samples."""
        assert build_training_samples([], {}) == []


# ===========================================================================
# TestBacktestFromCSV
# ===========================================================================


class _MockBooster:
    """Fake LightGBM booster that returns fixed probabilities."""

    def __init__(self, prob: float) -> None:
        self._prob = prob

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self._prob)


class _MockCalibrator:
    """Identity calibrator — returns predictions unchanged."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x


def _write_test_csv(path: Path, samples: list[dict[str, float]]) -> None:
    """Write sample dicts to a CSV matching SPEC format."""
    fieldnames = SPEC.names + ["outcome", "timestamp"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _write_mock_model(path: Path, prob: float) -> None:
    """Write a mock model bundle to a pickle file."""
    bundle = {"model": _MockBooster(prob), "calibrator": _MockCalibrator()}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def _make_sample(yes_price: float, outcome: float, timestamp: float) -> dict[str, float]:
    """Create a minimal training sample dict."""
    sample: dict[str, float] = {}
    sample["yes_price"] = yes_price
    sample["no_price"] = 1.0 - yes_price
    sample["bid_ask_spread"] = float("nan")
    sample["last_price"] = yes_price
    sample["log_volume_24h"] = math.log1p(100)
    sample["log_open_interest"] = float("nan")
    sample["time_to_expiry_hours"] = 48.0
    sample["overround"] = 0.0
    sample["day_of_week"] = 1.0
    sample["hour_of_day"] = 14.0
    sample["price_discrepancy"] = float("nan")
    sample["volume_ratio"] = float("nan")
    sample["price_delta_1h"] = float("nan")
    sample["price_delta_24h"] = float("nan")
    sample["volume_ratio_24h"] = float("nan")
    sample["price_volatility_24h"] = float("nan")
    sample["outcome"] = outcome
    sample["timestamp"] = timestamp
    return sample


class TestBacktestFromCSV:
    def test_returns_metrics_dict(self) -> None:
        """Should return a dict with expected metric keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            # Model predicts 0.70 — market at 0.50 → positive EV on YES
            _write_mock_model(model_path, prob=0.70)
            samples = [_make_sample(0.50, 1.0, float(i)) for i in range(100)]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(str(model_path), str(csv_path))

        assert "total_pnl" in result
        assert "num_trades" in result
        assert "win_rate" in result
        assert "max_drawdown" in result
        assert "sharpe" in result
        assert "final_bankroll" in result
        assert "test_samples" in result

    def test_positive_edge_profitable(self) -> None:
        """Model with genuine edge should produce positive P&L."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            # Model predicts 0.80, market at 0.50, all resolve YES
            _write_mock_model(model_path, prob=0.80)
            samples = [_make_sample(0.50, 1.0, float(i)) for i in range(100)]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(str(model_path), str(csv_path), fee_rate=0.03)

        assert result["total_pnl"] > 0
        assert result["num_trades"] > 0
        assert result["win_rate"] == 1.0  # all resolve as predicted

    def test_negative_edge_no_trades(self) -> None:
        """Model with no edge should make zero trades."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            # Model predicts 0.50, market at 0.50 → EV = -fee_rate → no trades
            _write_mock_model(model_path, prob=0.50)
            samples = [_make_sample(0.50, 1.0, float(i)) for i in range(100)]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(str(model_path), str(csv_path), fee_rate=0.03)

        assert result["num_trades"] == 0
        assert result["total_pnl"] == 0.0

    def test_uses_test_fraction_only(self) -> None:
        """Should only trade on the test split, not the training portion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            _write_mock_model(model_path, prob=0.80)
            samples = [_make_sample(0.50, 1.0, float(i)) for i in range(100)]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(
                str(model_path), str(csv_path), test_fraction=0.20
            )

        # 20% of 100 = 20 test samples
        assert result["test_samples"] == 20.0

    def test_empty_csv_returns_zero_metrics(self) -> None:
        """Empty CSV should return zeroed-out metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            _write_mock_model(model_path, prob=0.70)
            _write_test_csv(csv_path, [])

            result = backtest_from_csv(str(model_path), str(csv_path))

        assert result["num_trades"] == 0
        assert result["total_pnl"] == 0.0

    def test_losing_trades_drawdown(self) -> None:
        """When all trades lose, max_drawdown should be significant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            # Model predicts 0.80 YES, but all resolve NO → losses
            _write_mock_model(model_path, prob=0.80)
            samples = [_make_sample(0.50, 0.0, float(i)) for i in range(100)]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(str(model_path), str(csv_path), fee_rate=0.03)

        assert result["total_pnl"] < 0
        assert result["max_drawdown"] > 0
        assert result["win_rate"] == 0.0
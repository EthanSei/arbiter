"""Audit tests: verify consistency between historical.py, engine.py, ev.py, and metrics.py.

These tests expose inconsistencies and bugs found during the ML pipeline audit.
Each test targets a specific discrepancy between the two backtesting systems.
"""

from __future__ import annotations

import csv
import math
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from arbiter.models.features import SPEC
from arbiter.scoring.kelly import kelly_criterion
from arbiter.training.historical import backtest_from_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockBooster:
    """Fake LightGBM booster returning a fixed probability."""

    def __init__(self, prob: float) -> None:
        self._prob = prob

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self._prob)


class _MockCalibrator:
    """Identity calibrator — returns predictions unchanged."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x


def _make_sample(yes_price: float, outcome: float, timestamp: float) -> dict[str, float]:
    sample: dict[str, float] = {}
    sample["yes_price"] = yes_price
    sample["no_price"] = 1.0 - yes_price
    sample["bid_ask_spread"] = float("nan")
    sample["last_price"] = yes_price
    sample["log_volume_24h"] = math.log1p(100)
    sample["log_open_interest"] = float("nan")
    sample["time_to_expiry_hours"] = 48.0
    sample["day_of_week"] = 1.0
    sample["hour_of_day"] = 14.0
    sample["price_delta_1h"] = float("nan")
    sample["price_delta_24h"] = float("nan")
    sample["volume_ratio_24h"] = float("nan")
    sample["price_volatility_24h"] = float("nan")
    sample["outcome"] = outcome
    sample["timestamp"] = timestamp
    return sample


def _write_test_csv(path: Path, samples: list[dict[str, float]]) -> None:
    fieldnames = SPEC.names + ["outcome", "timestamp"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _write_mock_model(path: Path, prob: float) -> None:
    bundle = {"model": _MockBooster(prob), "calibrator": _MockCalibrator()}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


# ===========================================================================
# 1. Sharpe ratio: historical.py should match metrics.sharpe_ratio
# ===========================================================================


class TestSharpeRatioConsistency:
    def test_backtest_sharpe_is_annualized(self) -> None:
        """historical.py's Sharpe should be annualized (multiplied by sqrt(252)),
        matching backtesting/metrics.sharpe_ratio.

        Setup: Use small kelly_fraction to keep bet sizes nearly constant, with a
        balanced mix of wins and losses. This creates a scenario where:
        - Unannualized Sharpe ≈ 0.1-0.3 (mean/std of individual P&L values)
        - Annualized Sharpe ≈ 1.5-5.0 (× sqrt(252))
        If the returned Sharpe < 1.0, it's clearly not annualized.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            _write_mock_model(model_path, prob=0.65)
            # Deterministic 60/40 win/loss pattern across 200 samples
            outcomes = [1.0 if i % 5 < 3 else 0.0 for i in range(200)]
            samples = [
                _make_sample(0.40, outcomes[i], float(i)) for i in range(200)
            ]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(
                str(model_path), str(csv_path),
                fee_rate=0.03,
                kelly_fraction=0.01,  # tiny Kelly keeps bankroll ≈ 1.0
            )

        assert result["num_trades"] > 5, "Need enough trades for meaningful Sharpe"
        assert result["sharpe"] != 0.0, "Need non-zero Sharpe to test"

        # With ~60% win rate and small Kelly on a modest edge, the unannualized
        # Sharpe (mean_pnl / std_pnl) is ~0.1-0.3. The annualized value should
        # be 15.87x larger, putting it well above 1.0.
        # If historical.py is NOT annualizing, the returned value will be < 1.0.
        assert abs(result["sharpe"]) > 1.0, (
            f"Sharpe={result['sharpe']:.4f} appears unannualized (expected > 1.0 "
            f"after multiplying by sqrt(252)≈15.87)"
        )


# ===========================================================================
# 2. Fee handling: historical.py should deduct fees on ALL trades
# ===========================================================================


class TestFeeConsistency:
    def test_single_losing_trade_deducts_fee(self) -> None:
        """A single losing trade's P&L should include the fee deduction,
        not just -bet_size. After the fix: pnl = -bet_size * (1 + fee_rate).

        Uses 2 samples with test_fraction=0.50 to isolate one test trade.
        """
        market_price = 0.50
        fee_rate = 0.05
        prob = 0.80

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            _write_mock_model(model_path, prob=prob)
            samples = [
                _make_sample(market_price, 0.0, 0.0),  # train
                _make_sample(market_price, 0.0, 1.0),  # test — loses
            ]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(
                str(model_path), str(csv_path),
                fee_rate=fee_rate, kelly_fraction=0.25, test_fraction=0.50,
            )

        assert result["num_trades"] == 1.0

        # Analytically compute what the P&L SHOULD be with fee on loss:
        # yes_cost = 0.50 + 0.05 = 0.55
        # payout_ratio = 1/0.55 - 1 ≈ 0.8182
        # full_kelly = (0.8182*0.8 - 0.2)/0.8182 ≈ 0.5556
        # bet_frac = 0.5556 * 0.25 = 0.1389
        # bet_size = min(0.1389 * 1.0, 1.0) = 0.1389
        # Correct loss pnl = -bet_size - bet_size * fee_rate
        yes_cost = market_price + fee_rate
        payout_ratio = (1.0 / yes_cost) - 1.0
        full_kelly = kelly_criterion(prob, payout_ratio)
        bet_frac = full_kelly * 0.25
        bet_size = bet_frac  # bankroll = 1.0
        expected_pnl_with_fee = -bet_size - bet_size * fee_rate
        expected_pnl_no_fee = -bet_size  # current buggy behavior

        actual_pnl = result["total_pnl"]

        # The actual P&L should match the fee-inclusive calculation
        assert actual_pnl == pytest.approx(expected_pnl_with_fee, abs=1e-6), (
            f"Losing trade P&L={actual_pnl:.6f} matches -bet_size ({expected_pnl_no_fee:.6f}) "
            f"but should be -bet_size*(1+fee)={expected_pnl_with_fee:.6f}. "
            f"Fee is not being deducted on losses."
        )


# ===========================================================================
# 3. EV threshold: should be >= (inclusive), matching engine.py
# ===========================================================================


class TestFeeDeductionOnWinningTrades:
    def test_single_winning_trade_deducts_fee(self) -> None:
        """A winning trade's P&L should have fee deducted separately, matching
        engine.py: pnl = gross_profit - bet_size * fee_rate.
        """
        market_price = 0.50
        fee_rate = 0.05
        prob = 0.80

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            csv_path = Path(tmpdir) / "data.csv"
            _write_mock_model(model_path, prob=prob)
            samples = [
                _make_sample(market_price, 1.0, 0.0),  # train
                _make_sample(market_price, 1.0, 1.0),  # test — wins
            ]
            _write_test_csv(csv_path, samples)

            result = backtest_from_csv(
                str(model_path), str(csv_path),
                fee_rate=fee_rate, kelly_fraction=0.25, test_fraction=0.50,
            )

        assert result["num_trades"] == 1.0

        # Analytical P&L with fee deducted:
        yes_cost = market_price + fee_rate
        payout_ratio = (1.0 / yes_cost) - 1.0
        full_kelly = kelly_criterion(prob, payout_ratio)
        bet_size = full_kelly * 0.25  # bankroll = 1.0
        expected_pnl = bet_size * payout_ratio - bet_size * fee_rate

        assert result["total_pnl"] == pytest.approx(expected_pnl, abs=1e-6)


# ===========================================================================
# 4. Calibration: val metrics should NOT be reported from data used to fit
# ===========================================================================


class TestCalibrationMethodology:
    def test_isotonic_val_ece_not_zero(self) -> None:
        """If isotonic calibration is fitted on validation data, the reported
        val_ece should NOT be 0.0000 — that indicates the calibrator is evaluated
        on its own training data (overfitting).

        Uses noisy labels so the booster can't achieve perfect calibration.
        """
        from arbiter.training.train import train_model

        rng = np.random.default_rng(42)
        n = 500
        features = rng.standard_normal((n, len(SPEC.names)))
        # Noisy labels: true probability = sigmoid(0.5 * feat[0]), then sample
        true_probs = 1.0 / (1.0 + np.exp(-0.5 * features[:, 0]))
        labels = (rng.random(n) < true_probs).astype(float)
        timestamps = np.arange(n, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pkl")
            metrics = train_model(
                features=features,
                labels=labels,
                timestamps=timestamps,
                output_path=output_path,
            )

        # val_ece reports raw (uncalibrated) booster predictions on val set.
        # With noisy labels and limited data, raw predictions won't be perfectly
        # calibrated. If val_ece ≈ 0, the calibrator is being evaluated on its
        # own training data.
        assert metrics["val_ece"] > 0.005, (
            f"val_ece={metrics['val_ece']:.6f} is suspiciously low — "
            f"calibrator may still be evaluated on its own training data"
        )
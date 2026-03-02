"""Convert Kalshi candlestick data into LightGBM training samples."""

from __future__ import annotations

import csv
import math
import pickle
from datetime import UTC, datetime
from typing import Any

import numpy as np

from arbiter.models.features import SPEC
from arbiter.scoring.kelly import kelly_criterion

NAN = float("nan")


def normalize_kalshi_candle(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert real Kalshi API candlestick format to the internal format.

    Real API: price.close in cents (0-100), yes_bid/yes_ask in cents.
    Internal: yes_price.close as decimal (0-1).
    """
    price_close = raw.get("price", {}).get("close")
    if price_close is None:
        # Fall back to midpoint of yes_bid and yes_ask
        bid = raw.get("yes_bid", {}).get("close", 0)
        ask = raw.get("yes_ask", {}).get("close", 0)
        price_close = (bid + ask) / 2

    yes = price_close / 100.0
    return {
        "end_period_ts": raw["end_period_ts"],
        "yes_price": {"close": yes},
        "volume": raw.get("volume", 0),
    }


def candle_to_sample(
    candle: dict[str, Any],
    *,
    close_time_ts: int | None,
    result: str,
    price_history: list[float] | None = None,
) -> dict[str, float]:
    """Convert one candlestick + market metadata into a training sample dict.

    Args:
        candle: Kalshi candlestick dict with yes_price.close, volume, end_period_ts.
        close_time_ts: Market close time as Unix timestamp (for time_to_expiry).
        result: Market result ("yes" or "no").
        price_history: Prior close prices for lag features (most recent last).

    Returns:
        Dict with all SPEC feature names + "outcome" + "timestamp".
    """
    ts = candle["end_period_ts"]
    yes_price = candle["yes_price"]["close"]
    no_price = 1.0 - yes_price
    volume = candle.get("volume", 0)

    sample: dict[str, float] = {}

    # Market features
    sample["yes_price"] = yes_price
    sample["no_price"] = no_price
    sample["bid_ask_spread"] = NAN
    sample["last_price"] = yes_price
    sample["log_volume_24h"] = math.log1p(volume)
    sample["log_open_interest"] = NAN

    if close_time_ts is not None:
        sample["time_to_expiry_hours"] = max(0.0, (close_time_ts - ts) / 3600.0)
    else:
        sample["time_to_expiry_hours"] = NAN

    sample["overround"] = yes_price + no_price - 1.0

    dt = datetime.fromtimestamp(ts, tz=UTC)
    sample["day_of_week"] = float(dt.weekday())
    sample["hour_of_day"] = float(dt.hour)

    # Cross-platform features (not available from candle data)
    sample["price_discrepancy"] = NAN
    sample["volume_ratio"] = NAN

    # Lag features
    if price_history is not None and len(price_history) >= 2:
        sample["price_delta_1h"] = price_history[-1] - price_history[-2]
        sample["price_delta_24h"] = price_history[-1] - price_history[0]
        sample["price_volatility_24h"] = float(np.std(price_history))
    else:
        sample["price_delta_1h"] = NAN
        sample["price_delta_24h"] = NAN
        sample["price_volatility_24h"] = NAN
    sample["volume_ratio_24h"] = NAN

    # Label + metadata
    sample["outcome"] = 1.0 if result == "yes" else 0.0
    sample["timestamp"] = float(ts)

    return sample


def build_training_samples(
    settled_markets: list[dict[str, Any]],
    candles_by_ticker: dict[str, list[dict[str, Any]]],
    *,
    exclude_hours: float = 24.0,
) -> list[dict[str, float]]:
    """Build training samples from settled markets and their candlestick history.

    Args:
        settled_markets: Raw Kalshi API dicts for settled markets.
        candles_by_ticker: Mapping of ticker → list of candlestick dicts,
            sorted chronologically.
        exclude_hours: Exclude candles within this many hours of close_time
            to prevent price convergence leakage.

    Returns:
        List of sample dicts, each with SPEC feature names + outcome + timestamp.
    """
    samples: list[dict[str, float]] = []

    for market in settled_markets:
        ticker = market.get("ticker")
        result = market.get("result")
        if not ticker or not result:
            continue

        candles = candles_by_ticker.get(ticker, [])
        if not candles:
            continue

        # Parse close_time
        close_time_ts: int | None = None
        close_time_raw = market.get("close_time")
        if close_time_raw is not None:
            dt = datetime.fromisoformat(str(close_time_raw).replace("Z", "+00:00"))
            close_time_ts = int(dt.timestamp())

        # Filter candles outside the exclusion window
        exclude_seconds = exclude_hours * 3600
        qualifying: list[dict[str, Any]] = []
        for candle in candles:
            if close_time_ts is not None:
                time_before_close = close_time_ts - candle["end_period_ts"]
                if time_before_close < exclude_seconds:
                    continue
            qualifying.append(candle)

        # Build samples with lag features from prior candles
        for i, candle in enumerate(qualifying):
            price_history: list[float] | None = None
            if i > 0:
                price_history = [c["yes_price"]["close"] for c in qualifying[:i + 1]]

            sample = candle_to_sample(
                candle,
                close_time_ts=close_time_ts,
                result=result,
                price_history=price_history,
            )
            samples.append(sample)

    return samples


def backtest_from_csv(
    model_path: str,
    csv_path: str,
    *,
    fee_rate: float = 0.03,
    kelly_fraction: float = 0.25,
    test_fraction: float = 0.15,
    ev_threshold: float = 0.0,
) -> dict[str, float]:
    """Backtest EVStrategy on CSV training data using a trained model.

    Loads the model, splits data temporally, predicts on the test set,
    simulates Kelly-sized trades on positive-EV opportunities, and returns
    P&L metrics.

    Args:
        model_path: Path to pickled model (booster + calibrator).
        csv_path: Path to training CSV with SPEC features + outcome + timestamp.
        fee_rate: Execution cost as fraction of contract price.
        kelly_fraction: Fractional Kelly multiplier (e.g., 0.25 = quarter Kelly).
        test_fraction: Fraction of data to use as test set (chronological tail).
        ev_threshold: Minimum EV to trigger a trade.

    Returns:
        Dict with: total_pnl, num_trades, win_rate, max_drawdown, sharpe.
    """
    # Load model
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    booster = bundle["model"]
    calibrator = bundle["calibrator"]

    # Load CSV
    rows: list[list[float]] = []
    outcomes: list[float] = []
    timestamps: list[float] = []
    yes_prices: list[float] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append([float(row.get(name, "nan")) for name in SPEC.names])
            outcomes.append(float(row["outcome"]))
            timestamps.append(float(row.get("timestamp", i)))
            yes_prices.append(float(row.get("yes_price", "nan")))

    if not rows:
        return _empty_metrics()

    features = np.array(rows, dtype=np.float64)
    outcomes_arr = np.array(outcomes, dtype=np.float64)
    timestamps_arr = np.array(timestamps, dtype=np.float64)
    yes_prices_arr = np.array(yes_prices, dtype=np.float64)

    # Temporal split — use only the test fraction
    sorted_idx = np.argsort(timestamps_arr)
    n = len(sorted_idx)
    test_start = n - int(n * test_fraction)

    test_idx = sorted_idx[test_start:]
    test_features = features[test_idx]
    test_outcomes = outcomes_arr[test_idx]
    test_yes_prices = yes_prices_arr[test_idx]

    # Predict calibrated probabilities
    raw_preds = booster.predict(test_features)
    cal_preds = calibrator.predict(raw_preds)

    # Simulate trades
    bankroll = 1.0
    peak_bankroll = 1.0
    max_drawdown = 0.0
    trades: list[float] = []  # per-trade P&L

    for i in range(len(test_idx)):
        prob_yes = float(cal_preds[i])
        market_price = float(test_yes_prices[i])
        outcome = float(test_outcomes[i])

        # Check YES side
        yes_cost = market_price + fee_rate
        if 0 < yes_cost < 1:
            ev_yes = prob_yes - yes_cost
            if ev_yes > ev_threshold:
                payout_ratio = (1.0 / yes_cost) - 1.0
                kelly = kelly_criterion(prob_yes, payout_ratio) * kelly_fraction
                bet_size = min(kelly * bankroll, bankroll)
                if bet_size > 0:
                    pnl = bet_size * payout_ratio if outcome == 1.0 else -bet_size
                    bankroll += pnl
                    trades.append(pnl)
                    peak_bankroll = max(peak_bankroll, bankroll)
                    dd = (peak_bankroll - bankroll) / peak_bankroll
                    drawdown = dd if peak_bankroll > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                    continue  # one trade per sample

        # Check NO side
        no_price = 1.0 - market_price
        no_cost = no_price + fee_rate
        prob_no = 1.0 - prob_yes
        if 0 < no_cost < 1:
            ev_no = prob_no - no_cost
            if ev_no > ev_threshold:
                payout_ratio = (1.0 / no_cost) - 1.0
                kelly = kelly_criterion(prob_no, payout_ratio) * kelly_fraction
                bet_size = min(kelly * bankroll, bankroll)
                if bet_size > 0:
                    pnl = bet_size * payout_ratio if outcome == 0.0 else -bet_size
                    bankroll += pnl
                    trades.append(pnl)
                    peak_bankroll = max(peak_bankroll, bankroll)
                    dd = (peak_bankroll - bankroll) / peak_bankroll
                    drawdown = dd if peak_bankroll > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)

    if not trades:
        return _empty_metrics()

    total_pnl = bankroll - 1.0
    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades)

    # Sharpe: mean(returns) / std(returns)
    mean_ret = float(np.mean(trades))
    std_ret = float(np.std(trades))
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "final_bankroll": bankroll,
        "num_trades": float(len(trades)),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "test_samples": float(len(test_idx)),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "total_pnl": 0.0,
        "final_bankroll": 1.0,
        "num_trades": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "test_samples": 0.0,
    }

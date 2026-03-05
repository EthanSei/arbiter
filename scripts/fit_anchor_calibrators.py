"""Fit per-series Platt scaling calibrators from offline walk-forward backtest.

Trains one logistic regression (Platt scaling) per indicator on historical
anchor_prob → outcome pairs using walk-forward, MAD-winsorized σ (no look-ahead).
Saves the calibrators to models/anchor_calibrators.pkl for use in production
AnchorStrategy.score().

Platt scaling (2 parameters) is far more resistant to overfitting than isotonic
regression on small samples. Falls back to a pooled calibrator when a per-series
sample is too small.

Usage:
    python scripts/fit_anchor_calibrators.py

Output: models/anchor_calibrators.pkl
    dict[str, PlattCalibrator] — keyed by indicator_id (e.g. "KXCPI")
    Load via: build_default_strategies(calibrators_path="models/anchor_calibrators.pkl")
"""

from __future__ import annotations

import asyncio
import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from arbiter.data.indicators import INDICATORS
from arbiter.data.providers.fred import compute_sigma
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient
from arbiter.scoring.anchor import PlattCalibrator, compute_anchor_prob, extract_threshold

DATA_DIR = Path("data/features/fred")
OUTPUT_PATH = Path("models/anchor_calibrators.pkl")

MIN_SAMPLES_PER_SERIES = 30


async def fetch_settled_history() -> dict[str, list[dict[str, Any]]]:
    """Fetch all settled markets for each indicator series from Kalshi."""
    raw: dict[str, list[dict[str, Any]]] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        http = RateLimitedClient(client, rpm=20)
        kalshi = KalshiClient(http=http)
        for indicator_id in INDICATORS:
            markets = await kalshi.fetch_settled(series_ticker=indicator_id)
            raw[indicator_id] = markets
            print(f"  {indicator_id}: {len(markets)} settled markets")
    return raw


def build_records(history: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Walk-forward backtest: compute anchor_prob for each settled contract."""
    records: list[dict[str, Any]] = []
    for indicator_id, markets in history.items():
        if not markets:
            continue
        config = INDICATORS[indicator_id]
        path = DATA_DIR / f"{indicator_id}.json"
        if not path.exists():
            print(f"  No FRED cache for {indicator_id} — run scripts/fetch_fred_data.py first")
            continue

        fred_data = json.loads(path.read_text())
        obs = fred_data["observations"]
        obs_dates = [datetime.strptime(o["date"], "%Y-%m-%d").date() for o in obs]

        for market in markets:
            ticker = market.get("ticker", "")
            result = market.get("result", "")
            if result not in ("yes", "no"):
                continue
            parsed = extract_threshold(ticker)
            threshold = parsed[1] if parsed is not None else None
            if threshold is None:
                continue
            close_raw = market.get("close_time", "")
            if not close_raw:
                continue
            close_date = datetime.fromisoformat(close_raw.replace("Z", "+00:00")).date()
            obs_idx = next(
                (i for i in reversed(range(len(obs_dates))) if obs_dates[i] <= close_date),
                None,
            )
            if obs_idx is None or obs_idx < 2:
                continue
            mu = obs[obs_idx]["consensus"]
            sigma = compute_sigma([o["surprise"] for o in obs[:obs_idx]], config.recency_halflife)
            if sigma <= 0:
                continue
            try:
                anchor_prob = compute_anchor_prob(threshold * config.threshold_scale, mu, sigma)
            except ValueError:
                continue
            records.append(
                {
                    "indicator_id": indicator_id,
                    "anchor_prob": anchor_prob,
                    "outcome": 1.0 if result == "yes" else 0.0,
                }
            )
    return records


def fit_calibrators(records: list[dict[str, Any]]) -> dict[str, PlattCalibrator]:
    """Fit Platt scaling per indicator, falling back to pooled calibrator for small samples."""
    by_indicator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_indicator[r["indicator_id"]].append(r)

    # Fit pooled calibrator on all data as fallback
    all_probs = [r["anchor_prob"] for r in records]
    all_outcomes = [r["outcome"] for r in records]
    pooled = PlattCalibrator().fit(all_probs, all_outcomes)
    slope, intercept = pooled.coef
    print(f"\nPooled calibrator (n={len(records)}): slope={slope:.3f}, intercept={intercept:.3f}")

    calibrators: dict[str, PlattCalibrator] = {}
    header = (
        f"{'Indicator':<22} {'n':>5}  {'slope':>7}  {'intcpt':>7}"
        f"  {'raw Brier':>10}  {'cal Brier':>10}  {'Δ':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for indicator_id, recs in sorted(by_indicator.items()):
        probs = [r["anchor_prob"] for r in recs]
        outcomes = [r["outcome"] for r in recs]

        if len(recs) < MIN_SAMPLES_PER_SERIES:
            cal = pooled
            label = f"{indicator_id} (pooled)"
        else:
            cal = PlattCalibrator().fit(probs, outcomes)
            label = indicator_id

        calibrators[indicator_id] = cal
        s, i = cal.coef

        brier_raw = sum((p - o) ** 2 for p, o in zip(probs, outcomes, strict=True)) / len(probs)
        cal_probs = cal.predict(probs)
        brier_cal = sum((p - o) ** 2 for p, o in zip(cal_probs, outcomes, strict=True)) / len(probs)
        print(
            f"  {label:<20} {len(recs):>5}  {s:>7.3f}  {i:>7.3f}  "
            f"{brier_raw:>10.4f}  {brier_cal:>10.4f}  {brier_cal - brier_raw:>+8.4f}"
        )

    return calibrators


async def main() -> None:
    load_dotenv()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching settled history from Kalshi...")
    history = await fetch_settled_history()

    print("\nBuilding walk-forward records...")
    records = build_records(history)
    print(f"  {len(records)} records with FRED anchor data")

    if not records:
        print("No records — check FRED cache and Kalshi settled history")
        return

    print("\nFitting Platt scaling calibrators:")
    calibrators = fit_calibrators(records)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(calibrators, f)

    print(f"\nSaved {len(calibrators)} calibrators → {OUTPUT_PATH}")

    # Show what the calibrators do at key probability levels
    print("\nCalibration curves (raw → calibrated):")
    test_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    header = f"  {'raw':>5}  " + "  ".join(f"{k:>12}" for k in sorted(calibrators))
    print(header)
    for p in test_probs:
        vals = "  ".join(f"{calibrators[k].predict([p])[0]:>12.4f}" for k in sorted(calibrators))
        print(f"  {p:>5.1f}  {vals}")


if __name__ == "__main__":
    asyncio.run(main())

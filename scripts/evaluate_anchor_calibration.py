"""Offline calibration evaluation for the FRED anchor probability model.

Walk-forward backtest: for each historical observation, compute σ from *prior*
surprises only (no look-ahead), predict P(actual > threshold) for thresholds at
μ ± 0/1/2/3σ, compare to the actual binary outcome, and report calibration.

Usage:
    python scripts/evaluate_anchor_calibration.py           # All indicators
    python scripts/evaluate_anchor_calibration.py KXCPIYOY  # One indicator
    python scripts/evaluate_anchor_calibration.py --live    # + live DB predictions
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

from scipy.stats import norm

from arbiter.data.indicators import INDICATORS

DATA_DIR = Path("data/features/fred")
MIN_HISTORY = 24  # minimum prior observations before evaluating
THRESHOLD_OFFSETS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]  # multiples of σ
NUM_BINS = 10


# ---------------------------------------------------------------------------
# Core calibration math
# ---------------------------------------------------------------------------


def _sigma_from_surprises(surprises: list[float]) -> float:
    """Population std of surprises. Returns 0.0 if fewer than 2 values."""
    n = len(surprises)
    if n < 2:
        return 0.0
    mean = sum(surprises) / n
    variance = sum((s - mean) ** 2 for s in surprises) / n
    return math.sqrt(variance)


def _skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return float("nan")
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    return m3 / (m2**1.5) if m2 > 0 else float("nan")


def _kurtosis_excess(values: list[float]) -> float:
    """Excess kurtosis (0 = normal)."""
    n = len(values)
    if n < 4:
        return float("nan")
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    return (m4 / (m2**2) - 3.0) if m2 > 0 else float("nan")


def evaluate_indicator(indicator_id: str, observations: list[dict]) -> dict:
    """Walk-forward calibration for one indicator. Returns summary dict."""
    surprises_all = [obs["surprise"] for obs in observations]

    # --- Distribution stats (full history) ---
    sigma_full = _sigma_from_surprises(surprises_all)
    mean_surprise = sum(surprises_all) / len(surprises_all) if surprises_all else 0.0
    skew = _skewness(surprises_all)
    kurt = _kurtosis_excess(surprises_all)
    # Fraction of surprises within ±1σ and ±2σ (should be ~68% / ~95% if Gaussian)
    pct_1sigma = sum(1 for s in surprises_all if abs(s) <= sigma_full) / len(surprises_all)
    pct_2sigma = sum(1 for s in surprises_all if abs(s) <= 2 * sigma_full) / len(surprises_all)

    # --- Walk-forward calibration ---
    predictions: list[tuple[float, float]] = []  # (predicted_prob, actual_outcome)

    for i in range(MIN_HISTORY, len(observations)):
        obs = observations[i]
        prior_surprises = surprises_all[:i]
        sigma_walk = _sigma_from_surprises(prior_surprises)
        if sigma_walk <= 0:
            continue

        mu = obs["consensus"]
        actual = obs["actual"]

        for offset in THRESHOLD_OFFSETS:
            threshold = mu + offset * sigma_walk
            predicted_prob = float(norm.sf(threshold, loc=mu, scale=sigma_walk))
            actual_outcome = 1.0 if actual > threshold else 0.0
            predictions.append((predicted_prob, actual_outcome))

    # --- Brier score ---
    n_pred = len(predictions)
    brier = sum((p - a) ** 2 for p, a in predictions) / n_pred if n_pred else float("nan")

    # --- Calibration bins ---
    bins: list[list[float]] = [[] for _ in range(NUM_BINS)]
    for p, a in predictions:
        idx = min(int(p * NUM_BINS), NUM_BINS - 1)
        bins[idx].append(a)

    return {
        "n_obs": len(observations),
        "n_predictions": len(predictions),
        "mean_surprise": mean_surprise,
        "sigma_full": sigma_full,
        "skewness": skew,
        "kurtosis_excess": kurt,
        "pct_1sigma": pct_1sigma,
        "pct_2sigma": pct_2sigma,
        "brier_score": brier,
        "bins": bins,
    }


def print_indicator_report(indicator_id: str, config, result: dict) -> None:
    """Print a formatted calibration report for one indicator."""
    sep = "-" * 60
    print(f"\n{'=' * 60}")
    print(f"  {indicator_id}  ({config.fred_series}, transform={config.transform})")
    print(f"{'=' * 60}")
    print(f"  Observations : {result['n_obs']}")
    print(f"  Walk-forward : {result['n_predictions']} predictions (from obs #{MIN_HISTORY}+)")
    print(f"  σ (full)     : {result['sigma_full']:.6g}")
    print(f"  Mean surprise: {result['mean_surprise']:.6g}  (want ≈ 0 = unbiased consensus)")
    print(f"  Skewness     : {result['skewness']:.3f}  (|<0.5| = approx normal)")
    print(f"  Kurt excess  : {result['kurtosis_excess']:.3f}  (0 = normal, >0 = fat tails)")
    print(f"  Within ±1σ   : {result['pct_1sigma']:.1%}  (Gaussian expects 68.3%)")
    print(f"  Within ±2σ   : {result['pct_2sigma']:.1%}  (Gaussian expects 95.4%)")
    print(f"  Brier score  : {result['brier_score']:.4f}  (naive=0.25, perfect=0.00)")

    print(f"\n  {sep}")
    print("  Calibration  (predicted bin | actual freq | n | error)")
    print(f"  {sep}")
    bins = result["bins"]
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        predicted_mid = (i + 0.5) / NUM_BINS
        actual_freq = sum(bucket) / len(bucket)
        error = actual_freq - predicted_mid
        bar = "+" * int(abs(error) * 40) if error > 0 else "-" * int(abs(error) * 40)
        print(
            f"  [{predicted_mid:.2f}] actual={actual_freq:.3f}  n={len(bucket):4d}  "
            f"err={error:+.3f}  {bar}"
        )


# ---------------------------------------------------------------------------
# Live market predictions (requires DB)
# ---------------------------------------------------------------------------


async def _fetch_live_predictions(indicator_ids: list[str]) -> None:
    """Query DB for most recent T-suffix snapshots and show anchor predictions."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        from sqlalchemy import desc, select

        from arbiter.data.providers.fred import FREDSurpriseProvider
        from arbiter.db.models import MarketSnapshot
        from arbiter.db.session import async_session_factory
        from arbiter.scoring.anchor import compute_anchor_prob, extract_threshold
    except ImportError as e:
        print(f"\n[live] Import error: {e}")
        return

    provider = FREDSurpriseProvider(data_dir=str(DATA_DIR))

    print(f"\n{'=' * 60}")
    print("  LIVE MARKET PREDICTIONS  (most recent snapshot per contract)")
    print(f"{'=' * 60}")

    async with async_session_factory() as session:
        for indicator_id in indicator_ids:
            fs = provider.load(indicator_id)
            if fs is None or fs.anchor_mu is None or fs.anchor_sigma is None:
                print(f"\n  {indicator_id}: no anchor params (cache missing or sigma=0)")
                continue
            if fs.anchor_sigma <= 0:
                print(f"\n  {indicator_id}: sigma={fs.anchor_sigma:.6g} <= 0, skip")
                continue

            mu = fs.anchor_mu
            sigma = fs.anchor_sigma

            # Get distinct contract_ids matching this indicator
            pattern = f"{indicator_id}-%T%"
            rows = (
                await session.execute(
                    select(MarketSnapshot.contract_id, MarketSnapshot.features)
                    .where(MarketSnapshot.contract_id.like(pattern))
                    .order_by(MarketSnapshot.contract_id, desc(MarketSnapshot.snapshot_at))
                    .distinct(MarketSnapshot.contract_id)
                )
            ).all()

            if not rows:
                print(f"\n  {indicator_id}: no snapshots in DB")
                continue

            print(f"\n  {indicator_id}  μ={mu:.6g}  σ={sigma:.6g}")
            print(f"  {'Contract':<40} {'Market':>7} {'Anchor':>7} {'EV':>7}")
            print(f"  {'-' * 63}")

            for contract_id, features in sorted(rows, key=lambda r: r[0]):
                parsed = extract_threshold(contract_id)
                if parsed is None:
                    continue
                _, threshold = parsed
                anchor_prob = compute_anchor_prob(threshold, mu, sigma)
                yes_price = float((features or {}).get("yes_price", 0.5))
                ev = anchor_prob - yes_price - 0.01  # fee_rate=0.01
                flag = " ◄ MISPRICED" if ev > 0.03 else ""
                print(
                    f"  {contract_id:<40} {yes_price:>7.3f} {anchor_prob:>7.3f} {ev:>+7.3f}{flag}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = sys.argv[1:]
    run_live = "--live" in args
    args = [a for a in args if not a.startswith("--")]

    target_ids = args if args else list(INDICATORS.keys())

    for indicator_id in target_ids:
        config = INDICATORS.get(indicator_id)
        if config is None or "fred" not in config.providers:
            print(f"[skip] {indicator_id}: not a FRED indicator")
            continue

        path = DATA_DIR / f"{indicator_id}.json"
        if not path.exists():
            print(f"[skip] {indicator_id}: no cache at {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        observations = data.get("observations", [])
        min_needed = MIN_HISTORY + 5
        if len(observations) < min_needed:
            n = len(observations)
            print(f"[skip] {indicator_id}: only {n} observations (need {min_needed}+)")
            continue

        result = evaluate_indicator(indicator_id, observations)
        print_indicator_report(indicator_id, config, result)

    if run_live:
        asyncio.run(_fetch_live_predictions(target_ids))

    print()


if __name__ == "__main__":
    main()

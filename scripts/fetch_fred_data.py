"""Fetch FRED economic data and compute surprise history for anchor pricing.

Usage:
    python scripts/fetch_fred_data.py                    # Fetch all indicators
    python scripts/fetch_fred_data.py KXJOBLESSCLAIMS    # Fetch one indicator

Requires: FRED_API_KEY env var (free at https://fred.stlouisfed.org/docs/api/api_key.html)
Install:  pip install 'arbiter[data]'  (fredapi)

Output: data/features/fred/{indicator}.json with schema:
    {
        "series": "ICSA",
        "transform": "level",
        "consensus_method": "moving_average_4w",
        "current_consensus": 220000.0,
        "observations": [
            {"date": "2024-01-04", "actual": 218000, "consensus": 220000, "surprise": -2000},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fredapi import Fred

from arbiter.data.indicators import INDICATORS, IndicatorConfig

DATA_DIR = Path("data/features/fred")


def fetch_and_cache(indicator_id: str, config: IndicatorConfig, fred: Fred) -> None:
    """Fetch FRED series, compute surprises, and write JSON cache."""
    print(f"Fetching {config.fred_series} for {indicator_id}...")
    series = fred.get_series(config.fred_series)
    series = series.dropna()

    if len(series) < 5:
        print(f"  Skipping {indicator_id}: only {len(series)} observations")
        return

    # Apply transform
    if config.transform == "mom_pct":
        values = series.pct_change().dropna()
    elif config.transform == "mom_change":
        values = series.diff().dropna()
    else:  # "level"
        values = series

    # Compute naive consensus
    if config.consensus_method == "moving_average_4w":
        consensus = values.rolling(4, min_periods=1).mean().shift(1)
    else:  # "prior_value"
        consensus = values.shift(1)

    # Drop rows where consensus is NaN (first row after shift)
    mask = consensus.notna()
    values = values[mask]
    consensus = consensus[mask]
    surprises = values - consensus

    observations = []
    for date, actual, cons, surp in zip(
        values.index, values.values, consensus.values, surprises.values, strict=True
    ):
        observations.append({
            "date": date.strftime("%Y-%m-%d"),
            "actual": float(actual),
            "consensus": float(cons),
            "surprise": float(surp),
        })

    current_consensus = float(consensus.iloc[-1]) if len(consensus) > 0 else None

    output = {
        "series": config.fred_series,
        "transform": config.transform,
        "consensus_method": config.consensus_method,
        "current_consensus": current_consensus,
        "observations": observations,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{indicator_id}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {len(observations)} observations to {path}")


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("Error: FRED_API_KEY env var not set")
        print("Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    fred = Fred(api_key=api_key)

    # Filter to requested indicator or all
    target = sys.argv[1] if len(sys.argv) > 1 else None
    for indicator_id, config in INDICATORS.items():
        if "fred" not in config.providers:
            continue
        if target and indicator_id != target:
            continue
        fetch_and_cache(indicator_id, config, fred)

    print("Done.")


if __name__ == "__main__":
    main()

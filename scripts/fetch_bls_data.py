"""Fetch BLS CPI sub-component data for supplementary features.

Usage:
    python scripts/fetch_bls_data.py                 # Fetch for all BLS-enabled indicators
    python scripts/fetch_bls_data.py KXCPI           # Fetch for one indicator

Requires: BLS_API_KEY env var (free at https://www.bls.gov/developers/home.htm)
Install:  pip install 'arbiter[data]'  (requests)

Output: data/features/bls/{indicator}.json with schema:
    {
        "components": {
            "shelter_cpi_mom": 0.40,
            "food_cpi_mom": 0.25,
            "energy_cpi_mom": -1.2,
            "core_cpi_mom": 0.28
        }
    }
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from arbiter.data.indicators import INDICATORS

DATA_DIR = Path("data/features/bls")

# BLS series IDs for CPI sub-components
CPI_COMPONENTS = {
    "food_cpi_mom": "CUSR0000SAF1",       # Food
    "energy_cpi_mom": "CUSR0000SA0E",      # Energy
    "shelter_cpi_mom": "CUSR0000SASLE",    # Shelter
    "core_cpi_mom": "CUSR0000SA0L1E",      # Core (all items less food & energy)
}

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def fetch_bls_series(series_ids: list[str], api_key: str) -> dict[str, list[dict]]:
    """Fetch multiple BLS series via v2 API."""
    now = datetime.now()
    payload = {
        "seriesid": series_ids,
        "startyear": str(now.year - 2),
        "endyear": str(now.year),
        "registrationkey": api_key,
        "calculations": True,
    }
    resp = requests.post(BLS_API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        print(f"BLS API error: {data.get('message', 'unknown')}")
        sys.exit(1)

    result = {}
    for series in data.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        result[sid] = series.get("data", [])
    return result


def compute_mom_pct(observations: list[dict]) -> float | None:
    """Compute the most recent month-over-month percent change from BLS data.

    BLS returns data in reverse chronological order. The 'calculations' field
    contains pct_changes when requested.
    """
    if not observations:
        return None

    # BLS includes calculations.pct_changes.1 for 1-month percent change
    latest = observations[0]
    calcs = latest.get("calculations", {})
    pct_1m = calcs.get("pct_changes", {}).get("1")
    if pct_1m is not None:
        return float(pct_1m)

    # Fallback: compute manually from latest two values
    if len(observations) < 2:
        return None
    curr = float(observations[0]["value"])
    prev = float(observations[1]["value"])
    if prev == 0:
        return None
    return ((curr - prev) / prev) * 100.0


def fetch_and_cache(indicator_id: str, api_key: str) -> None:
    """Fetch BLS CPI components and write JSON cache."""
    print(f"Fetching BLS CPI components for {indicator_id}...")

    series_ids = list(CPI_COMPONENTS.values())
    raw = fetch_bls_series(series_ids, api_key)

    components: dict[str, float] = {}
    for name, sid in CPI_COMPONENTS.items():
        obs = raw.get(sid, [])
        mom = compute_mom_pct(obs)
        if mom is not None:
            components[name] = round(mom, 4)
            print(f"  {name}: {mom:.4f}%")
        else:
            print(f"  {name}: no data")

    output = {"components": components}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{indicator_id}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote to {path}")


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("BLS_API_KEY")
    if not api_key:
        print("Error: BLS_API_KEY env var not set")
        print("Get a free key at https://www.bls.gov/developers/home.htm")
        sys.exit(1)

    target = sys.argv[1] if len(sys.argv) > 1 else None
    for indicator_id, config in INDICATORS.items():
        if "bls" not in config.providers:
            continue
        if target and indicator_id != target:
            continue
        fetch_and_cache(indicator_id, api_key)

    print("Done.")


if __name__ == "__main__":
    main()

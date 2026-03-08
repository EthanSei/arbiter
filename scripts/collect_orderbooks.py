"""Collect order book snapshots for target series to parquet.

Fetches order book depth for the top-N contracts (by volume) in each target
series and saves to data/orderbooks.parquet (appending to existing data).

Usage:
    python scripts/collect_orderbooks.py              # All target series
    python scripts/collect_orderbooks.py KXCPI KXBTCD  # Specific series only

Rate limit: Uses data_collection_rpm from settings (default 5 RPM).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from arbiter.config import settings
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "orderbooks.parquet"


def _append_parquet(rows: list[dict], path: Path) -> None:
    """Append rows to a parquet file, creating it if it doesn't exist."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(path, index=False)


async def main() -> None:
    load_dotenv()

    all_series = [s.strip() for s in settings.kalshi_target_series.split(",") if s.strip()]

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
        unknown = [s for s in targets if s not in all_series]
        if unknown:
            print(f"Warning: {unknown} not in kalshi_target_series, fetching anyway")
        series_list = targets
    else:
        series_list = all_series

    top_n = settings.orderbook_top_n
    rows: list[dict] = []
    now = datetime.now(UTC)

    async with httpx.AsyncClient(timeout=60.0) as http:
        rl = RateLimitedClient(http, rpm=settings.data_collection_rpm)

        for series in series_list:
            print(f"  {series}: fetching contracts...")
            kalshi = KalshiClient(rl, base_url=settings.kalshi_api_base, series_tickers=[series])
            contracts = await kalshi.fetch_markets()
            kalshi_contracts = [c for c in contracts if c.source == "kalshi"]
            if not kalshi_contracts:
                print(f"  {series}: no open contracts, skipping")
                continue

            top = sorted(kalshi_contracts, key=lambda c: c.volume_24h, reverse=True)[:top_n]
            collected = 0
            for contract in top:
                try:
                    ob = await kalshi.fetch_orderbook(contract.contract_id)
                except Exception as exc:
                    print(f"    {contract.contract_id}: failed ({exc})")
                    continue
                rows.append({
                    "source": "kalshi",
                    "contract_id": contract.contract_id,
                    "series_ticker": contract.series_ticker,
                    "event_ticker": contract.event_ticker,
                    "bids": ob["bids"],
                    "asks": ob["asks"],
                    "snapshot_at": now,
                })
                collected += 1
            print(f"  {series}: {collected} order book snapshots")

    _append_parquet(rows, PARQUET_PATH)
    print(f"\nDone. {len(rows)} snapshots written to {PARQUET_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

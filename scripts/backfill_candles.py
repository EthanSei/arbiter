"""Backfill hourly candlestick bars for target series to parquet.

Fetches OHLCV candles from the Kalshi API in 7-day windows (to stay within
API limits) for all contracts in each target series. Saves to data/candles.parquet.

Usage:
    python scripts/backfill_candles.py              # All target series
    python scripts/backfill_candles.py KXCPI KXBTCD  # Specific series only

Rate limit: Uses data_collection_rpm from settings (default 5 RPM).
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from arbiter.config import settings
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "candles.parquet"

DAYS = 90
PERIOD_INTERVAL = 60  # 1-hour bars
WINDOW_DAYS = 7  # Kalshi API max range per request


def _append_parquet(rows: list[dict], path: Path) -> None:
    """Append rows to a parquet file, deduping on (contract_id, period_start, period_interval)."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(
            subset=["contract_id", "period_start", "period_interval"], keep="last"
        )
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

    now = int(time.time())
    total_start = now - DAYS * 86400
    interval_seconds = PERIOD_INTERVAL * 60
    window_seconds = WINDOW_DAYS * 86400
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=60.0) as http:
        rl = RateLimitedClient(http, rpm=settings.data_collection_rpm)

        for series in series_list:
            print(f"  {series}: fetching contracts...")
            kalshi = KalshiClient(rl, base_url=settings.kalshi_api_base, series_tickers=[series])
            contracts = await kalshi.fetch_markets()
            tickers = [c.contract_id for c in contracts]
            if not tickers:
                print(f"  {series}: no open contracts, skipping")
                continue

            # Fetch in 7-day windows to stay within Kalshi API limits
            count = 0
            window_start = total_start
            while window_start < now:
                window_end = min(window_start + window_seconds, now)
                batch = await kalshi.fetch_candlesticks_batch(
                    tickers,
                    start_ts=window_start,
                    end_ts=window_end,
                    period_interval=PERIOD_INTERVAL,
                )

                for ticker, candles in batch.items():
                    for candle in candles:
                        price = candle.get("price", {})
                        end_ts_val = candle["end_period_ts"]
                        period_start = datetime.fromtimestamp(
                            end_ts_val - interval_seconds, tz=UTC
                        )
                        rows.append({
                            "source": "kalshi",
                            "contract_id": ticker,
                            "series_ticker": series,
                            "period_start": period_start,
                            "period_interval": PERIOD_INTERVAL,
                            "open": (price.get("open") or 0) / 100.0,
                            "high": (price.get("high") or 0) / 100.0,
                            "low": (price.get("low") or 0) / 100.0,
                            "close": (price.get("close") or 0) / 100.0,
                            "volume": candle.get("volume"),
                        })
                        count += 1

                window_start = window_end

            print(f"  {series}: {count} bars")

    _append_parquet(rows, PARQUET_PATH)
    print(f"\nDone. {len(rows)} bars written to {PARQUET_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

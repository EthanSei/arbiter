"""Collect training data from Kalshi settled markets + candlestick history.

Fetches settled markets for configured series, retrieves hourly candlestick
data for each, converts to training samples, and writes CSV.

Run with:
    python scripts/collect_training_data.py
    python scripts/collect_training_data.py --months 6 --output data/training.csv
    python scripts/collect_training_data.py --series KXCPI KXPAYROLLS
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import csv
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient
from arbiter.models.features import SPEC
from arbiter.training.historical import build_training_samples, normalize_kalshi_candle

TIER1_SERIES = ["KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXCPICOREYOY", "KXJOBLESSCLAIMS"]


def _months_ago_ts(months: int) -> int:
    now = datetime.now(UTC)
    month = now.month - months
    year = now.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(now.day, calendar.monthrange(year, month)[1])
    dt = now.replace(year=year, month=month, day=day)
    return int(dt.timestamp())


async def fetch_candles_for_market(
    http: httpx.AsyncClient,
    ticker: str,
    start_ts: int,
    end_ts: int,
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
) -> list[dict[str, Any]]:
    """Fetch hourly candlesticks for a single market via the batch endpoint."""
    url = f"{base_url}/markets/candlesticks"
    resp = await http.get(
        url,
        params={
            "market_tickers": ticker,
            "period_interval": 60,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets", [])
    if not markets:
        return []
    return markets[0].get("candlesticks", [])


async def run(args: argparse.Namespace) -> None:
    start_ts = _months_ago_ts(args.months)
    now_ts = int(time.time())
    series_tickers = args.series or TIER1_SERIES

    http = httpx.AsyncClient(timeout=30.0)
    rl = RateLimitedClient(http, rpm=20)
    client = KalshiClient(rl, min_volume_24h=0.0)

    all_settled: list[dict[str, Any]] = []
    all_candles: dict[str, list[dict[str, Any]]] = {}

    try:
        # 1. Fetch settled markets for each series
        for si, series_ticker in enumerate(series_tickers):
            print(f"\n[{si + 1}/{len(series_tickers)}] Fetching settled markets: {series_ticker}")
            t0 = time.monotonic()
            raw_markets = await client.fetch_settled(
                series_ticker=series_ticker,
                min_close_ts=start_ts,
            )
            elapsed = time.monotonic() - t0
            print(f"  Found {len(raw_markets)} settled markets ({elapsed:.1f}s)")
            all_settled.extend(raw_markets)

        print(f"\nTotal settled markets: {len(all_settled)}")

        # 2. Fetch candlesticks for each market
        tickers = [m["ticker"] for m in all_settled if "ticker" in m]
        print(f"Fetching candlesticks for {len(tickers)} markets...")

        for i, ticker in enumerate(tickers):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i + 1}/{len(tickers)}] {ticker}")
            try:
                raw_candles = await fetch_candles_for_market(rl, ticker, start_ts, now_ts)
                if raw_candles:
                    # Normalize from real API format to internal format
                    normalized = [normalize_kalshi_candle(c) for c in raw_candles]
                    all_candles[ticker] = normalized
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    continue  # Market may have been removed
                raise

        print(f"  Got candlesticks for {len(all_candles)} markets")

        # 3. Build training samples
        samples = build_training_samples(
            all_settled,
            all_candles,
            exclude_hours=args.exclude_hours,
        )
        print(f"\nGenerated {len(samples)} training samples")

        if not samples:
            print("No samples generated. Check API responses.")
            return

        # 4. Write CSV
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = SPEC.names + ["outcome", "timestamp"]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        print(f"Wrote {len(samples)} samples to {output}")

    finally:
        await http.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect training data from Kalshi")
    parser.add_argument("--months", type=int, default=6, help="Look back N months (default: 6)")
    parser.add_argument("--output", type=str, default="data/training.csv", help="Output CSV path")
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help=f"Series tickers (default: {', '.join(TIER1_SERIES)})",
    )
    parser.add_argument(
        "--exclude-hours",
        type=float,
        default=24.0,
        help="Exclude candles within N hours of close (default: 24)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

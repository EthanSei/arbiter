"""Fetch pre-release candlestick prices for settled economic indicator contracts.

For each settled indicator contract, fetches the daily OHLC data for the 7-day window
ending 7 days before close_time. The last candle's close price is the tradeable entry
price — before the result is known, so it is NOT contaminated by the outcome.

This unlocks real P&L simulation in the offline backtest (Step 3 in anchor_backtest.ipynb)
by replacing `last_price_dollars` (post-release, contaminated) with a genuine pre-release
entry price.

Output: data/features/candlesticks/{indicator_id}.json
Schema:
    {
        "indicator_id": "KXPAYROLLS",
        "pre_release_prices": {
            "KXPAYROLLS-26FEB-T200": {
                "entry_price": 0.63,
                "entry_ts": 1741132800,
                "candle_count": 7
            },
            ...
        }
    }

Usage:
    python scripts/fetch_candlestick_prices.py                # All indicators
    python scripts/fetch_candlestick_prices.py KXPAYROLLS     # Single indicator

Rate limit: Kalshi public API is 20 RPM — handled by RateLimitedClient at 20 RPM.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from arbiter.data.indicators import INDICATORS
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient

DATA_DIR = Path("data/features/candlesticks")
# 7 days before close as the pre-release window end
PRE_RELEASE_DAYS = 7
# Fetch 14-day window ending PRE_RELEASE_DAYS before close (gives ~7 daily candles)
LOOKBACK_DAYS = 14
_BATCH_SIZE = 100
# period_interval=1440 → daily candles (1440 minutes)
_PERIOD_INTERVAL = 1440


def _last_close_price(candles: list[dict[str, Any]]) -> float | None:
    """Extract the yes close price from the last candle.

    Kalshi candle price fields are in cents (0-100); divide by 100 to get
    the decimal probability used everywhere else in arbiter.
    """
    if not candles:
        return None
    last = candles[-1]
    price_block = last.get("price") or last
    close_cents = price_block.get("close") or price_block.get("yes_price")
    if close_cents is None:
        return None
    return float(close_cents) / 100.0


async def process_indicator(
    client: RateLimitedClient,
    indicator_id: str,
) -> dict[str, dict[str, Any]]:
    """Fetch pre-release prices for all settled contracts in one indicator series."""
    kalshi = KalshiClient(http=client)
    print(f"\n{indicator_id}: fetching settled markets...")
    markets = await kalshi.fetch_settled(series_ticker=indicator_id)
    print(f"  {len(markets)} settled markets with results")

    if not markets:
        return {}

    # Group tickers by their close_time so we can batch calls for the same window
    # (most monthly-release series settle on the same day)
    by_close: dict[int, list[dict[str, Any]]] = {}
    for m in markets:
        close_raw = m.get("close_time", "")
        try:
            close_dt = datetime.fromisoformat(close_raw.replace("Z", "+00:00"))
            close_ts = int(close_dt.timestamp())
        except (ValueError, TypeError):
            continue
        by_close.setdefault(close_ts, []).append(m)

    pre_release_prices: dict[str, dict[str, Any]] = {}

    for close_ts, group in sorted(by_close.items()):
        end_ts = close_ts - PRE_RELEASE_DAYS * 86400
        start_ts = end_ts - LOOKBACK_DAYS * 86400
        tickers = [m["ticker"] for m in group]

        # Batch in chunks of _BATCH_SIZE
        for i in range(0, len(tickers), _BATCH_SIZE):
            chunk = tickers[i : i + _BATCH_SIZE]
            try:
                candle_data = await kalshi.fetch_candlesticks_batch(
                    chunk,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    period_interval=_PERIOD_INTERVAL,
                )
            except httpx.HTTPStatusError as exc:
                print(f"  HTTP {exc.response.status_code} for batch at close_ts={close_ts}")
                candle_data = {}

            for ticker in chunk:
                candles = candle_data.get(ticker, [])
                entry_price = _last_close_price(candles)
                if entry_price is not None:
                    pre_release_prices[ticker] = {
                        "entry_price": round(entry_price, 4),
                        "entry_ts": end_ts,
                        "candle_count": len(candles),
                    }

    print(f"  {len(pre_release_prices)} of {len(markets)} tickers have pre-release prices")
    return pre_release_prices


async def main() -> None:
    load_dotenv()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target not in INDICATORS:
        print(f"Error: unknown indicator '{target}'. Choose from: {list(INDICATORS)}")
        sys.exit(1)

    indicators = [target] if target else list(INDICATORS)

    async with httpx.AsyncClient(timeout=60.0) as http:
        client = RateLimitedClient(http, rpm=20)
        for indicator_id in indicators:
            pre_release = await process_indicator(client, indicator_id)

            output = {"indicator_id": indicator_id, "pre_release_prices": pre_release}
            path = DATA_DIR / f"{indicator_id}.json"
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Wrote {path}")

    print("\nDone. Load in notebook with:")
    print("  import json; d = json.load(open('data/features/candlesticks/KXPAYROLLS.json'))")
    print("  entry = d['pre_release_prices']['KXPAYROLLS-26FEB-T200']['entry_price']")


if __name__ == "__main__":
    asyncio.run(main())

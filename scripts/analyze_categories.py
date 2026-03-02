"""Analyze Kalshi settled markets by series to identify arb opportunity hotspots.

Fetches historical settled market data from the Kalshi API for specified series,
computes calibration metrics, mid-range density, surprise rates, and bracket
structure, then prints a ranked summary.

Run with:
    python scripts/analyze_categories.py                     # default Tier 1 series
    python scripts/analyze_categories.py KXCPI KXPAYROLLS    # specific series
    python scripts/analyze_categories.py --category Economics # all series in category
    python scripts/analyze_categories.py --months 2          # last 2 months (default)
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import time
from datetime import UTC, datetime

import httpx

from arbiter.analysis.historical import (
    SeriesAnalysis,
    analyze_series,
    parse_settled_market,
)
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient

# Tier 1 Economics bracket series — best arb targets from research
DEFAULT_SERIES = [
    "KXCPI",
    "KXCPIYOY",
    "KXPAYROLLS",
    "KXINX",
    "KXNASDAQ100",
    "KXTNOTEW",
    "KXAAAGASM",
    "KXFEDDECISION",
]


def _months_ago_ts(months: int) -> int:
    """Return a Unix timestamp for N months ago from now."""
    now = datetime.now(UTC)
    month = now.month - months
    year = now.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(now.day, calendar.monthrange(year, month)[1])
    dt = now.replace(year=year, month=month, day=day)
    return int(dt.timestamp())


async def fetch_series_data(
    client: KalshiClient,
    series_ticker: str,
    min_close_ts: int | None = None,
) -> SeriesAnalysis:
    """Fetch settled markets for a series and return analysis."""
    raw_markets = await client.fetch_settled(
        series_ticker=series_ticker,
        min_close_ts=min_close_ts,
    )
    parsed = []
    for raw in raw_markets:
        m = parse_settled_market(raw)
        if m is not None:
            parsed.append(m)

    return analyze_series(series_ticker, parsed)


async def discover_series_by_category(
    http: httpx.AsyncClient,
    category: str,
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    min_volume: int = 10_000,
) -> list[str]:
    """Fetch series tickers for a category from the Kalshi API."""
    resp = await http.get(
        f"{base_url}/series",
        params={"category": category, "include_volume": "true"},
    )
    resp.raise_for_status()
    data = resp.json()
    series_list = data.get("series", [])

    # Filter by minimum volume and return tickers
    tickers = []
    for s in series_list:
        vol = s.get("volume", 0) or 0
        if vol >= min_volume:
            tickers.append(s["ticker"])
    return tickers


def print_results(results: list[SeriesAnalysis]) -> None:
    """Print analysis results as a ranked table."""
    # Sort by midrange density (primary) then surprise rate (secondary)
    ranked = sorted(results, key=lambda r: (r.midrange_pct, r.surprise_rate), reverse=True)

    print(f"\n{'=' * 90}")
    print("  CATEGORY ANALYSIS — Arb & Volatility Opportunity Ranking")
    print(f"{'=' * 90}")
    print(
        f"  {'Series':<16} {'Total':>6} {'w/Vol':>6} {'MidRng':>7} {'Mid%':>6} "
        f"{'Surp':>5} {'Surp%':>6} {'Brier':>7} {'MidVol':>10} {'Brkts':>6} "
        f"{'Avg':>5}"
    )
    print(
        f"  {'-' * 16} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 6} "
        f"{'-' * 5} {'-' * 6} {'-' * 7} {'-' * 10} {'-' * 6} {'-' * 5}"
    )

    for r in ranked:
        print(
            f"  {r.series_ticker:<16} {r.total_markets:>6} {r.markets_with_volume:>6} "
            f"{r.midrange_count:>7} {r.midrange_pct:>5.1%} "
            f"{r.surprise_count:>5} {r.surprise_rate:>5.1%} "
            f"{r.brier_score:>7.4f} {r.midrange_volume:>10,} "
            f"{r.bracket_families:>6} {r.avg_bracket_size:>5.1f}"
        )

    print("\n  Legend:")
    print("    MidRng  = markets with last_price 10-90% (where arbs exist)")
    print("    Mid%    = midrange / total with volume")
    print("    Surp    = midrange markets where outcome diverged >30% from price")
    print("    Surp%   = surprise_count / midrange_count")
    print("    Brier   = mean squared error of price vs outcome (lower = better calibrated)")
    print("    MidVol  = total volume in midrange markets")
    print("    Brkts   = distinct event_ticker bracket families")
    print("    Avg     = average markets per bracket family")
    print()


async def run(args: argparse.Namespace) -> None:
    min_close_ts = _months_ago_ts(args.months) if args.months > 0 else None

    http = httpx.AsyncClient(timeout=30.0)
    rl = RateLimitedClient(http, rpm=20)
    client = KalshiClient(rl, min_volume_24h=0.0)

    try:
        # Determine which series to analyze
        if args.category:
            print(f"  Discovering series in category '{args.category}'...")
            series_tickers = await discover_series_by_category(
                http, args.category, min_volume=args.min_volume
            )
            print(f"  Found {len(series_tickers)} series with volume >= {args.min_volume:,}")
        elif args.series:
            series_tickers = args.series
        else:
            series_tickers = DEFAULT_SERIES

        print(f"  Analyzing {len(series_tickers)} series (last {args.months} months)...")
        print(f"  Series: {', '.join(series_tickers)}")

        results: list[SeriesAnalysis] = []
        for i, ticker in enumerate(series_tickers):
            t0 = time.monotonic()
            analysis = await fetch_series_data(client, ticker, min_close_ts=min_close_ts)
            elapsed = time.monotonic() - t0
            results.append(analysis)
            print(
                f"    [{i + 1}/{len(series_tickers)}] {ticker}: "
                f"{analysis.total_markets} markets, "
                f"{analysis.midrange_count} midrange "
                f"({elapsed:.1f}s)"
            )

        print_results(results)

    finally:
        await http.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Kalshi settled markets for arb opportunity hotspots"
    )
    parser.add_argument(
        "series",
        nargs="*",
        help="Series tickers to analyze (default: Tier 1 economics)",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Analyze all series in a Kalshi category (e.g., Economics, Financials)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=2,
        help="Look back N months from now (default: 2)",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=10_000,
        help="Minimum series volume for category discovery (default: 10000)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

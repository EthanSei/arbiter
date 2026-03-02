"""Deep mispricing analysis on Kalshi Economic bracket markets.

Fetches settled markets and their candlestick OHLC data, then analyzes
how often mispricings occur, how long they last, and how large they get.
Uses bracket fair values (1/N for N-contract brackets) as the anchor.

Run with:
    python scripts/analyze_mispricings.py
    python scripts/analyze_mispricings.py --months 3 --threshold 0.05
    python scripts/analyze_mispricings.py --series KXCPI KXPAYROLLS
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from arbiter.analysis.historical import parse_settled_market
from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient

TIER1_SERIES = ["KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXCPICOREYOY", "KXJOBLESSCLAIMS"]


@dataclass(frozen=True)
class Episode:
    """A period where a market price deviated from fair value."""

    ticker: str
    start_ts: int
    end_ts: int
    duration_minutes: int
    peak_deviation: float
    direction: str  # "overpriced" or "underpriced"


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


def find_mispricing_episodes(
    candlesticks: list[dict[str, Any]],
    fair_value: float,
    threshold: float,
    ticker: str,
) -> list[Episode]:
    """Find episodes where price.close deviated from fair_value by > threshold.

    Works with the real Kalshi candlestick format:
        candle["price"]["close"] is an int in cents (0-100)
        candle["end_period_ts"] is Unix timestamp
    """
    episodes: list[Episode] = []
    start_ts: int | None = None
    peak: float = 0.0
    direction: str = ""

    for candle in candlesticks:
        price_raw = candle.get("price", {}).get("close")
        if price_raw is None:
            continue  # no trade in this period

        ts = candle["end_period_ts"]
        price = price_raw / 100.0  # cents to probability
        deviation = price - fair_value

        if abs(deviation) > threshold:
            d = "overpriced" if deviation > 0 else "underpriced"
            if start_ts is None:
                start_ts = ts
                peak = abs(deviation)
                direction = d
            else:
                peak = max(peak, abs(deviation))
            last_ts = ts
        elif start_ts is not None:
            duration = (last_ts - start_ts) // 60
            episodes.append(Episode(ticker, start_ts, last_ts, duration, peak, direction))
            start_ts = None
            peak = 0.0

    # Close open episode
    if start_ts is not None:
        duration = (last_ts - start_ts) // 60  # noqa: F821
        episodes.append(Episode(ticker, start_ts, last_ts, duration, peak, direction))  # noqa: F821

    return episodes


async def fetch_candlesticks_for_market(
    http: httpx.AsyncClient,
    ticker: str,
    start_ts: int,
    end_ts: int,
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
) -> list[dict[str, Any]]:
    """Fetch hourly candlesticks for a single market via the batch endpoint."""
    url = f"{base_url}/markets/candlesticks"
    resp = await http.get(url, params={
        "market_tickers": ticker,
        "period_interval": 60,
        "start_ts": start_ts,
        "end_ts": end_ts,
    })
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets", [])
    if not markets:
        return []
    return markets[0].get("candlesticks", [])


async def run(args: argparse.Namespace) -> None:
    now_ts = int(time.time())
    start_ts = _months_ago_ts(args.months)
    series_tickers = args.series or TIER1_SERIES
    threshold = args.threshold

    http = httpx.AsyncClient(timeout=30.0)
    rl = RateLimitedClient(http, rpm=20)
    client = KalshiClient(rl, min_volume_24h=0.0)

    all_episodes: list[Episode] = []
    series_stats: dict[str, dict[str, Any]] = {}
    total_markets_analyzed = 0
    total_markets_with_candles = 0

    try:
        for si, series_ticker in enumerate(series_tickers):
            print(f"\n{'=' * 70}")
            print(f"  [{si + 1}/{len(series_tickers)}] Series: {series_ticker}")
            print(f"{'=' * 70}")

            # 1. Fetch settled markets
            t0 = time.monotonic()
            raw_markets = await client.fetch_settled(
                series_ticker=series_ticker,
                min_close_ts=start_ts,
            )
            elapsed = time.monotonic() - t0
            print(f"  Fetched {len(raw_markets)} settled markets ({elapsed:.1f}s)")

            parsed = []
            for raw in raw_markets:
                m = parse_settled_market(raw)
                if m is not None:
                    parsed.append(m)

            if not parsed:
                print("  No parsed markets — skipping")
                continue

            # 2. Group by event_ticker (bracket family) to compute fair values
            families: dict[str, list] = defaultdict(list)
            for m in parsed:
                families[m.event_ticker].append(m)

            print(f"  {len(parsed)} markets in {len(families)} bracket families")

            # 3. For each market in midrange, fetch candlesticks and analyze
            series_episodes: list[Episode] = []
            markets_checked = 0
            markets_with_data = 0

            midrange = [m for m in parsed if 0.10 <= m.last_price <= 0.90]
            print(f"  {len(midrange)} midrange markets (10-90% last_price)")

            if not midrange:
                series_stats[series_ticker] = {
                    "total_markets": len(parsed),
                    "midrange": 0,
                    "families": len(families),
                    "episodes": 0,
                }
                continue

            total_markets_analyzed += len(midrange)

            for mi, market in enumerate(midrange):
                family = families.get(market.event_ticker, [market])
                fair_value = 1.0 / len(family) if len(family) > 1 else 0.50

                try:
                    candles = await fetch_candlesticks_for_market(
                        http, market.ticker, start_ts, now_ts,
                    )
                except httpx.HTTPStatusError:
                    continue

                markets_checked += 1
                if not candles:
                    continue

                # Filter to candles with actual trades
                traded = [c for c in candles if c.get("price", {}).get("close") is not None]
                if not traded:
                    continue

                markets_with_data += 1
                total_markets_with_candles += 1

                episodes = find_mispricing_episodes(traded, fair_value, threshold, market.ticker)
                series_episodes.extend(episodes)

                if (mi + 1) % 10 == 0 or mi == len(midrange) - 1:
                    print(
                        f"    Progress: {mi + 1}/{len(midrange)} markets, "
                        f"{len(series_episodes)} episodes found"
                    )

            all_episodes.extend(series_episodes)
            series_stats[series_ticker] = {
                "total_markets": len(parsed),
                "midrange": len(midrange),
                "families": len(families),
                "checked": markets_checked,
                "with_data": markets_with_data,
                "episodes": len(series_episodes),
            }

            if series_episodes:
                durations = [e.duration_minutes for e in series_episodes]
                peaks = [e.peak_deviation for e in series_episodes]
                print(f"\n  Series episodes: {len(series_episodes)}")
                print(f"  Duration (min):  median={statistics.median(durations):.0f}, "
                      f"mean={statistics.mean(durations):.0f}, "
                      f"p25={_percentile(durations, 25):.0f}, "
                      f"p75={_percentile(durations, 75):.0f}, "
                      f"max={max(durations):.0f}")
                print(f"  Peak deviation:  median={statistics.median(peaks):.3f}, "
                      f"mean={statistics.mean(peaks):.3f}, "
                      f"max={max(peaks):.3f}")
            else:
                print(f"\n  No mispricing episodes found (threshold={threshold})")

        # ===== GLOBAL SUMMARY =====
        print(f"\n\n{'#' * 70}")
        print("  GLOBAL MISPRICING ANALYSIS SUMMARY")
        print(f"{'#' * 70}")
        print(f"\n  Series analyzed:     {len(series_tickers)}")
        print(f"  Lookback:            {args.months} months")
        print(f"  Threshold:           {threshold:.0%} deviation from fair value")
        print(f"  Markets analyzed:    {total_markets_analyzed} (midrange 10-90%)")
        print(f"  Markets with data:   {total_markets_with_candles}")
        print(f"  Total episodes:      {len(all_episodes)}")

        if all_episodes:
            durations = [e.duration_minutes for e in all_episodes]
            peaks = [e.peak_deviation for e in all_episodes]
            overpriced = [e for e in all_episodes if e.direction == "overpriced"]
            underpriced = [e for e in all_episodes if e.direction == "underpriced"]

            print("\n  Direction split:")
            print(f"    Overpriced:  {len(overpriced)} ({len(overpriced)/len(all_episodes):.0%})")
            print(f"    Underpriced: {len(underpriced)} ({len(underpriced)/len(all_episodes):.0%})")

            print("\n  Duration distribution (minutes):")
            print(f"    Min:    {min(durations):>8.0f}")
            print(f"    P10:    {_percentile(durations, 10):>8.0f}")
            print(f"    P25:    {_percentile(durations, 25):>8.0f}")
            print(f"    Median: {statistics.median(durations):>8.0f}")
            print(f"    Mean:   {statistics.mean(durations):>8.0f}")
            print(f"    P75:    {_percentile(durations, 75):>8.0f}")
            print(f"    P90:    {_percentile(durations, 90):>8.0f}")
            print(f"    Max:    {max(durations):>8.0f}")

            print("\n  Duration buckets:")
            buckets = [
                ("< 1 hour", 0, 60),
                ("1-4 hours", 60, 240),
                ("4-12 hours", 240, 720),
                ("12-24 hours", 720, 1440),
                ("1-3 days", 1440, 4320),
                ("3-7 days", 4320, 10080),
                ("> 1 week", 10080, float("inf")),
            ]
            for label, lo, hi in buckets:
                count = sum(1 for d in durations if lo <= d < hi)
                pct = count / len(durations) if durations else 0
                bar = "#" * int(pct * 40)
                print(f"    {label:<14} {count:>5} ({pct:>5.1%}) {bar}")

            print("\n  Peak deviation distribution:")
            print(f"    Min:    {min(peaks):>8.3f}")
            print(f"    Median: {statistics.median(peaks):>8.3f}")
            print(f"    Mean:   {statistics.mean(peaks):>8.3f}")
            print(f"    P75:    {_percentile(peaks, 75):>8.3f}")
            print(f"    P90:    {_percentile(peaks, 90):>8.3f}")
            print(f"    Max:    {max(peaks):>8.3f}")

            # Actionability analysis
            print("\n  ACTIONABILITY ANALYSIS (Discord alert viability):")
            actionable_30 = [e for e in all_episodes if e.duration_minutes >= 30]
            actionable_60 = [e for e in all_episodes if e.duration_minutes >= 60]
            large = [e for e in all_episodes if e.peak_deviation >= 0.10]
            actionable_large = [
                e for e in all_episodes
                if e.duration_minutes >= 30 and e.peak_deviation >= 0.10
            ]
            print(f"    Episodes >= 30 min:       {len(actionable_30):>5} "
                  f"({len(actionable_30)/len(all_episodes):.0%})")
            print(f"    Episodes >= 60 min:       {len(actionable_60):>5} "
                  f"({len(actionable_60)/len(all_episodes):.0%})")
            print(f"    Episodes >= 10% dev:      {len(large):>5} "
                  f"({len(large)/len(all_episodes):.0%})")
            print(f"    >= 30 min AND >= 10% dev: {len(actionable_large):>5} "
                  f"({len(actionable_large)/len(all_episodes):.0%})")

            if actionable_large:
                al_durations = [e.duration_minutes for e in actionable_large]
                al_peaks = [e.peak_deviation for e in actionable_large]
                print("\n    Actionable episode stats:")
                print(f"      Duration median: {statistics.median(al_durations):.0f} min, "
                      f"mean: {statistics.mean(al_durations):.0f} min")
                print(f"      Peak dev median: {statistics.median(al_peaks):.3f}, "
                      f"mean: {statistics.mean(al_peaks):.3f}")

            if total_markets_with_candles > 0:
                episodes_per_market = len(all_episodes) / total_markets_with_candles
                print(f"\n    Episodes per market:       {episodes_per_market:.1f}")
                if actionable_large:
                    print(f"    Actionable per market:     "
                          f"{len(actionable_large) / total_markets_with_candles:.1f}")

            # Estimate monthly frequency
            months = args.months or 1
            print("\n    Estimated monthly frequency (across all markets):")
            print(f"      Total episodes/month:      {len(all_episodes) / months:.0f}")
            if actionable_large:
                print(f"      Actionable/month:          {len(actionable_large) / months:.0f}")

        # Per-series breakdown table
        print(f"\n  {'=' * 70}")
        print("  PER-SERIES BREAKDOWN")
        print(f"  {'=' * 70}")
        print(f"  {'Series':<18} {'Markets':>8} {'MidRng':>7} {'Checked':>8} "
              f"{'w/Data':>7} {'Episodes':>9}")
        print(f"  {'-' * 18} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 9}")
        for st, stats in series_stats.items():
            print(f"  {st:<18} {stats['total_markets']:>8} {stats['midrange']:>7} "
                  f"{stats.get('checked', 0):>8} {stats.get('with_data', 0):>7} "
                  f"{stats['episodes']:>9}")

        print()

    finally:
        await http.aclose()


def _percentile(data: list[float | int], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return float(sorted_data[-1])
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep mispricing analysis on Kalshi Economic bracket markets"
    )
    parser.add_argument(
        "--series",
        nargs="*",
        help=f"Series tickers (default: {', '.join(TIER1_SERIES)})",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Look back N months (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Mispricing threshold (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

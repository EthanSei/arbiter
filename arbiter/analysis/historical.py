"""Analysis functions for historical settled market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SettledMarket:
    """A settled Kalshi market with its outcome."""

    ticker: str
    event_ticker: str
    title: str
    category: str
    last_price: float  # market-implied probability [0, 1]
    result: str  # "yes" or "no"
    volume: int  # total lifetime volume (contracts traded)
    floor_strike: float | None  # for bracket/range markets
    cap_strike: float | None
    close_time: datetime | None


@dataclass(frozen=True)
class SeriesAnalysis:
    """Aggregate analysis of a series' settled markets."""

    series_ticker: str
    total_markets: int
    markets_with_volume: int
    midrange_count: int  # markets with last_price between 10-90%
    midrange_pct: float  # midrange_count / markets_with_volume
    surprise_count: int  # markets where |last_price - outcome| > threshold
    surprise_rate: float  # surprise_count / midrange_count (only midrange matters)
    brier_score: float  # mean squared error of last_price vs outcome
    total_volume: int
    midrange_volume: int
    bracket_families: int  # number of distinct event_tickers (bracket groups)
    avg_bracket_size: float  # average markets per bracket family


@dataclass(frozen=True)
class MispricingEpisode:
    """A period where a market was mispriced (deviated from anchor/fair value)."""

    ticker: str
    start_ts: int
    end_ts: int
    duration_minutes: int
    peak_deviation: float  # max distance from fair value during episode
    direction: str  # "overpriced" or "underpriced"


def analyze_mispricing_duration(
    candlesticks: list[dict[str, Any]],
    fair_value: float,
    threshold: float = 0.05,
    ticker: str = "",
) -> list[MispricingEpisode]:
    """Find episodes where price deviated from fair_value by more than threshold.

    Uses the yes_price close for each candlestick to detect deviations.
    An episode starts when |close - fair_value| > threshold and ends when the
    deviation returns within threshold. Duration is computed from timestamps.

    Args:
        candlesticks: List of candlestick dicts from Kalshi API.
        fair_value: The anchor/fair probability to measure deviation from.
        threshold: Minimum absolute deviation to count as mispriced.
        ticker: Market ticker to attach to episodes.

    Returns:
        List of MispricingEpisode sorted chronologically.
    """
    if not candlesticks:
        return []

    episodes: list[MispricingEpisode] = []

    # Track current episode state
    episode_start_ts: int | None = None
    episode_peak: float = 0.0
    episode_direction: str = ""
    episode_timestamps: list[int] = []

    for candle in candlesticks:
        ts = candle["end_period_ts"]
        close = candle["yes_price"]["close"]
        deviation = close - fair_value

        if abs(deviation) > threshold:
            direction = "overpriced" if deviation > 0 else "underpriced"
            if episode_start_ts is None:
                # Start new episode
                episode_start_ts = ts
                episode_peak = abs(deviation)
                episode_direction = direction
                episode_timestamps = [ts]
            else:
                # Continue existing episode
                episode_peak = max(episode_peak, abs(deviation))
                episode_timestamps.append(ts)
        else:
            if episode_start_ts is not None:
                # End current episode
                end_ts = episode_timestamps[-1]
                duration_seconds = end_ts - episode_start_ts
                duration_minutes = duration_seconds // 60
                episodes.append(
                    MispricingEpisode(
                        ticker=ticker,
                        start_ts=episode_start_ts,
                        end_ts=end_ts,
                        duration_minutes=duration_minutes,
                        peak_deviation=episode_peak,
                        direction=episode_direction,
                    )
                )
                episode_start_ts = None
                episode_peak = 0.0
                episode_direction = ""
                episode_timestamps = []

    # Close any open episode at end of data
    if episode_start_ts is not None:
        end_ts = episode_timestamps[-1]
        duration_seconds = end_ts - episode_start_ts
        duration_minutes = duration_seconds // 60
        episodes.append(
            MispricingEpisode(
                ticker=ticker,
                start_ts=episode_start_ts,
                end_ts=end_ts,
                duration_minutes=duration_minutes,
                peak_deviation=episode_peak,
                direction=episode_direction,
            )
        )

    return episodes


def parse_settled_market(raw: dict[str, Any]) -> SettledMarket | None:
    """Convert a raw Kalshi API dict to a SettledMarket.

    Returns None if essential fields (ticker, event_ticker, title, category,
    last_price, result, volume) are missing.

    last_price is in cents (0-100 int) in the API -- normalized to [0, 1].
    """
    try:
        ticker = raw["ticker"]
        event_ticker = raw["event_ticker"]
        title = raw["title"]
        category = raw.get("category") or ""
        last_price_cents = raw["last_price"]
        result = raw["result"]
        volume = raw["volume"]
    except KeyError:
        return None

    # Normalize price from cents to probability
    last_price = last_price_cents / 100.0

    # Optional fields
    floor_strike = raw.get("floor_strike")
    cap_strike = raw.get("cap_strike")

    close_time_raw = raw.get("close_time")
    close_time: datetime | None = None
    if close_time_raw is not None:
        close_time = datetime.fromisoformat(close_time_raw.replace("Z", "+00:00"))

    return SettledMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        title=title,
        category=category,
        last_price=last_price,
        result=result,
        volume=volume,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        close_time=close_time,
    )


def brier_score(markets: list[SettledMarket]) -> float:
    """Compute mean squared error: avg of (last_price - outcome)^2.

    Outcome is 1.0 if result == "yes", else 0.0.
    Only includes markets with volume > 0. Returns 0.0 if no qualifying markets.
    """
    qualifying = [m for m in markets if m.volume > 0]
    if not qualifying:
        return 0.0

    total = 0.0
    for m in qualifying:
        outcome = 1.0 if m.result == "yes" else 0.0
        total += (m.last_price - outcome) ** 2

    return total / len(qualifying)


def midrange_density(
    markets: list[SettledMarket],
    low: float = 0.10,
    high: float = 0.90,
) -> tuple[int, int]:
    """Count midrange markets vs total markets with volume.

    Returns (midrange_count, total_with_volume).
    Midrange = markets with low <= last_price <= high AND volume > 0.
    """
    with_volume = [m for m in markets if m.volume > 0]
    midrange = [m for m in with_volume if low <= m.last_price <= high]
    return len(midrange), len(with_volume)


def surprise_rate(
    markets: list[SettledMarket],
    threshold: float = 0.30,
) -> tuple[int, int]:
    """Count surprises among midrange markets.

    Only considers midrange markets (10-90% with volume > 0).
    A surprise is where |last_price - outcome| > threshold.
    Returns (surprise_count, midrange_count).
    """
    with_volume = [m for m in markets if m.volume > 0]
    midrange = [m for m in with_volume if 0.10 <= m.last_price <= 0.90]

    if not midrange:
        return 0, 0

    surprises = 0
    for m in midrange:
        outcome = 1.0 if m.result == "yes" else 0.0
        if abs(m.last_price - outcome) > threshold:
            surprises += 1

    return surprises, len(midrange)


def detect_bracket_families(
    markets: list[SettledMarket],
) -> dict[str, list[SettledMarket]]:
    """Group markets by event_ticker.

    Each group is a "bracket family" (e.g., all CPI ranges for January 2026).
    """
    families: dict[str, list[SettledMarket]] = {}
    for m in markets:
        families.setdefault(m.event_ticker, []).append(m)
    return families


def analyze_series(
    series_ticker: str,
    markets: list[SettledMarket],
) -> SeriesAnalysis:
    """Aggregate analysis of a series' settled markets.

    Calls brier_score, midrange_density, surprise_rate, and detect_bracket_families
    to compute a full SeriesAnalysis.
    """
    if not markets:
        return SeriesAnalysis(
            series_ticker=series_ticker,
            total_markets=0,
            markets_with_volume=0,
            midrange_count=0,
            midrange_pct=0.0,
            surprise_count=0,
            surprise_rate=0.0,
            brier_score=0.0,
            total_volume=0,
            midrange_volume=0,
            bracket_families=0,
            avg_bracket_size=0.0,
        )

    with_volume = [m for m in markets if m.volume > 0]
    mid_count, total_with_vol = midrange_density(markets)
    sur_count, mid_for_surprise = surprise_rate(markets)
    bs = brier_score(markets)
    families = detect_bracket_families(markets)

    total_volume = sum(m.volume for m in markets)

    # Midrange volume: sum volume of midrange markets with volume > 0
    midrange_vol = sum(m.volume for m in with_volume if 0.10 <= m.last_price <= 0.90)

    midrange_pct = mid_count / total_with_vol if total_with_vol > 0 else 0.0
    sur_rate = sur_count / mid_for_surprise if mid_for_surprise > 0 else 0.0

    num_families = len(families)
    avg_bracket = len(markets) / num_families if num_families > 0 else 0.0

    return SeriesAnalysis(
        series_ticker=series_ticker,
        total_markets=len(markets),
        markets_with_volume=len(with_volume),
        midrange_count=mid_count,
        midrange_pct=midrange_pct,
        surprise_count=sur_count,
        surprise_rate=sur_rate,
        brier_score=bs,
        total_volume=total_volume,
        midrange_volume=midrange_vol,
        bracket_families=num_families,
        avg_bracket_size=avg_bracket,
    )

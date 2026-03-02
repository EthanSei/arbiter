"""Tests for historical settled market analysis."""

from __future__ import annotations

import pytest

from arbiter.analysis.historical import (
    MispricingEpisode,
    SeriesAnalysis,
    SettledMarket,
    analyze_mispricing_duration,
    analyze_series,
    brier_score,
    detect_bracket_families,
    midrange_density,
    parse_settled_market,
    surprise_rate,
)

# ---------------------------------------------------------------------------
# Fixtures — realistic Kalshi settled market data
# ---------------------------------------------------------------------------

SETTLED_CPI_MARKETS = [
    {
        "ticker": "KXCPI-26JAN-T0.004",
        "event_ticker": "KXCPI-26JAN",
        "title": "Will CPI rise more than 0.4% in January 2026?",
        "category": "Economics",
        "last_price": 6,  # 6 cents = 0.06 probability
        "result": "no",
        "volume": 37900,
        "floor_strike": 0.004,
        "cap_strike": None,
        "close_time": "2026-01-15T21:00:00Z",
        "status": "finalized",
        "settlement_value": 0,
    },
    {
        "ticker": "KXCPI-26JAN-T0.003",
        "event_ticker": "KXCPI-26JAN",
        "title": "Will CPI rise more than 0.3% in January 2026?",
        "category": "Economics",
        "last_price": 25,  # 25 cents = 0.25 probability
        "result": "no",
        "volume": 108385,
        "floor_strike": 0.003,
        "cap_strike": None,
        "close_time": "2026-01-15T21:00:00Z",
        "status": "finalized",
        "settlement_value": 0,
    },
    {
        "ticker": "KXCPI-26JAN-T0.002",
        "event_ticker": "KXCPI-26JAN",
        "title": "Will CPI rise more than 0.2% in January 2026?",
        "category": "Economics",
        "last_price": 83,  # 83 cents = 0.83 probability
        "result": "no",  # surprise! market was wrong
        "volume": 162835,
        "floor_strike": 0.002,
        "cap_strike": None,
        "close_time": "2026-01-15T21:00:00Z",
        "status": "finalized",
        "settlement_value": 0,
    },
    {
        "ticker": "KXCPI-26JAN-T0.001",
        "event_ticker": "KXCPI-26JAN",
        "title": "Will CPI rise more than 0.1% in January 2026?",
        "category": "Economics",
        "last_price": 92,  # 92 cents = 0.92
        "result": "yes",
        "volume": 184087,
        "floor_strike": 0.001,
        "cap_strike": None,
        "close_time": "2026-01-15T21:00:00Z",
        "status": "finalized",
        "settlement_value": 100,
    },
]

# A second event_ticker for bracket family testing
SETTLED_GDP_MARKETS = [
    {
        "ticker": "KXGDP-26Q1-T3.0",
        "event_ticker": "KXGDP-26Q1",
        "title": "Will GDP growth exceed 3.0%?",
        "category": "Economics",
        "last_price": 40,
        "result": "no",
        "volume": 50000,
        "floor_strike": 3.0,
        "cap_strike": None,
        "close_time": "2026-04-01T12:00:00Z",
        "status": "finalized",
        "settlement_value": 0,
    },
    {
        "ticker": "KXGDP-26Q1-T2.0",
        "event_ticker": "KXGDP-26Q1",
        "title": "Will GDP growth exceed 2.0%?",
        "category": "Economics",
        "last_price": 70,
        "result": "yes",
        "volume": 85000,
        "floor_strike": 2.0,
        "cap_strike": None,
        "close_time": "2026-04-01T12:00:00Z",
        "status": "finalized",
        "settlement_value": 100,
    },
]


@pytest.fixture
def cpi_markets() -> list[SettledMarket]:
    """Parsed CPI settled markets."""
    result = []
    for raw in SETTLED_CPI_MARKETS:
        m = parse_settled_market(raw)
        assert m is not None
        result.append(m)
    return result


@pytest.fixture
def gdp_markets() -> list[SettledMarket]:
    """Parsed GDP settled markets."""
    result = []
    for raw in SETTLED_GDP_MARKETS:
        m = parse_settled_market(raw)
        assert m is not None
        result.append(m)
    return result


@pytest.fixture
def all_markets(
    cpi_markets: list[SettledMarket], gdp_markets: list[SettledMarket]
) -> list[SettledMarket]:
    """All parsed markets combined."""
    return cpi_markets + gdp_markets


# ===========================================================================
# TestParseSettledMarket
# ===========================================================================


class TestParseSettledMarket:
    def test_parses_valid_market(self) -> None:
        """Should parse a valid raw dict into a SettledMarket with normalized price."""
        raw = SETTLED_CPI_MARKETS[0]
        m = parse_settled_market(raw)
        assert m is not None
        assert m.ticker == "KXCPI-26JAN-T0.004"
        assert m.event_ticker == "KXCPI-26JAN"
        assert m.title == "Will CPI rise more than 0.4% in January 2026?"
        assert m.category == "Economics"
        assert m.last_price == pytest.approx(0.06)
        assert m.result == "no"
        assert m.volume == 37900
        assert m.floor_strike == 0.004
        assert m.cap_strike is None

    def test_normalizes_price_from_cents(self) -> None:
        """last_price in cents (0-100) should be normalized to [0, 1]."""
        raw = SETTLED_CPI_MARKETS[1]
        m = parse_settled_market(raw)
        assert m is not None
        assert m.last_price == pytest.approx(0.25)

    def test_parses_close_time(self) -> None:
        """close_time ISO string should be parsed to datetime."""
        raw = SETTLED_CPI_MARKETS[0]
        m = parse_settled_market(raw)
        assert m is not None
        assert m.close_time is not None
        assert m.close_time.year == 2026
        assert m.close_time.month == 1
        assert m.close_time.day == 15

    def test_returns_none_missing_ticker(self) -> None:
        """Should return None if ticker is missing."""
        raw = {k: v for k, v in SETTLED_CPI_MARKETS[0].items() if k != "ticker"}
        assert parse_settled_market(raw) is None

    def test_returns_none_missing_result(self) -> None:
        """Should return None if result is missing."""
        raw = {k: v for k, v in SETTLED_CPI_MARKETS[0].items() if k != "result"}
        assert parse_settled_market(raw) is None

    def test_returns_none_missing_last_price(self) -> None:
        """Should return None if last_price is missing."""
        raw = {k: v for k, v in SETTLED_CPI_MARKETS[0].items() if k != "last_price"}
        assert parse_settled_market(raw) is None

    def test_handles_none_close_time(self) -> None:
        """Should handle None close_time gracefully."""
        raw = {**SETTLED_CPI_MARKETS[0], "close_time": None}
        m = parse_settled_market(raw)
        assert m is not None
        assert m.close_time is None

    def test_handles_missing_close_time(self) -> None:
        """Should handle missing close_time key gracefully."""
        raw = {k: v for k, v in SETTLED_CPI_MARKETS[0].items() if k != "close_time"}
        m = parse_settled_market(raw)
        assert m is not None
        assert m.close_time is None

    def test_handles_missing_strikes(self) -> None:
        """Should handle missing floor_strike and cap_strike gracefully."""
        raw = {
            k: v
            for k, v in SETTLED_CPI_MARKETS[0].items()
            if k not in ("floor_strike", "cap_strike")
        }
        m = parse_settled_market(raw)
        assert m is not None
        assert m.floor_strike is None
        assert m.cap_strike is None

    def test_handles_none_category(self) -> None:
        """Kalshi API returns category=None for settled markets — default to empty string."""
        raw = {**SETTLED_CPI_MARKETS[0], "category": None}
        m = parse_settled_market(raw)
        assert m is not None
        assert m.category == ""

    def test_frozen_dataclass(self) -> None:
        """SettledMarket should be immutable."""
        m = parse_settled_market(SETTLED_CPI_MARKETS[0])
        assert m is not None
        with pytest.raises(AttributeError):
            m.ticker = "mutated"  # type: ignore[misc]


# ===========================================================================
# TestBrierScore
# ===========================================================================


class TestBrierScore:
    def test_perfect_calibration(self) -> None:
        """Markets where price matches outcome perfectly should have Brier score ~0."""
        perfect = [
            SettledMarket(
                ticker="P1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.99,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="P2",
                event_ticker="E1",
                title="T2",
                category="C",
                last_price=0.01,
                result="no",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        score = brier_score(perfect)
        assert score == pytest.approx(0.0001, abs=1e-6)

    def test_worst_calibration(self) -> None:
        """Markets maximally wrong should have Brier score ~1.0."""
        worst = [
            SettledMarket(
                ticker="W1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.0,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        score = brier_score(worst)
        assert score == pytest.approx(1.0)

    def test_cpi_markets(self, cpi_markets: list[SettledMarket]) -> None:
        """Brier score for CPI markets should be between 0 and 1."""
        score = brier_score(cpi_markets)
        assert 0 < score < 1
        # Manual: (0.06-0)^2 + (0.25-0)^2 + (0.83-0)^2 + (0.92-1)^2
        # = 0.0036 + 0.0625 + 0.6889 + 0.0064 = 0.7614
        # Mean = 0.7614 / 4 = 0.19035
        assert score == pytest.approx(0.19035, abs=1e-4)

    def test_excludes_zero_volume(self) -> None:
        """Markets with zero volume should be excluded."""
        markets = [
            SettledMarket(
                ticker="Z1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.50,
                result="yes",
                volume=0,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="Z2",
                event_ticker="E1",
                title="T2",
                category="C",
                last_price=0.80,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        score = brier_score(markets)
        # Only Z2: (0.80 - 1.0)^2 = 0.04
        assert score == pytest.approx(0.04)

    def test_empty_list(self) -> None:
        """Empty list should return 0.0."""
        assert brier_score([]) == 0.0

    def test_all_zero_volume(self) -> None:
        """All markets with zero volume should return 0.0."""
        markets = [
            SettledMarket(
                ticker="Z1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.50,
                result="yes",
                volume=0,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        assert brier_score(markets) == 0.0


# ===========================================================================
# TestMidrangeDensity
# ===========================================================================


class TestMidrangeDensity:
    def test_cpi_markets(self, cpi_markets: list[SettledMarket]) -> None:
        """CPI markets: 0.06 (no), 0.25 (yes), 0.83 (yes), 0.92 (no)."""
        count, total = midrange_density(cpi_markets)
        # 0.06 < 0.10 -> not midrange
        # 0.25 -> midrange
        # 0.83 -> midrange
        # 0.92 > 0.90 -> not midrange
        assert count == 2
        assert total == 4

    def test_custom_bounds(self, cpi_markets: list[SettledMarket]) -> None:
        """Custom bounds [0.05, 0.95] should include more markets."""
        count, total = midrange_density(cpi_markets, low=0.05, high=0.95)
        # 0.06 -> midrange (>= 0.05)
        # 0.25 -> midrange
        # 0.83 -> midrange
        # 0.92 -> midrange (<= 0.95)
        assert count == 4
        assert total == 4

    def test_excludes_zero_volume(self) -> None:
        """Zero volume markets should not count toward total."""
        markets = [
            SettledMarket(
                ticker="M1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.50,
                result="yes",
                volume=0,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="M2",
                event_ticker="E1",
                title="T2",
                category="C",
                last_price=0.50,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        count, total = midrange_density(markets)
        assert count == 1
        assert total == 1

    def test_empty_list(self) -> None:
        """Empty list should return (0, 0)."""
        assert midrange_density([]) == (0, 0)

    def test_boundary_inclusive(self) -> None:
        """Boundaries should be inclusive: 0.10 and 0.90 are both midrange."""
        markets = [
            SettledMarket(
                ticker="B1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.10,
                result="no",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="B2",
                event_ticker="E1",
                title="T2",
                category="C",
                last_price=0.90,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        count, total = midrange_density(markets)
        assert count == 2
        assert total == 2


# ===========================================================================
# TestSurpriseRate
# ===========================================================================


class TestSurpriseRate:
    def test_cpi_markets(self, cpi_markets: list[SettledMarket]) -> None:
        """CPI surprise rate: only midrange markets (0.25, 0.83) are considered.
        0.25 resolved no: |0.25 - 0| = 0.25, not > 0.30
        0.83 resolved no: |0.83 - 0| = 0.83, > 0.30 -> surprise!
        """
        surprises, midrange = surprise_rate(cpi_markets)
        assert surprises == 1
        assert midrange == 2

    def test_custom_threshold(self, cpi_markets: list[SettledMarket]) -> None:
        """Lower threshold should catch more surprises."""
        surprises, midrange = surprise_rate(cpi_markets, threshold=0.20)
        # 0.25 resolved no: |0.25 - 0| = 0.25 > 0.20 -> surprise
        # 0.83 resolved no: |0.83 - 0| = 0.83 > 0.20 -> surprise
        assert surprises == 2
        assert midrange == 2

    def test_no_midrange_markets(self) -> None:
        """If no markets are in midrange, surprise count should be (0, 0)."""
        markets = [
            SettledMarket(
                ticker="E1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.05,
                result="no",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="E2",
                event_ticker="E1",
                title="T2",
                category="C",
                last_price=0.95,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        surprises, midrange = surprise_rate(markets)
        assert surprises == 0
        assert midrange == 0

    def test_empty_list(self) -> None:
        """Empty list should return (0, 0)."""
        assert surprise_rate([]) == (0, 0)

    def test_no_surprises(self) -> None:
        """Well-calibrated midrange markets should have no surprises."""
        markets = [
            SettledMarket(
                ticker="NS1",
                event_ticker="E1",
                title="T1",
                category="C",
                last_price=0.80,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="NS2",
                event_ticker="E1",
                title="T2",
                category="C",
                last_price=0.20,
                result="no",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        surprises, midrange = surprise_rate(markets)
        assert surprises == 0
        assert midrange == 2


# ===========================================================================
# TestDetectBracketFamilies
# ===========================================================================


class TestDetectBracketFamilies:
    def test_single_family(self, cpi_markets: list[SettledMarket]) -> None:
        """All CPI markets share the same event_ticker."""
        families = detect_bracket_families(cpi_markets)
        assert len(families) == 1
        assert "KXCPI-26JAN" in families
        assert len(families["KXCPI-26JAN"]) == 4

    def test_multiple_families(self, all_markets: list[SettledMarket]) -> None:
        """CPI + GDP markets should produce two families."""
        families = detect_bracket_families(all_markets)
        assert len(families) == 2
        assert "KXCPI-26JAN" in families
        assert "KXGDP-26Q1" in families
        assert len(families["KXCPI-26JAN"]) == 4
        assert len(families["KXGDP-26Q1"]) == 2

    def test_empty_list(self) -> None:
        """Empty list should return empty dict."""
        assert detect_bracket_families([]) == {}

    def test_preserves_markets(self, cpi_markets: list[SettledMarket]) -> None:
        """Markets in families should be the same objects."""
        families = detect_bracket_families(cpi_markets)
        for market in families["KXCPI-26JAN"]:
            assert market in cpi_markets


# ===========================================================================
# TestAnalyzeSeries
# ===========================================================================


class TestAnalyzeSeries:
    def test_cpi_series(self, cpi_markets: list[SettledMarket]) -> None:
        """End-to-end analysis of CPI series."""
        result = analyze_series("KXCPI", cpi_markets)

        assert isinstance(result, SeriesAnalysis)
        assert result.series_ticker == "KXCPI"
        assert result.total_markets == 4
        assert result.markets_with_volume == 4
        assert result.midrange_count == 2  # 0.25 and 0.83
        assert result.midrange_pct == pytest.approx(0.5)
        assert result.surprise_count == 1  # 0.83 resolved no
        assert result.surprise_rate == pytest.approx(0.5)  # 1/2
        assert 0 < result.brier_score < 1
        assert result.total_volume == 37900 + 108385 + 162835 + 184087
        assert result.bracket_families == 1  # all same event_ticker
        assert result.avg_bracket_size == pytest.approx(4.0)

    def test_mixed_series(self, all_markets: list[SettledMarket]) -> None:
        """Analysis across multiple event_tickers."""
        result = analyze_series("ECON", all_markets)

        assert result.total_markets == 6
        assert result.bracket_families == 2  # CPI + GDP
        assert result.avg_bracket_size == pytest.approx(3.0)  # 6 markets / 2 families

    def test_midrange_volume(self, cpi_markets: list[SettledMarket]) -> None:
        """midrange_volume should sum volumes of midrange markets only."""
        result = analyze_series("KXCPI", cpi_markets)
        # Midrange: 0.25 (volume=108385) and 0.83 (volume=162835)
        assert result.midrange_volume == 108385 + 162835

    def test_empty_markets(self) -> None:
        """Empty market list should return zeroed-out SeriesAnalysis."""
        result = analyze_series("EMPTY", [])

        assert result.series_ticker == "EMPTY"
        assert result.total_markets == 0
        assert result.markets_with_volume == 0
        assert result.midrange_count == 0
        assert result.midrange_pct == 0.0
        assert result.surprise_count == 0
        assert result.surprise_rate == 0.0
        assert result.brier_score == 0.0
        assert result.total_volume == 0
        assert result.midrange_volume == 0
        assert result.bracket_families == 0
        assert result.avg_bracket_size == 0.0

    def test_frozen_dataclass(self, cpi_markets: list[SettledMarket]) -> None:
        """SeriesAnalysis should be immutable."""
        result = analyze_series("KXCPI", cpi_markets)
        with pytest.raises(AttributeError):
            result.total_markets = 999  # type: ignore[misc]

    def test_zero_midrange_surprise_rate(self) -> None:
        """When no midrange markets exist, surprise_rate should be 0.0."""
        markets = [
            SettledMarket(
                ticker="EX1",
                event_ticker="EX",
                title="T1",
                category="C",
                last_price=0.02,
                result="no",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
            SettledMarket(
                ticker="EX2",
                event_ticker="EX",
                title="T2",
                category="C",
                last_price=0.98,
                result="yes",
                volume=100,
                floor_strike=None,
                cap_strike=None,
                close_time=None,
            ),
        ]
        result = analyze_series("EXTREME", markets)
        assert result.midrange_count == 0
        assert result.surprise_rate == 0.0


# ===========================================================================
# TestMispricingEpisode
# ===========================================================================


class TestMispricingEpisode:
    def test_frozen_dataclass(self) -> None:
        """MispricingEpisode should be immutable."""
        ep = MispricingEpisode(
            ticker="KXBTC-26MAR14-T100000",
            start_ts=1709300400,
            end_ts=1709304000,
            duration_minutes=60,
            peak_deviation=0.10,
            direction="overpriced",
        )
        with pytest.raises(AttributeError):
            ep.ticker = "mutated"  # type: ignore[misc]

    def test_fields(self) -> None:
        """MispricingEpisode should store all fields correctly."""
        ep = MispricingEpisode(
            ticker="KXBTC-26MAR14-T100000",
            start_ts=1709300400,
            end_ts=1709307600,
            duration_minutes=120,
            peak_deviation=0.15,
            direction="underpriced",
        )
        assert ep.ticker == "KXBTC-26MAR14-T100000"
        assert ep.start_ts == 1709300400
        assert ep.end_ts == 1709307600
        assert ep.duration_minutes == 120
        assert ep.peak_deviation == pytest.approx(0.15)
        assert ep.direction == "underpriced"


# ===========================================================================
# TestAnalyzeMispricingDuration
# ===========================================================================

# Candlestick fixtures for mispricing analysis.
# Each candlestick has 60-minute intervals (period_interval=60).
# Timestamps are spaced 3600s apart.

CANDLES_OVERPRICED = [
    # Fair value = 0.50, threshold = 0.05
    # Candle 1: close=0.50 -> no deviation (within threshold)
    {
        "end_period_ts": 1709300400,
        "yes_price": {"open": 0.48, "high": 0.52, "low": 0.47, "close": 0.50},
        "volume": 100,
    },
    # Candle 2: close=0.58 -> overpriced by 0.08 (>0.05) - episode starts
    {
        "end_period_ts": 1709304000,
        "yes_price": {"open": 0.52, "high": 0.60, "low": 0.51, "close": 0.58},
        "volume": 200,
    },
    # Candle 3: close=0.62 -> overpriced by 0.12 - episode continues, peak deviation
    {
        "end_period_ts": 1709307600,
        "yes_price": {"open": 0.58, "high": 0.65, "low": 0.56, "close": 0.62},
        "volume": 150,
    },
    # Candle 4: close=0.51 -> back within threshold - episode ends
    {
        "end_period_ts": 1709311200,
        "yes_price": {"open": 0.60, "high": 0.61, "low": 0.49, "close": 0.51},
        "volume": 300,
    },
]

CANDLES_UNDERPRICED = [
    # Fair value = 0.70, threshold = 0.05
    # Candle 1: close=0.63 -> underpriced by 0.07 (>0.05) - episode starts
    {
        "end_period_ts": 1709300400,
        "yes_price": {"open": 0.65, "high": 0.66, "low": 0.60, "close": 0.63},
        "volume": 100,
    },
    # Candle 2: close=0.60 -> underpriced by 0.10 - peak
    {
        "end_period_ts": 1709304000,
        "yes_price": {"open": 0.63, "high": 0.64, "low": 0.58, "close": 0.60},
        "volume": 200,
    },
    # Candle 3: close=0.68 -> back within threshold - episode ends
    {
        "end_period_ts": 1709307600,
        "yes_price": {"open": 0.62, "high": 0.70, "low": 0.61, "close": 0.68},
        "volume": 150,
    },
]

CANDLES_MULTIPLE_EPISODES = [
    # Fair value = 0.50, threshold = 0.05
    # Candle 1: close=0.58 -> overpriced episode 1 starts
    {
        "end_period_ts": 1709300400,
        "yes_price": {"open": 0.50, "high": 0.60, "low": 0.49, "close": 0.58},
        "volume": 100,
    },
    # Candle 2: close=0.50 -> back to fair value, episode 1 ends
    {
        "end_period_ts": 1709304000,
        "yes_price": {"open": 0.58, "high": 0.58, "low": 0.48, "close": 0.50},
        "volume": 200,
    },
    # Candle 3: close=0.42 -> underpriced episode 2 starts
    {
        "end_period_ts": 1709307600,
        "yes_price": {"open": 0.50, "high": 0.51, "low": 0.40, "close": 0.42},
        "volume": 150,
    },
    # Candle 4: close=0.40 -> still underpriced, peak deviation
    {
        "end_period_ts": 1709311200,
        "yes_price": {"open": 0.42, "high": 0.44, "low": 0.38, "close": 0.40},
        "volume": 300,
    },
    # Candle 5: close=0.49 -> back within threshold, episode 2 ends
    {
        "end_period_ts": 1709314800,
        "yes_price": {"open": 0.40, "high": 0.52, "low": 0.39, "close": 0.49},
        "volume": 250,
    },
]

CANDLES_NO_MISPRICING = [
    # Fair value = 0.50, threshold = 0.05 -> all close prices within 0.05
    {
        "end_period_ts": 1709300400,
        "yes_price": {"open": 0.49, "high": 0.53, "low": 0.47, "close": 0.51},
        "volume": 100,
    },
    {
        "end_period_ts": 1709304000,
        "yes_price": {"open": 0.51, "high": 0.54, "low": 0.48, "close": 0.52},
        "volume": 200,
    },
    {
        "end_period_ts": 1709307600,
        "yes_price": {"open": 0.52, "high": 0.55, "low": 0.46, "close": 0.49},
        "volume": 150,
    },
]


class TestAnalyzeMispricingDuration:
    def test_detects_overpriced_episode(self) -> None:
        """Should detect a period where market was overpriced above threshold."""
        episodes = analyze_mispricing_duration(
            CANDLES_OVERPRICED,
            fair_value=0.50,
            threshold=0.05,
        )

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.direction == "overpriced"
        assert ep.start_ts == 1709304000
        assert ep.end_ts == 1709307600
        assert ep.duration_minutes == 60
        assert ep.peak_deviation == pytest.approx(0.12)

    def test_detects_underpriced_episode(self) -> None:
        """Should detect a period where market was underpriced below threshold."""
        episodes = analyze_mispricing_duration(
            CANDLES_UNDERPRICED,
            fair_value=0.70,
            threshold=0.05,
        )

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.direction == "underpriced"
        assert ep.start_ts == 1709300400
        assert ep.end_ts == 1709304000
        assert ep.duration_minutes == 60
        assert ep.peak_deviation == pytest.approx(0.10)

    def test_detects_multiple_episodes(self) -> None:
        """Should detect multiple separate mispricing episodes."""
        episodes = analyze_mispricing_duration(
            CANDLES_MULTIPLE_EPISODES,
            fair_value=0.50,
            threshold=0.05,
        )

        assert len(episodes) == 2
        assert episodes[0].direction == "overpriced"
        assert episodes[1].direction == "underpriced"

    def test_multiple_episodes_durations(self) -> None:
        """Each episode should have correct duration."""
        episodes = analyze_mispricing_duration(
            CANDLES_MULTIPLE_EPISODES,
            fair_value=0.50,
            threshold=0.05,
        )

        # Episode 1: single candle overpriced (1709300400 only)
        assert episodes[0].duration_minutes == 0  # single point
        # Episode 2: two candles underpriced (1709307600 to 1709311200)
        assert episodes[1].duration_minutes == 60

    def test_multiple_episodes_peak_deviations(self) -> None:
        """Peak deviation should be the max absolute distance from fair value."""
        episodes = analyze_mispricing_duration(
            CANDLES_MULTIPLE_EPISODES,
            fair_value=0.50,
            threshold=0.05,
        )

        assert episodes[0].peak_deviation == pytest.approx(0.08)  # 0.58 - 0.50
        assert episodes[1].peak_deviation == pytest.approx(0.10)  # 0.50 - 0.40

    def test_no_mispricing(self) -> None:
        """Should return empty list when no prices deviate beyond threshold."""
        episodes = analyze_mispricing_duration(
            CANDLES_NO_MISPRICING,
            fair_value=0.50,
            threshold=0.05,
        )

        assert episodes == []

    def test_empty_candlesticks(self) -> None:
        """Should return empty list for empty candlestick data."""
        episodes = analyze_mispricing_duration([], fair_value=0.50, threshold=0.05)
        assert episodes == []

    def test_ticker_passed_through(self) -> None:
        """Ticker parameter should be passed through to episodes."""
        episodes = analyze_mispricing_duration(
            CANDLES_OVERPRICED,
            fair_value=0.50,
            threshold=0.05,
            ticker="KXBTC-26MAR14-T100000",
        )

        assert len(episodes) == 1
        assert episodes[0].ticker == "KXBTC-26MAR14-T100000"

    def test_default_ticker_is_empty(self) -> None:
        """Default ticker should be empty string."""
        episodes = analyze_mispricing_duration(
            CANDLES_OVERPRICED,
            fair_value=0.50,
            threshold=0.05,
        )

        assert episodes[0].ticker == ""

    def test_high_threshold_no_episodes(self) -> None:
        """High threshold should filter out minor deviations."""
        episodes = analyze_mispricing_duration(
            CANDLES_OVERPRICED,
            fair_value=0.50,
            threshold=0.20,
        )

        assert episodes == []

    def test_zero_threshold_catches_all_deviations(self) -> None:
        """Zero threshold should catch any deviation from fair value."""
        episodes = analyze_mispricing_duration(
            CANDLES_NO_MISPRICING,
            fair_value=0.50,
            threshold=0.0,
        )

        # All candles deviate from 0.50: 0.51, 0.52, 0.49
        # This forms one continuous episode since they all deviate
        assert len(episodes) >= 1

    def test_episode_at_end_of_data(self) -> None:
        """Episode that runs to the end of candlestick data should be captured."""
        # The last two candles are overpriced and data ends
        candles = [
            {
                "end_period_ts": 1709300400,
                "yes_price": {"open": 0.50, "high": 0.52, "low": 0.48, "close": 0.50},
                "volume": 100,
            },
            {
                "end_period_ts": 1709304000,
                "yes_price": {"open": 0.52, "high": 0.62, "low": 0.51, "close": 0.60},
                "volume": 200,
            },
            {
                "end_period_ts": 1709307600,
                "yes_price": {"open": 0.60, "high": 0.68, "low": 0.58, "close": 0.65},
                "volume": 150,
            },
        ]
        episodes = analyze_mispricing_duration(candles, fair_value=0.50, threshold=0.05)

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.direction == "overpriced"
        assert ep.end_ts == 1709307600  # last candle timestamp
        assert ep.peak_deviation == pytest.approx(0.15)  # 0.65 - 0.50
        assert ep.duration_minutes == 60  # 2 candles

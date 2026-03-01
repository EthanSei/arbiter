"""Tests for feature extraction."""

from datetime import UTC, datetime

import numpy as np
import pytest

from arbiter.ingestion.base import Contract
from arbiter.models.features import SPEC, extract_features


@pytest.fixture
def contract() -> Contract:
    """A typical open contract with all fields populated."""
    return Contract(
        source="kalshi",
        contract_id="TICKER-YES",
        title="Will X happen?",
        category="politics",
        yes_price=0.60,
        no_price=0.40,
        yes_bid=0.58,
        yes_ask=0.62,
        last_price=0.59,
        volume_24h=50_000.0,
        open_interest=120_000.0,
        expires_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        url="https://kalshi.com/markets/TICKER",
        status="open",
    )


class TestFeatureVectorShape:
    """extract_features returns an array matching SPEC."""

    def test_returns_array_matching_spec_length(self, contract: Contract) -> None:
        features = extract_features(contract)
        assert features.shape == (len(SPEC.names),)

    def test_returns_float64_array(self, contract: Contract) -> None:
        features = extract_features(contract)
        assert features.dtype == np.float64


class TestMarketFeatures:
    """Core market-level features extracted from the contract."""

    def test_yes_price(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("yes_price")
        assert features[idx] == pytest.approx(0.60)

    def test_no_price(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("no_price")
        assert features[idx] == pytest.approx(0.40)

    def test_bid_ask_spread(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("bid_ask_spread")
        assert features[idx] == pytest.approx(0.04)  # 0.62 - 0.58

    def test_last_price(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("last_price")
        assert features[idx] == pytest.approx(0.59)

    def test_last_price_none_falls_back_to_yes_price(self, contract: Contract) -> None:
        c = Contract(
            source=contract.source,
            contract_id=contract.contract_id,
            title=contract.title,
            category=contract.category,
            yes_price=0.60,
            no_price=0.40,
            yes_bid=0.58,
            yes_ask=0.62,
            last_price=None,
            volume_24h=50_000.0,
            open_interest=120_000.0,
            expires_at=contract.expires_at,
            url=contract.url,
            status=contract.status,
        )
        features = extract_features(c)
        idx = SPEC.names.index("last_price")
        assert features[idx] == pytest.approx(0.60)

    def test_log_volume_24h(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("log_volume_24h")
        assert features[idx] == pytest.approx(np.log1p(50_000.0))

    def test_log_volume_zero(self) -> None:
        c = Contract(
            source="kalshi",
            contract_id="T",
            title="T",
            category="c",
            yes_price=0.5,
            no_price=0.5,
            yes_bid=0.49,
            yes_ask=0.51,
            last_price=None,
            volume_24h=0.0,
            open_interest=0.0,
            expires_at=None,
            url="",
            status="open",
        )
        features = extract_features(c)
        idx = SPEC.names.index("log_volume_24h")
        assert features[idx] == pytest.approx(0.0)

    def test_log_open_interest(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("log_open_interest")
        assert features[idx] == pytest.approx(np.log1p(120_000.0))

    def test_overround(self, contract: Contract) -> None:
        # overround = yes_price + no_price - 1.0
        features = extract_features(contract)
        idx = SPEC.names.index("overround")
        assert features[idx] == pytest.approx(0.0)  # 0.60 + 0.40 - 1.0


class TestTemporalFeatures:
    """Time-based features: expiry, day-of-week, hour-of-day."""

    def test_time_to_expiry_hours(self, contract: Contract) -> None:
        now = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)
        features = extract_features(contract, now=now)
        idx = SPEC.names.index("time_to_expiry_hours")
        assert features[idx] == pytest.approx(24.0)

    def test_time_to_expiry_clamps_expired_to_zero(self, contract: Contract) -> None:
        # Contract expired 2 hours ago — should clamp to 0, not return -2
        now = datetime(2026, 3, 15, 14, 0, tzinfo=UTC)
        features = extract_features(contract, now=now)
        idx = SPEC.names.index("time_to_expiry_hours")
        assert features[idx] == pytest.approx(0.0)

    def test_time_to_expiry_none_is_nan(self) -> None:
        c = Contract(
            source="kalshi",
            contract_id="T",
            title="T",
            category="c",
            yes_price=0.5,
            no_price=0.5,
            yes_bid=0.49,
            yes_ask=0.51,
            last_price=None,
            volume_24h=0.0,
            open_interest=0.0,
            expires_at=None,
            url="",
            status="open",
        )
        features = extract_features(c)
        idx = SPEC.names.index("time_to_expiry_hours")
        assert np.isnan(features[idx])

    def test_day_of_week(self, contract: Contract) -> None:
        # 2026-02-28 is a Saturday = day 5
        now = datetime(2026, 2, 28, 14, 30, tzinfo=UTC)
        features = extract_features(contract, now=now)
        idx = SPEC.names.index("day_of_week")
        assert features[idx] == pytest.approx(5.0)

    def test_hour_of_day(self, contract: Contract) -> None:
        now = datetime(2026, 2, 28, 14, 30, tzinfo=UTC)
        features = extract_features(contract, now=now)
        idx = SPEC.names.index("hour_of_day")
        assert features[idx] == pytest.approx(14.0)


class TestCrossPlatformFeatures:
    """Cross-platform features (price_discrepancy, volume_ratio)."""

    def test_price_discrepancy_with_match(self, contract: Contract) -> None:
        features = extract_features(contract, cross_platform_price=0.55)
        idx = SPEC.names.index("price_discrepancy")
        assert features[idx] == pytest.approx(0.05)  # 0.60 - 0.55

    def test_price_discrepancy_no_match_is_nan(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("price_discrepancy")
        assert np.isnan(features[idx])

    def test_volume_ratio_with_match(self, contract: Contract) -> None:
        features = extract_features(contract, cross_platform_volume=25_000.0)
        idx = SPEC.names.index("volume_ratio")
        assert features[idx] == pytest.approx(50_000.0 / 25_000.0)

    def test_volume_ratio_no_match_is_nan(self, contract: Contract) -> None:
        features = extract_features(contract)
        idx = SPEC.names.index("volume_ratio")
        assert np.isnan(features[idx])

    def test_volume_ratio_zero_cross_volume_is_nan(self, contract: Contract) -> None:
        features = extract_features(contract, cross_platform_volume=0.0)
        idx = SPEC.names.index("volume_ratio")
        assert np.isnan(features[idx])


class TestLagFeatures:
    """Lag features from price/volume history."""

    def test_price_delta_1h_with_history(self, contract: Contract) -> None:
        # Most recent last; if we have >= 2 values, delta_1h = last - second-to-last
        history = [0.50, 0.55, 0.60]
        features = extract_features(contract, price_history=history)
        idx = SPEC.names.index("price_delta_1h")
        assert features[idx] == pytest.approx(0.05)

    def test_price_delta_24h_with_history(self, contract: Contract) -> None:
        # delta_24h = last - first
        history = [0.50, 0.55, 0.60]
        features = extract_features(contract, price_history=history)
        idx = SPEC.names.index("price_delta_24h")
        assert features[idx] == pytest.approx(0.10)

    def test_price_volatility_24h_with_history(self, contract: Contract) -> None:
        history = [0.50, 0.55, 0.60]
        features = extract_features(contract, price_history=history)
        idx = SPEC.names.index("price_volatility_24h")
        expected = float(np.std(history))
        assert features[idx] == pytest.approx(expected)

    def test_volume_ratio_24h_with_history(self, contract: Contract) -> None:
        vol_history = [1000.0, 2000.0, 3000.0]
        features = extract_features(contract, volume_history=vol_history)
        idx = SPEC.names.index("volume_ratio_24h")
        # current / mean of history
        assert features[idx] == pytest.approx(50_000.0 / 2000.0)

    def test_lag_features_nan_without_history(self, contract: Contract) -> None:
        features = extract_features(contract)
        lag_names = [
            "price_delta_1h",
            "price_delta_24h",
            "price_volatility_24h",
            "volume_ratio_24h",
        ]
        for name in lag_names:
            idx = SPEC.names.index(name)
            assert np.isnan(features[idx]), f"{name} should be NaN without history"

    def test_price_delta_1h_single_point_is_nan(self, contract: Contract) -> None:
        features = extract_features(contract, price_history=[0.60])
        idx = SPEC.names.index("price_delta_1h")
        assert np.isnan(features[idx])

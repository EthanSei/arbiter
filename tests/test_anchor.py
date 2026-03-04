"""Tests for anchor scoring functions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbiter.ingestion.base import Contract
from arbiter.scoring.anchor import (
    compute_anchor_prob,
    extract_threshold,
    find_anchor_mispricings,
    group_anchor_contracts,
)


def _make_contract(
    contract_id: str,
    yes_price: float,
    source: str = "kalshi",
    volume_24h: float = 100.0,
) -> Contract:
    return Contract(
        source=source,
        contract_id=contract_id,
        title=f"Test {contract_id}",
        category="Economics",
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 6),
        yes_bid=round(yes_price - 0.01, 6),
        yes_ask=round(yes_price + 0.01, 6),
        last_price=None,
        volume_24h=volume_24h,
        open_interest=500.0,
        expires_at=datetime(2026, 3, 31, tzinfo=UTC),
        url="https://kalshi.com/test",
        status="open",
    )


# ---------------------------------------------------------------------------
# extract_threshold
# ---------------------------------------------------------------------------


class TestExtractThreshold:
    def test_cpi_t_suffix(self):
        result = extract_threshold("KXCPI-26JAN-T0.003")
        assert result == ("KXCPI-26JAN", pytest.approx(0.003))

    def test_jobless_claims_t_suffix(self):
        result = extract_threshold("KXJOBLESSCLAIMS-26MAR06-T220")
        assert result == ("KXJOBLESSCLAIMS-26MAR06", pytest.approx(220.0))

    def test_cpi_yoy_t_suffix(self):
        result = extract_threshold("KXCPIYOY-26JAN-T3.0")
        assert result == ("KXCPIYOY-26JAN", pytest.approx(3.0))

    def test_non_t_suffix_maxmon_returns_none(self):
        result = extract_threshold("KXBTCMAXMON-BTC-26MAR31-8000000")
        assert result is None

    def test_non_range_ticker_returns_none(self):
        result = extract_threshold("KXTRUMP-NOMINEE-26MAR")
        assert result is None


# ---------------------------------------------------------------------------
# group_anchor_contracts
# ---------------------------------------------------------------------------


class TestGroupAnchorContracts:
    def test_groups_by_event_prefix(self):
        contracts = [
            _make_contract("KXCPI-26JAN-T0.001", 0.90),
            _make_contract("KXCPI-26JAN-T0.002", 0.65),
            _make_contract("KXCPI-26JAN-T0.003", 0.35),
            _make_contract("KXCPI-26JAN-T0.004", 0.08),
        ]
        groups = group_anchor_contracts(contracts)
        assert "KXCPI-26JAN" in groups
        assert len(groups["KXCPI-26JAN"]) == 4

    def test_sorted_by_threshold_ascending(self):
        contracts = [
            _make_contract("KXCPI-26JAN-T0.004", 0.08),
            _make_contract("KXCPI-26JAN-T0.001", 0.90),
            _make_contract("KXCPI-26JAN-T0.003", 0.35),
        ]
        groups = group_anchor_contracts(contracts)
        thresholds = [t for t, _ in groups["KXCPI-26JAN"]]
        assert thresholds == sorted(thresholds)

    def test_skips_zero_volume(self):
        contracts = [
            _make_contract("KXCPI-26JAN-T0.001", 0.90, volume_24h=100.0),
            _make_contract("KXCPI-26JAN-T0.002", 0.50, volume_24h=0.0),
        ]
        groups = group_anchor_contracts(contracts)
        assert len(groups["KXCPI-26JAN"]) == 1

    def test_ignores_non_kalshi(self):
        contracts = [
            _make_contract("KXCPI-26JAN-T0.001", 0.90, source="polymarket"),
        ]
        groups = group_anchor_contracts(contracts)
        assert len(groups) == 0

    def test_separate_groups_for_different_indicators(self):
        contracts = [
            _make_contract("KXCPI-26JAN-T0.003", 0.35),
            _make_contract("KXJOBLESSCLAIMS-26MAR06-T220", 0.60),
        ]
        groups = group_anchor_contracts(contracts)
        assert "KXCPI-26JAN" in groups
        assert "KXJOBLESSCLAIMS-26MAR06" in groups

    def test_non_t_suffix_contracts_excluded(self):
        contracts = [
            _make_contract("KXBTCMAXMON-BTC-26MAR31-8000000", 0.40),
            _make_contract("KXCPI-26JAN-T0.003", 0.35),
        ]
        groups = group_anchor_contracts(contracts)
        assert len(groups) == 1
        assert "KXCPI-26JAN" in groups


# ---------------------------------------------------------------------------
# compute_anchor_prob
# ---------------------------------------------------------------------------


class TestComputeAnchorProb:
    def test_at_the_mean(self):
        # P(X > μ) ≈ 0.50
        assert compute_anchor_prob(0.003, mu=0.003, sigma=0.0015) == pytest.approx(0.50)

    def test_one_sigma_above(self):
        # P(X > μ+σ) ≈ 0.159
        mu, sigma = 0.003, 0.0015
        assert compute_anchor_prob(mu + sigma, mu=mu, sigma=sigma) == pytest.approx(
            0.1587, abs=0.001
        )

    def test_one_sigma_below(self):
        # P(X > μ-σ) ≈ 0.841
        mu, sigma = 0.003, 0.0015
        assert compute_anchor_prob(mu - sigma, mu=mu, sigma=sigma) == pytest.approx(
            0.8413, abs=0.001
        )

    def test_monotonically_decreasing(self):
        mu, sigma = 0.003, 0.0015
        thresholds = [0.001, 0.002, 0.003, 0.004, 0.005]
        probs = [compute_anchor_prob(t, mu=mu, sigma=sigma) for t in thresholds]
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1]

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            compute_anchor_prob(0.003, mu=0.003, sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            compute_anchor_prob(0.003, mu=0.003, sigma=-0.001)


# ---------------------------------------------------------------------------
# find_anchor_mispricings
# ---------------------------------------------------------------------------


class TestFindAnchorMispricings:
    def test_detects_underpriced_tail(self):
        # CPI toy example: μ=0.003, σ=0.0015
        # Threshold 0.004 → anchor ≈ 25%, market = 8¢ → EV ≈ 16¢
        group = [
            (0.001, _make_contract("KXCPI-26JAN-T0.001", 0.90)),
            (0.002, _make_contract("KXCPI-26JAN-T0.002", 0.65)),
            (0.003, _make_contract("KXCPI-26JAN-T0.003", 0.50)),
            (0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08)),
        ]
        results = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        # The tail at T0.004 should be flagged (anchor ≈ 25¢ vs market 8¢)
        flagged = [r for r in results if r.contract.contract_id == "KXCPI-26JAN-T0.004"]
        assert len(flagged) == 1
        assert flagged[0].expected_value > 0.10  # big edge

    def test_no_false_positives_on_consistent_pricing(self):
        # Prices roughly match normal CDF with μ=0.003, σ=0.0015
        from arbiter.scoring.anchor import compute_anchor_prob

        mu, sigma = 0.003, 0.0015
        group = [
            (0.001, _make_contract("KXCPI-26JAN-T0.001", compute_anchor_prob(0.001, mu, sigma))),
            (0.002, _make_contract("KXCPI-26JAN-T0.002", compute_anchor_prob(0.002, mu, sigma))),
            (0.003, _make_contract("KXCPI-26JAN-T0.003", compute_anchor_prob(0.003, mu, sigma))),
            (0.004, _make_contract("KXCPI-26JAN-T0.004", compute_anchor_prob(0.004, mu, sigma))),
        ]
        results = find_anchor_mispricings(group, mu=mu, sigma=sigma, fee_rate=0.01)
        assert results == []

    def test_fee_eliminates_small_edge(self):
        # Anchor prob slightly above market, but fee wipes out edge
        group = [
            (0.003, _make_contract("KXCPI-26JAN-T0.003", 0.49)),
        ]
        # With μ=0.003, σ=0.0015: anchor ≈ 0.50, edge = 0.50 - 0.49 - 0.02 < 0
        results = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.02)
        assert results == []

    def test_only_flags_underpriced_yes(self):
        # Market ABOVE anchor → overpriced, should NOT be flagged
        group = [
            (0.003, _make_contract("KXCPI-26JAN-T0.003", 0.70)),  # anchor ≈ 50¢
        ]
        results = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        assert results == []

    def test_kelly_sizing_positive(self):
        group = [
            (0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08)),
        ]
        results = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        assert len(results) == 1
        assert results[0].kelly_size > 0

    def test_strategy_name_set(self):
        group = [
            (0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08)),
        ]
        results = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        assert len(results) == 1
        assert results[0].strategy_name == "AnchorStrategy"

    def test_direction_always_yes(self):
        group = [
            (0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08)),
        ]
        results = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        assert all(r.direction == "yes" for r in results)

    def test_empty_group_returns_empty(self):
        results = find_anchor_mispricings([], mu=0.003, sigma=0.0015)
        assert results == []

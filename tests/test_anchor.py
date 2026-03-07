"""Tests for anchor scoring functions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbiter.ingestion.base import Contract
from arbiter.scoring.anchor import (
    PlattCalibrator,
    compute_anchor_prob,
    extract_threshold,
    find_anchor_mispricings,
    group_anchor_contracts,
)
from arbiter.scoring.fees import kalshi_fee


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

    def test_threshold_scale_converts_kalshi_pct_to_fred_decimal(self):
        # Kalshi T0.3 means 0.3% MoM; FRED stores in decimal (0.003).
        # With threshold_scale=0.01: 0.3 * 0.01 = 0.003 → same result as
        # calling directly with threshold=0.003 and no scaling.
        group_raw = [(0.003, _make_contract("KXCPI-26JAN-T0.003", 0.08))]
        group_pct = [(0.3, _make_contract("KXCPI-26JAN-T0.003", 0.08))]
        results_raw = find_anchor_mispricings(group_raw, mu=0.003, sigma=0.0015, fee_rate=0.01)
        results_scaled = find_anchor_mispricings(
            group_pct, mu=0.003, sigma=0.0015, fee_rate=0.01, threshold_scale=0.01
        )
        assert len(results_scaled) == len(results_raw)
        if results_raw:
            assert results_scaled[0].expected_value == pytest.approx(results_raw[0].expected_value)

    def test_threshold_scale_default_one_preserves_existing_behaviour(self):
        group = [(0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08))]
        results_default = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        results_explicit = find_anchor_mispricings(
            group, mu=0.003, sigma=0.0015, fee_rate=0.01, threshold_scale=1.0
        )
        assert len(results_default) == len(results_explicit)

    def test_calibrator_adjusts_model_probability(self):
        """Calibrator.predict() is applied to anchor_prob before scoring."""

        class _HalfCalibrator:
            def predict(self, x: list[float]) -> list[float]:
                return [v * 0.5 for v in x]

        group = [(0.004, _make_contract("KXCPI-26JAN-T0.004", 0.05))]
        raw_prob = compute_anchor_prob(0.004, mu=0.003, sigma=0.0015)
        results = find_anchor_mispricings(
            group, mu=0.003, sigma=0.0015, fee_rate=0.01, calibrator=_HalfCalibrator()
        )
        assert len(results) == 1
        assert results[0].model_probability == pytest.approx(raw_prob * 0.5, abs=0.001)

    def test_calibrator_can_eliminate_opportunity(self):
        """Calibrator pushing prob to zero removes what was an opportunity."""

        class _ZeroCalibrator:
            def predict(self, x: list[float]) -> list[float]:
                return [0.0 for _ in x]

        group = [(0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08))]
        results = find_anchor_mispricings(
            group, mu=0.003, sigma=0.0015, fee_rate=0.01, calibrator=_ZeroCalibrator()
        )
        assert results == []

    def test_no_calibrator_preserves_raw_prob(self):
        """calibrator=None produces identical results to no calibrator argument."""
        group = [(0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08))]
        results_default = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_rate=0.01)
        results_none = find_anchor_mispricings(
            group, mu=0.003, sigma=0.0015, fee_rate=0.01, calibrator=None
        )
        assert len(results_none) == len(results_default)
        if results_default:
            assert results_none[0].model_probability == pytest.approx(
                results_default[0].model_probability
            )


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class TestPlattCalibrator:
    def _fit_simple(self) -> PlattCalibrator:
        """Fit on data where high probs → outcome 1, low probs → outcome 0."""
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 5
        outcomes = [0, 0, 0, 1, 1, 1] * 5
        return PlattCalibrator().fit(probs, outcomes)

    def test_unfitted_predict_returns_input(self):
        cal = PlattCalibrator()
        result = cal.predict([0.2, 0.5, 0.8])
        assert result == [0.2, 0.5, 0.8]

    def test_unfitted_coef_returns_identity(self):
        cal = PlattCalibrator()
        assert cal.coef == (1.0, 0.0)

    def test_fit_predict_roundtrip(self):
        cal = self._fit_simple()
        result = cal.predict([0.1, 0.5, 0.9])
        assert len(result) == 3
        assert all(0 <= p <= 1 for p in result)

    def test_monotonically_increasing(self):
        cal = self._fit_simple()
        inputs = [0.1, 0.3, 0.5, 0.7, 0.9]
        outputs = cal.predict(inputs)
        for i in range(len(outputs) - 1):
            assert outputs[i] < outputs[i + 1]

    def test_coef_nontrivial_after_fit(self):
        cal = self._fit_simple()
        slope, intercept = cal.coef
        assert slope != 1.0 or intercept != 0.0

    def test_boundary_inputs(self):
        cal = self._fit_simple()
        result = cal.predict([0.0, 1.0])
        assert len(result) == 2
        assert all(0 <= p <= 1 for p in result)

    def test_empty_input(self):
        cal = self._fit_simple()
        assert cal.predict([]) == []

    def test_unfitted_empty_input(self):
        cal = PlattCalibrator()
        assert cal.predict([]) == []


class TestFindAnchorMispricingsFeeFn:
    def test_fee_fn_applies_parabolic_fee(self):
        group = [
            (0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08)),
        ]
        fn_result = find_anchor_mispricings(group, mu=0.003, sigma=0.0015, fee_fn=kalshi_fee)
        # kalshi_fee(0.08, True) = ceil(0.07*0.08*0.92 * 100)/100 = ceil(0.5152)/100 = 0.01
        # Same as flat 0.01 in this case, but uses the fee_fn code path
        assert len(fn_result) == 1
        assert fn_result[0].expected_value > 0

    def test_fee_fn_overrides_fee_rate(self):
        group = [
            (0.004, _make_contract("KXCPI-26JAN-T0.004", 0.08)),
        ]
        # fee_rate=0.99 would kill any edge, but fee_fn overrides it
        result = find_anchor_mispricings(
            group, mu=0.003, sigma=0.0015, fee_rate=0.99, fee_fn=kalshi_fee
        )
        assert len(result) == 1

"""Tests for Kelly criterion and expected value scoring."""

from datetime import UTC, datetime

import pytest

from arbiter.ingestion.base import Contract
from arbiter.scoring.ev import ScoredOpportunity, compute_ev
from arbiter.scoring.fees import flat_fee_rate, kalshi_fee
from arbiter.scoring.kelly import kelly_criterion


def _make_contract(
    yes_price: float = 0.50,
    no_price: float = 0.50,
    yes_bid: float = 0.49,
    yes_ask: float = 0.51,
) -> Contract:
    """Helper to create a test contract."""
    return Contract(
        source="kalshi",
        contract_id="TEST-001",
        title="Test market",
        category="test",
        yes_price=yes_price,
        no_price=no_price,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last_price=None,
        volume_24h=10000.0,
        open_interest=5000.0,
        expires_at=datetime(2026, 12, 31, tzinfo=UTC),
        url="https://example.com/test",
        status="open",
    )


# --- Kelly Criterion Tests ---


class TestKellyCriterion:
    def test_positive_edge(self):
        # 60% win prob, 1:1 payout -> kelly = (1*0.6 - 0.4)/1 = 0.2
        assert kelly_criterion(0.6, 1.0) == pytest.approx(0.2)

    def test_large_edge(self):
        # 80% win prob, 1:1 payout -> kelly = 0.6
        assert kelly_criterion(0.8, 1.0) == pytest.approx(0.6)

    def test_no_edge(self):
        # 50% win prob, 1:1 payout -> kelly = 0
        assert kelly_criterion(0.5, 1.0) == pytest.approx(0.0)

    def test_negative_edge(self):
        # 30% win prob, 1:1 payout -> kelly = 0 (clamped)
        assert kelly_criterion(0.3, 1.0) == 0.0

    def test_high_payout_ratio(self):
        # 20% win prob, 9:1 payout -> kelly = (9*0.2 - 0.8)/9 = 0.111...
        assert kelly_criterion(0.2, 9.0) == pytest.approx(1.0 / 9.0, rel=1e-6)

    def test_zero_payout_ratio(self):
        assert kelly_criterion(0.6, 0.0) == 0.0

    def test_negative_payout_ratio(self):
        assert kelly_criterion(0.6, -1.0) == 0.0

    def test_zero_win_prob(self):
        assert kelly_criterion(0.0, 1.0) == 0.0

    def test_certainty(self):
        # win_prob = 1.0 should return 0 (degenerate case)
        assert kelly_criterion(1.0, 1.0) == 0.0

    def test_near_certainty(self):
        # 99% win prob, 1:1 -> kelly = 0.98
        assert kelly_criterion(0.99, 1.0) == pytest.approx(0.98)

    def test_prediction_market_payout(self):
        # Contract priced at 0.40, payout_ratio = (1/0.4) - 1 = 1.5
        # Win prob 0.55 -> kelly = (1.5*0.55 - 0.45)/1.5 = 0.25
        payout = (1.0 / 0.40) - 1.0
        assert kelly_criterion(0.55, payout) == pytest.approx(0.25)


# --- EV Scoring Tests ---


class TestComputeEV:
    def test_identifies_yes_edge(self):
        contract = _make_contract(yes_price=0.40, no_price=0.60)
        results = compute_ev(contract, model_prob_yes=0.60)

        yes_result = next(r for r in results if r.direction == "yes")
        assert yes_result.expected_value == pytest.approx(0.20)
        assert yes_result.kelly_size > 0
        assert yes_result.market_price == 0.40
        assert yes_result.model_probability == 0.60

    def test_identifies_no_edge(self):
        contract = _make_contract(yes_price=0.70, no_price=0.30)
        results = compute_ev(contract, model_prob_yes=0.50)

        no_result = next(r for r in results if r.direction == "no")
        assert no_result.expected_value == pytest.approx(0.20)
        assert no_result.model_probability == 0.50

    def test_no_edge_when_model_agrees(self):
        contract = _make_contract(yes_price=0.50, no_price=0.50)
        results = compute_ev(contract, model_prob_yes=0.50)

        for r in results:
            assert r.expected_value == pytest.approx(0.0)
            assert r.kelly_size == pytest.approx(0.0)

    def test_returns_both_directions(self):
        contract = _make_contract(yes_price=0.50, no_price=0.50)
        results = compute_ev(contract, model_prob_yes=0.60)

        directions = {r.direction for r in results}
        assert directions == {"yes", "no"}

    def test_fee_rate_reduces_ev(self):
        contract = _make_contract(yes_price=0.40, no_price=0.60)

        no_fee = compute_ev(contract, model_prob_yes=0.60, fee_rate=0.0)
        with_fee = compute_ev(contract, model_prob_yes=0.60, fee_rate=0.02)

        yes_no_fee = next(r for r in no_fee if r.direction == "yes")
        yes_with_fee = next(r for r in with_fee if r.direction == "yes")

        assert yes_with_fee.expected_value < yes_no_fee.expected_value
        assert yes_no_fee.expected_value == pytest.approx(0.20)
        assert yes_with_fee.expected_value == pytest.approx(0.18)

    def test_fee_can_eliminate_edge(self):
        contract = _make_contract(yes_price=0.49, no_price=0.51)
        results = compute_ev(contract, model_prob_yes=0.50, fee_rate=0.02)

        # With 2% fee on a 1% edge, both sides should be negative
        for r in results:
            assert r.expected_value < 0

    def test_extreme_model_confidence(self):
        contract = _make_contract(yes_price=0.10, no_price=0.90)
        results = compute_ev(contract, model_prob_yes=0.95)

        yes_result = next(r for r in results if r.direction == "yes")
        assert yes_result.expected_value == pytest.approx(0.85)
        assert yes_result.kelly_size > 0

    def test_scored_opportunity_has_contract_ref(self):
        contract = _make_contract()
        results = compute_ev(contract, model_prob_yes=0.60)
        assert all(isinstance(r, ScoredOpportunity) for r in results)
        assert all(r.contract is contract for r in results)

    def test_high_fee_eliminates_yes_side(self):
        contract = _make_contract(yes_price=0.60, no_price=0.40)
        results = compute_ev(contract, model_prob_yes=0.60, fee_rate=0.50)
        # yes_cost = 0.60 + 0.50 = 1.10 > 1, so YES side is dropped
        directions = {r.direction for r in results}
        assert "yes" not in directions

    def test_high_fee_eliminates_both_sides(self):
        contract = _make_contract(yes_price=0.80, no_price=0.20)
        results = compute_ev(contract, model_prob_yes=0.50, fee_rate=0.90)
        # yes_cost = 1.70, no_cost = 1.10, both > 1
        assert len(results) == 0

    def test_extreme_low_price(self):
        contract = _make_contract(yes_price=0.001, no_price=0.999)
        results = compute_ev(contract, model_prob_yes=0.50)
        assert len(results) == 2
        yes_result = next(r for r in results if r.direction == "yes")
        assert yes_result.kelly_size > 0

    def test_fee_fn_overrides_fee_rate(self):
        contract = _make_contract(yes_price=0.40, no_price=0.60)
        # fee_fn should override fee_rate when both provided
        results = compute_ev(
            contract, model_prob_yes=0.60, fee_rate=0.10, fee_fn=flat_fee_rate(0.01)
        )
        yes_result = next(r for r in results if r.direction == "yes")
        # EV = 0.60 - 0.40 - 0.01 = 0.19 (uses fee_fn=0.01, not fee_rate=0.10)
        assert yes_result.expected_value == pytest.approx(0.19)

    def test_fee_fn_kalshi_parabolic(self):
        contract = _make_contract(yes_price=0.50, no_price=0.50)
        results = compute_ev(contract, model_prob_yes=0.70, fee_fn=kalshi_fee)
        yes_result = next(r for r in results if r.direction == "yes")
        # kalshi_fee(0.50, True) = 0.02; EV = 0.70 - 0.50 - 0.02 = 0.18
        assert yes_result.expected_value == pytest.approx(0.18)

    def test_fee_fn_maker_vs_taker(self):
        contract = _make_contract(yes_price=0.50, no_price=0.50)
        taker = compute_ev(contract, model_prob_yes=0.70, fee_fn=kalshi_fee, is_taker=True)
        maker = compute_ev(contract, model_prob_yes=0.70, fee_fn=kalshi_fee, is_taker=False)
        yes_taker = next(r for r in taker if r.direction == "yes")
        yes_maker = next(r for r in maker if r.direction == "yes")
        # Maker fee < taker fee, so maker EV should be higher
        assert yes_maker.expected_value > yes_taker.expected_value

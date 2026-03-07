"""Tests for range-market internal consistency scoring."""

from datetime import UTC, datetime

import pytest

from arbiter.ingestion.base import Contract
from arbiter.scoring.consistency import find_consistency_violations
from arbiter.scoring.fees import kalshi_fee


def _make_range_contract(
    contract_id: str,
    title: str,
    yes_price: float,
    source: str = "kalshi",
) -> Contract:
    return Contract(
        source=source,
        contract_id=contract_id,
        title=title,
        category="",
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 6),
        yes_bid=round(yes_price - 0.01, 6),
        yes_ask=round(yes_price + 0.01, 6),
        last_price=None,
        volume_24h=100.0,
        open_interest=500.0,
        expires_at=datetime(2026, 3, 31, tzinfo=UTC),
        url="https://kalshi.com/test",
        status="open",
    )


# --- "above" (MAX) group helpers ---


def _btc_above(threshold_cents: int, yes_price: float) -> Contract:
    cid = f"KXBTCMAXMON-BTC-26MAR31-{threshold_cents}"
    title = f"Will BTC trimmed mean be above ${threshold_cents // 100:.2f} by Mar 31?"
    return _make_range_contract(cid, title, yes_price)


# --- "below" (MIN) group helpers ---


def _btc_below(threshold_cents: int, yes_price: float) -> Contract:
    cid = f"KXBTCMINMON-BTC-26MAR31-{threshold_cents}"
    title = f"Will BTC trimmed mean be below ${threshold_cents // 100:.2f} by Mar 31?"
    return _make_range_contract(cid, title, yes_price)


class TestFindConsistencyViolations:
    # --- "above" (MAX) groups ---

    def test_no_violations_in_consistent_above_group(self):
        contracts = [
            _btc_above(7000000, 0.40),
            _btc_above(7500000, 0.33),
            _btc_above(8000000, 0.25),
            _btc_above(8500000, 0.15),
        ]
        result = find_consistency_violations(contracts)
        assert result == []

    def test_detects_underpriced_above_contract(self):
        # P(above $80k)=0.31 < P(above $82.5k)=0.50 — $80k is underpriced
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
            _btc_above(8500000, 0.15),
        ]
        result = find_consistency_violations(contracts)
        assert len(result) == 1
        assert result[0].contract.contract_id == "KXBTCMAXMON-BTC-26MAR31-8000000"
        assert result[0].direction == "yes"

    def test_ev_equals_max_higher_prob_minus_price_minus_fee(self):
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
            _btc_above(8500000, 0.20),
        ]
        result = find_consistency_violations(contracts, fee_rate=0.01)
        assert len(result) == 1
        # model_prob = max(0.50, 0.20) = 0.50; EV = 0.50 - 0.31 - 0.01 = 0.18
        assert result[0].model_probability == pytest.approx(0.50)
        assert result[0].expected_value == pytest.approx(0.18)

    def test_model_prob_uses_max_of_all_higher_probs(self):
        # Violation at $70k; higher probs are 0.30, 0.55, 0.20 — max is 0.55
        contracts = [
            _btc_above(7000000, 0.28),
            _btc_above(7500000, 0.30),
            _btc_above(8000000, 0.55),
            _btc_above(8500000, 0.20),
        ]
        result = find_consistency_violations(contracts)
        flagged = [r for r in result if r.contract.contract_id == "KXBTCMAXMON-BTC-26MAR31-7000000"]
        assert len(flagged) == 1
        assert flagged[0].model_probability == pytest.approx(0.55)

    def test_multiple_violations_in_one_group(self):
        # Both $70k and $75k are underpriced relative to $80k=0.55
        contracts = [
            _btc_above(7000000, 0.28),
            _btc_above(7500000, 0.30),
            _btc_above(8000000, 0.55),
        ]
        result = find_consistency_violations(contracts)
        flagged_ids = {r.contract.contract_id for r in result}
        assert "KXBTCMAXMON-BTC-26MAR31-7000000" in flagged_ids
        assert "KXBTCMAXMON-BTC-26MAR31-7500000" in flagged_ids

    def test_no_violation_when_fee_eliminates_edge(self):
        # Apparent edge = 0.50 - 0.49 = 0.01, but fee=0.02 wipes it out
        contracts = [
            _btc_above(8000000, 0.49),
            _btc_above(8250000, 0.50),
        ]
        result = find_consistency_violations(contracts, fee_rate=0.02)
        assert result == []

    # --- "below" (MIN) groups ---

    def test_no_violations_in_consistent_below_group(self):
        contracts = [
            _btc_below(5000000, 0.10),
            _btc_below(5500000, 0.20),
            _btc_below(6000000, 0.35),
            _btc_below(6500000, 0.55),
        ]
        result = find_consistency_violations(contracts)
        assert result == []

    def test_detects_underpriced_below_contract(self):
        # P(below $57.5k)=0.38 < P(below $55k)=0.53 — $57.5k is underpriced
        contracts = [
            _btc_below(5500000, 0.53),
            _btc_below(5750000, 0.38),
            _btc_below(6000000, 0.64),
        ]
        result = find_consistency_violations(contracts)
        flagged = [r for r in result if r.contract.contract_id == "KXBTCMINMON-BTC-26MAR31-5750000"]
        assert len(flagged) == 1
        assert flagged[0].direction == "yes"

    def test_below_ev_calculation(self):
        contracts = [
            _btc_below(5500000, 0.53),
            _btc_below(5750000, 0.38),
        ]
        result = find_consistency_violations(contracts, fee_rate=0.01)
        assert len(result) == 1
        # model_prob = 0.53; EV = 0.53 - 0.38 - 0.01 = 0.14
        assert result[0].model_probability == pytest.approx(0.53)
        assert result[0].expected_value == pytest.approx(0.14)

    # --- Filtering ---

    def test_single_contract_group_produces_no_violations(self):
        contracts = [_btc_above(8000000, 0.31)]
        result = find_consistency_violations(contracts)
        assert result == []

    def test_empty_input_returns_empty(self):
        assert find_consistency_violations([]) == []

    def test_polymarket_contracts_are_ignored(self):
        c = _make_range_contract(
            "KXBTCMAXMON-BTC-26MAR31-8000000",
            "Will BTC be above $80000?",
            yes_price=0.31,
            source="polymarket",
        )
        result = find_consistency_violations([c, _btc_above(8250000, 0.50)])
        # Only the polymarket contract is at $80k; kalshi $82.5k has no partner
        assert all(r.contract.source == "kalshi" for r in result)

    def test_non_range_contract_without_numeric_suffix_is_ignored(self):
        non_range = _make_range_contract(
            "KXTRUMP-NOMINEE-26MAR",  # no trailing -DIGITS
            "Will Trump nominate X?",
            yes_price=0.20,
        )
        contracts = [non_range, _btc_above(8250000, 0.50)]
        # Non-range contract cannot form a group — no violation
        result = find_consistency_violations(contracts)
        assert result == []

    def test_minden_team_code_is_not_treated_as_below_group(self):
        """MINDEN (Minnesota/Denver) contains 'MIN' but is NOT a price-range series."""

        # These are NBA player-prop markets, not min/max price brackets.
        # If mis-classified as a "below" group the descending YES prices would
        # incorrectly generate false violations.
        def _nba_prop(suffix: int, yes_price: float) -> Contract:
            cid = f"KXNBAPTS-26MAR01MINDEN-MINAEDWARDS5-{suffix}"
            return _make_range_contract(cid, f"Anthony Edwards: {suffix}+ points", yes_price)

        contracts = [
            _nba_prop(20, 0.785),
            _nba_prop(25, 0.720),
            _nba_prop(30, 0.475),
            _nba_prop(35, 0.265),
            _nba_prop(40, 0.075),
        ]
        # Prices are monotonically consistent (lower threshold = higher prob).
        # No violation should be reported.
        result = find_consistency_violations(contracts)
        assert result == []

    def test_contracts_from_different_groups_do_not_cross_contaminate(self):
        btc_contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
        ]

        def _eth_above(threshold_cents: int, yes_price: float) -> Contract:
            cid = f"KXETHMAXMON-ETH-26MAR31-{threshold_cents}"
            return _make_range_contract(cid, f"ETH above ${threshold_cents}", yes_price)

        eth_contracts = [
            _eth_above(200000, 0.45),
            _eth_above(250000, 0.25),
        ]
        result = find_consistency_violations(btc_contracts + eth_contracts)
        ids = {r.contract.contract_id for r in result}
        # BTC $80k is underpriced, ETH is consistent — only one violation
        assert "KXBTCMAXMON-BTC-26MAR31-8000000" in ids
        assert not any("ETH" in i for i in ids)

    def test_zero_volume_anchor_is_ignored(self):
        """A sibling at 0.50 with zero volume is an uninformed default, not a real anchor."""

        def _zero_vol(cid: str, yes_price: float) -> Contract:
            c = _make_range_contract(cid, f"BTC above {cid}", yes_price)
            # Override volume_24h to zero via a new Contract (frozen dataclass)
            from dataclasses import replace

            return replace(c, volume_24h=0.0)

        contracts = [
            _btc_above(8000000, 0.31),
            _zero_vol("KXBTCMAXMON-BTC-26MAR31-8250000", 0.50),  # uninformed default
        ]
        result = find_consistency_violations(contracts)
        assert result == []

    def test_positive_volume_anchor_is_used(self):
        """A sibling with real volume produces a valid floor."""
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),  # has default volume=100.0 from _make_range_contract
        ]
        result = find_consistency_violations(contracts)
        assert len(result) == 1

    def test_anchor_contract_is_set_to_floor_sibling(self):
        """The anchor_contract should be the sibling that provides the price floor."""
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
            _btc_above(8500000, 0.20),
        ]
        result = find_consistency_violations(contracts)
        assert len(result) == 1
        # The anchor is the $82.5k contract (yes_price=0.50, the max higher sibling)
        assert result[0].anchor_contract is not None
        assert result[0].anchor_contract.contract_id == "KXBTCMAXMON-BTC-26MAR31-8250000"

    def test_anchor_contract_is_max_priced_sibling_for_above(self):
        """When multiple siblings are higher-priced, anchor is the one with max price."""
        contracts = [
            _btc_above(7000000, 0.28),
            _btc_above(7500000, 0.30),
            _btc_above(8000000, 0.55),
            _btc_above(8500000, 0.40),
        ]
        result = find_consistency_violations(contracts)
        flagged = [r for r in result if r.contract.contract_id == "KXBTCMAXMON-BTC-26MAR31-7000000"]
        assert len(flagged) == 1
        # Anchor is $80k (0.55) not $85k (0.40) — it's the max
        assert flagged[0].anchor_contract is not None
        assert flagged[0].anchor_contract.contract_id == "KXBTCMAXMON-BTC-26MAR31-8000000"

    def test_anchor_contract_for_below_group(self):
        """Below groups pick the anchor from lower-threshold siblings."""
        contracts = [
            _btc_below(5500000, 0.53),
            _btc_below(5750000, 0.38),
        ]
        result = find_consistency_violations(contracts)
        assert len(result) == 1
        assert result[0].anchor_contract is not None
        assert result[0].anchor_contract.contract_id == "KXBTCMINMON-BTC-26MAR31-5500000"

    def test_kelly_size_positive_for_violation(self):
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
        ]
        result = find_consistency_violations(contracts)
        assert len(result) == 1
        assert result[0].kelly_size > 0

    def test_fee_fn_applies_parabolic_fee(self):
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
        ]
        flat_result = find_consistency_violations(contracts, fee_rate=0.01)
        fn_result = find_consistency_violations(contracts, fee_fn=kalshi_fee)
        # kalshi_fee(0.31, True) = ceil(0.07*0.31*0.69 * 100)/100 = ceil(1.4973)/100 = 0.02
        # Flat fee = 0.01; parabolic fee = 0.02 → EV lower with fee_fn
        assert fn_result[0].expected_value < flat_result[0].expected_value

    def test_fee_fn_overrides_fee_rate(self):
        contracts = [
            _btc_above(8000000, 0.31),
            _btc_above(8250000, 0.50),
        ]
        # When both provided, fee_fn takes precedence
        result = find_consistency_violations(contracts, fee_rate=0.99, fee_fn=kalshi_fee)
        assert len(result) == 1  # fee_rate=0.99 would eliminate edge, but fee_fn is used

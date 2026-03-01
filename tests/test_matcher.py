"""Tests for cross-platform contract matching."""

from datetime import UTC, datetime

from arbiter.ingestion.base import Contract
from arbiter.ingestion.matcher import ContractMatcher


def _kalshi(
    contract_id: str = "K-BTC",
    title: str = "Bitcoin above $100k?",
    category: str = "Crypto",
    expires_at: datetime | None = datetime(2026, 3, 14, tzinfo=UTC),
) -> Contract:
    return Contract(
        source="kalshi",
        contract_id=contract_id,
        title=title,
        category=category,
        yes_price=0.60,
        no_price=0.40,
        yes_bid=0.59,
        yes_ask=0.61,
        last_price=0.60,
        volume_24h=10000.0,
        open_interest=50000.0,
        expires_at=expires_at,
        url="https://kalshi.com/test",
        status="open",
    )


def _poly(
    contract_id: str = "P-BTC",
    title: str = "Will Bitcoin hit $100k?",
    category: str = "Crypto",
    expires_at: datetime | None = datetime(2026, 3, 14, tzinfo=UTC),
) -> Contract:
    return Contract(
        source="polymarket",
        contract_id=contract_id,
        title=title,
        category=category,
        yes_price=0.62,
        no_price=0.38,
        yes_bid=0.61,
        yes_ask=0.63,
        last_price=None,
        volume_24h=200000.0,
        open_interest=100000.0,
        expires_at=expires_at,
        url="https://polymarket.com/test",
        status="open",
    )


class TestContractMatcherBasic:
    def test_matches_same_category_and_expiry(self):
        matcher = ContractMatcher()
        matches = matcher.match([_kalshi()], [_poly()])
        assert len(matches) == 1
        assert matches[0].kalshi.contract_id == "K-BTC"
        assert matches[0].polymarket.contract_id == "P-BTC"

    def test_match_has_confidence_score(self):
        matcher = ContractMatcher()
        matches = matcher.match([_kalshi()], [_poly()])
        assert 0 < matches[0].confidence <= 1.0

    def test_no_match_different_category(self):
        k = _kalshi(category="Crypto")
        p = _poly(category="Politics")
        matcher = ContractMatcher()
        matches = matcher.match([k], [p])
        assert len(matches) == 0

    def test_no_match_different_expiry(self):
        k = _kalshi(expires_at=datetime(2026, 3, 14, tzinfo=UTC))
        p = _poly(expires_at=datetime(2026, 12, 31, tzinfo=UTC))
        matcher = ContractMatcher()
        matches = matcher.match([k], [p])
        assert len(matches) == 0

    def test_empty_inputs(self):
        matcher = ContractMatcher()
        assert matcher.match([], [_poly()]) == []
        assert matcher.match([_kalshi()], []) == []
        assert matcher.match([], []) == []

    def test_case_insensitive_category_matching(self):
        k = _kalshi(category="crypto")
        p = _poly(category="CRYPTO")
        matcher = ContractMatcher()
        matches = matcher.match([k], [p])
        assert len(matches) == 1


class TestContractMatcherConfidence:
    def test_higher_confidence_for_similar_titles(self):
        k = _kalshi(title="Bitcoin above $100k by March?")
        p_similar = _poly(title="Bitcoin above $100k by March 2026?")
        p_different = _poly(
            contract_id="P-OTHER",
            title="Crypto market cap above 5T?",
        )
        matcher = ContractMatcher()
        match_similar = matcher.match([k], [p_similar])
        match_diff = matcher.match([k], [p_different])

        assert len(match_similar) == 1
        # Different title with same category/expiry may or may not match
        # but if it does, confidence should be lower
        if match_diff:
            assert match_similar[0].confidence > match_diff[0].confidence

    def test_min_confidence_threshold(self):
        k = _kalshi(title="Bitcoin above $100k?", category="Crypto")
        p = _poly(title="Completely unrelated question about weather", category="Crypto")
        matcher = ContractMatcher()
        # With a high threshold, weak title similarity should be filtered
        matches = matcher.match([k], [p], min_confidence=0.95)
        assert len(matches) == 0

    def test_confidence_bounded_zero_to_one(self):
        matcher = ContractMatcher()
        matches = matcher.match([_kalshi()], [_poly()])
        for m in matches:
            assert 0.0 <= m.confidence <= 1.0


class TestContractMatcherMultiple:
    def test_matches_multiple_pairs(self):
        k1 = _kalshi(contract_id="K-BTC", title="Bitcoin above $100k?", category="Crypto")
        k2 = _kalshi(
            contract_id="K-FED",
            title="Fed cuts rates?",
            category="Economics",
            expires_at=datetime(2026, 3, 19, tzinfo=UTC),
        )
        p1 = _poly(contract_id="P-BTC", title="Will Bitcoin hit $100k?", category="Crypto")
        p2 = _poly(
            contract_id="P-FED",
            title="Federal Reserve rate cut?",
            category="Economics",
            expires_at=datetime(2026, 3, 19, tzinfo=UTC),
        )
        matcher = ContractMatcher()
        matches = matcher.match([k1, k2], [p1, p2], min_confidence=0.5)

        kalshi_ids = {m.kalshi.contract_id for m in matches}
        poly_ids = {m.polymarket.contract_id for m in matches}
        assert "K-BTC" in kalshi_ids
        assert "P-BTC" in poly_ids

    def test_one_to_one_matching(self):
        """Each contract should appear in at most one match."""
        k = _kalshi()
        p1 = _poly(contract_id="P-1")
        p2 = _poly(contract_id="P-2")
        matcher = ContractMatcher()
        matches = matcher.match([k], [p1, p2])
        # Kalshi contract should match at most once
        assert len(matches) <= 1

    def test_none_expiry_still_matches_with_similar_titles(self):
        """Contracts without expiry dates can still match if titles are similar."""
        k = _kalshi(expires_at=None)
        p = _poly(expires_at=None)
        matcher = ContractMatcher()
        # Lower threshold since unknown expiry reduces confidence
        matches = matcher.match([k], [p], min_confidence=0.6)
        assert len(matches) == 1

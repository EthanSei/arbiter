"""Cross-platform contract matching between Kalshi and Polymarket."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from arbiter.ingestion.base import Contract

_EXPIRY_TOLERANCE = timedelta(days=3)
_CATEGORY_WEIGHT = 0.40
_EXPIRY_WEIGHT = 0.35
_TITLE_WEIGHT = 0.25


@dataclass
class ContractMatch:
    """A matched pair of contracts across platforms with a confidence score."""

    kalshi: Contract
    polymarket: Contract
    confidence: float  # 0-1, only use matches above a threshold


class ContractMatcher:
    """Matches contracts across Kalshi and Polymarket.

    Uses structured fields (category, expiry date, normalized entity) as
    the primary matching signal, with title similarity as a tiebreaker.
    Only high-confidence matches produce cross-platform features.
    """

    def match(
        self,
        kalshi_contracts: list[Contract],
        polymarket_contracts: list[Contract],
        min_confidence: float = 0.8,
    ) -> list[ContractMatch]:
        if not kalshi_contracts or not polymarket_contracts:
            return []

        matches: list[ContractMatch] = []
        used_poly: set[str] = set()

        for k in kalshi_contracts:
            best: ContractMatch | None = None
            for p in polymarket_contracts:
                if p.contract_id in used_poly:
                    continue
                conf = _compute_confidence(k, p)
                if conf >= min_confidence and (best is None or conf > best.confidence):
                    best = ContractMatch(kalshi=k, polymarket=p, confidence=conf)
            if best is not None:
                matches.append(best)
                used_poly.add(best.polymarket.contract_id)

        return matches


def _compute_confidence(k: Contract, p: Contract) -> float:
    cat_score = 1.0 if k.category.lower() == p.category.lower() else 0.0
    if cat_score == 0.0:
        return 0.0  # Category mismatch is a hard filter

    expiry_score = _expiry_similarity(k, p)
    if expiry_score == 0.0 and k.expires_at is not None and p.expires_at is not None:
        return 0.0  # Expiry mismatch with known dates is a hard filter

    title_score = _title_similarity(k.title, p.title)

    return (
        cat_score * _CATEGORY_WEIGHT + expiry_score * _EXPIRY_WEIGHT + title_score * _TITLE_WEIGHT
    )


def _expiry_similarity(k: Contract, p: Contract) -> float:
    if k.expires_at is None or p.expires_at is None:
        return 0.5  # Unknown expiry — neutral
    delta = abs(k.expires_at - p.expires_at)
    if delta <= _EXPIRY_TOLERANCE:
        return 1.0
    return 0.0


def _title_similarity(a: str, b: str) -> float:
    """Token-based Jaccard similarity between two titles."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum() or ch == " ":
            cleaned += ch
    return {w for w in cleaned.split() if len(w) > 1}

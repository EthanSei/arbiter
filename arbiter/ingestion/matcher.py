"""Cross-platform contract matching between Kalshi and Polymarket."""

from __future__ import annotations

from dataclasses import dataclass

from arbiter.ingestion.base import Contract


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
        """Find matching contracts across platforms.

        Args:
            kalshi_contracts: Contracts from Kalshi.
            polymarket_contracts: Contracts from Polymarket.
            min_confidence: Minimum confidence score to include a match.

        Returns:
            List of high-confidence contract matches.
        """
        raise NotImplementedError  # Phase 2

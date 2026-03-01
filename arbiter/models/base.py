"""Abstract base for probability estimation."""

from abc import ABC, abstractmethod

from arbiter.ingestion.base import Contract


class ProbabilityEstimator(ABC):
    """Estimates the true probability of a YES outcome for a market contract.

    Implementations may use ML models, LLM calls, or other approaches.
    The returned value should be a calibrated probability in (0, 1).
    """

    @abstractmethod
    async def estimate(self, contract: Contract) -> float:
        """Return estimated probability of YES outcome for the given contract."""
        ...

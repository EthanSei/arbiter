"""LightGBM-based probability estimator."""

from __future__ import annotations

from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator


class LGBMEstimator(ProbabilityEstimator):
    """Estimates contract probabilities using a trained LightGBM model.

    Loads model weights from disk. Applies isotonic calibration on top of
    raw model output. Falls back to the market midpoint if no model is loaded
    (cold start — produces no false signals).
    """

    def __init__(self, model_path: str | None = None) -> None:
        raise NotImplementedError  # Phase 3

    async def estimate(self, contract: Contract) -> float:
        raise NotImplementedError  # Phase 3

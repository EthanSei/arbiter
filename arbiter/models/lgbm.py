"""LightGBM-based probability estimator."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from arbiter.ingestion.base import Contract
from arbiter.models.base import ProbabilityEstimator
from arbiter.models.features import extract_features

logger = logging.getLogger(__name__)

# Clamp epsilon to avoid returning exactly 0 or 1
_EPS = 1e-6


class LGBMEstimator(ProbabilityEstimator):
    """Estimates contract probabilities using a trained LightGBM model.

    Loads model weights from disk. Applies isotonic calibration on top of
    raw model output. Falls back to the market midpoint if no model is loaded
    (cold start — produces no false signals).
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model: Any = None
        self._calibrator: Any = None

        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            with open(path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self._model = data["model"]
            self._calibrator = data.get("calibrator")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def calibrated(self) -> bool:
        return self._calibrator is not None

    async def estimate(self, contract: Contract) -> float:
        if not self.model_loaded:
            return float(np.clip(contract.yes_price, _EPS, 1.0 - _EPS))

        features = extract_features(contract)
        try:
            raw = self._model.predict(features.reshape(1, -1))
        except Exception:
            logger.warning(
                "Model prediction failed for %s — falling back to market midpoint",
                contract.contract_id,
                exc_info=True,
            )
            return float(np.clip(contract.yes_price, _EPS, 1.0 - _EPS))
        prob = float(raw[0])

        if self._calibrator is not None:
            prob = float(self._calibrator.predict(np.array([prob]))[0])

        if np.isnan(prob):
            return float(np.clip(contract.yes_price, _EPS, 1.0 - _EPS))

        return float(np.clip(prob, _EPS, 1.0 - _EPS))

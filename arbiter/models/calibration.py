"""Calibration wrappers for probability model post-processing."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Wraps LogisticRegression to present the same .predict() interface as
    IsotonicRegression, returning probabilities instead of class labels.

    This allows lgbm.py to call ``calibrator.predict(raw_probs)`` unchanged.
    """

    def __init__(self, lr: LogisticRegression) -> None:
        self.lr = lr

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return calibrated probabilities for raw model outputs."""
        probs: npt.NDArray[np.float64] = self.lr.predict_proba(
            np.asarray(x, dtype=np.float64).reshape(-1, 1)
        )[:, 1]
        return probs

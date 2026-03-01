"""Model evaluation utilities — calibration, Brier score, ECE."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def brier_score(probabilities: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64]) -> float:
    """Compute the Brier score (mean squared error of probabilistic predictions).

    Lower is better. Range [0, 1]. A perfectly calibrated model achieves
    the irreducible Brier score determined by the true probabilities.
    """
    raise NotImplementedError  # Phase 4


def expected_calibration_error(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the gap between
    predicted probability and observed frequency in each bin.
    """
    raise NotImplementedError  # Phase 4


def calibration_curve(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute a reliability diagram (calibration curve).

    Returns:
        Tuple of (mean_predicted, fraction_positive) for each bin.
    """
    raise NotImplementedError  # Phase 4

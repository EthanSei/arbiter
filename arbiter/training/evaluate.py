"""Model evaluation utilities — calibration, Brier score, ECE."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def brier_score(probabilities: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64]) -> float:
    """Compute the Brier score (mean squared error of probabilistic predictions).

    Lower is better. Range [0, 1]. A perfectly calibrated model achieves
    the irreducible Brier score determined by the true probabilities.
    """
    return float(np.mean((probabilities - outcomes) ** 2))


def expected_calibration_error(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the gap between
    predicted probability and observed frequency in each bin.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probabilities)

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
        n_bin = int(np.sum(mask))
        if n_bin == 0:
            continue
        avg_pred = float(np.mean(probabilities[mask]))
        avg_outcome = float(np.mean(outcomes[mask]))
        ece += (n_bin / n_total) * abs(avg_pred - avg_outcome)

    return ece


def calibration_curve(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute a reliability diagram (calibration curve).

    Returns:
        Tuple of (mean_predicted, fraction_positive) for non-empty bins only.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted: list[float] = []
    fraction_positive: list[float] = []

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
        if np.sum(mask) == 0:
            continue
        mean_predicted.append(float(np.mean(probabilities[mask])))
        fraction_positive.append(float(np.mean(outcomes[mask])))

    return np.array(mean_predicted, dtype=np.float64), np.array(fraction_positive, dtype=np.float64)

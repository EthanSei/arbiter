"""Dataset loading and preparation for model training."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sqlalchemy.ext.asyncio import AsyncSession


async def load_dataset(
    session: AsyncSession,
    feature_version: str | None = None,
    exclude_hours_before_resolution: float = 24.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load features and labels from MarketSnapshot table.

    Args:
        session: Database session.
        feature_version: Only load snapshots with this feature version.
            Defaults to the current FEATURE_VERSION.
        exclude_hours_before_resolution: Exclude snapshots within this many
            hours of market resolution to prevent leakage from converging prices.

    Returns:
        Tuple of (features, labels) numpy arrays.
    """
    raise NotImplementedError  # Phase 4


def temporal_split(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.float64],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Split data temporally into train/val/test sets.

    NOT random — splits by time so that train < val < test chronologically.
    This prevents temporal information leakage.

    Returns:
        Dict with keys "train", "val", "test", each mapping to (features, labels).
    """
    raise NotImplementedError  # Phase 4

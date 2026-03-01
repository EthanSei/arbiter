"""Dataset loading and preparation for model training."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import numpy.typing as npt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.db.models import MarketSnapshot
from arbiter.models.features import FEATURE_VERSION, SPEC


async def load_dataset(
    session: AsyncSession,
    feature_version: str | None = None,
    exclude_hours_before_resolution: float = 24.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load features, labels, and timestamps from MarketSnapshot table.

    Args:
        session: Database session.
        feature_version: Only load snapshots with this feature version.
            Defaults to the current FEATURE_VERSION.
        exclude_hours_before_resolution: Exclude snapshots within this many
            hours of market resolution to prevent leakage from converging prices.

    Returns:
        Tuple of (features, labels, timestamps) numpy arrays.
        timestamps contains Unix seconds from snapshot_at for temporal splitting.
    """
    version = feature_version if feature_version is not None else FEATURE_VERSION
    exclusion_window = timedelta(hours=exclude_hours_before_resolution)

    stmt = select(MarketSnapshot).where(
        MarketSnapshot.outcome.is_not(None),
        MarketSnapshot.feature_version == version,
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    feature_rows: list[list[float]] = []
    label_rows: list[float] = []
    timestamp_rows: list[float] = []

    for row in rows:
        if (
            row.resolved_at is not None
            and row.snapshot_at is not None
            and row.resolved_at - row.snapshot_at < exclusion_window
        ):
            continue
        if not row.features:
            continue
        vector = [float(row.features.get(name, float("nan"))) for name in SPEC.names]  # type: ignore[arg-type]
        feature_rows.append(vector)
        label_rows.append(float(row.outcome))  # type: ignore[arg-type]
        ts = row.snapshot_at.timestamp() if row.snapshot_at is not None else 0.0
        timestamp_rows.append(ts)

    if not feature_rows:
        empty_f = np.empty((0, len(SPEC.names)), dtype=np.float64)
        return empty_f, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    return (
        np.array(feature_rows, dtype=np.float64),
        np.array(label_rows, dtype=np.float64),
        np.array(timestamp_rows, dtype=np.float64),
    )


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
    sorted_idx = np.argsort(timestamps)
    features = features[sorted_idx]
    labels = labels[sorted_idx]

    n = len(features)
    test_size = int(n * test_fraction)
    val_size = int(n * val_fraction)
    train_size = n - val_size - test_size

    val_start = train_size
    test_start = train_size + val_size
    return {
        "train": (features[:train_size], labels[:train_size]),
        "val": (features[val_start:test_start], labels[val_start:test_start]),
        "test": (features[test_start:], labels[test_start:]),
    }

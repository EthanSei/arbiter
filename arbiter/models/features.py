"""Feature extraction for the probability estimation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from arbiter.ingestion.base import Contract

FEATURE_VERSION = "0.1.0"
"""Bump this when the feature schema changes. Stored in MarketSnapshot.feature_version
so training can filter by compatible versions."""


@dataclass
class FeatureSpec:
    """Documents the feature vector layout for the current FEATURE_VERSION."""

    names: list[str]
    version: str = FEATURE_VERSION


SPEC = FeatureSpec(
    names=[
        # Market features
        "yes_price",
        "no_price",
        "bid_ask_spread",
        "last_price",
        "log_volume_24h",
        "log_open_interest",
        "time_to_expiry_hours",
        "overround",
        "day_of_week",
        "hour_of_day",
        # Cross-platform features (NaN if no match)
        "price_discrepancy",
        "volume_ratio",
        # Lag features (NaN if no history)
        "price_delta_1h",
        "price_delta_24h",
        "volume_ratio_24h",
        "price_volatility_24h",
    ]
)


def extract_features(
    contract: Contract,
    cross_platform_price: float | None = None,
    cross_platform_volume: float | None = None,
    price_history: list[float] | None = None,
    volume_history: list[float] | None = None,
) -> npt.NDArray[np.float64]:
    """Extract a feature vector from a contract and optional context.

    Args:
        contract: The normalized market contract.
        cross_platform_price: YES price from the matched contract on the other platform.
        cross_platform_volume: 24h volume from the matched contract.
        price_history: Historical YES prices (most recent last) for lag features.
        volume_history: Historical 24h volumes (most recent last) for lag features.

    Returns:
        1D numpy array matching SPEC.names ordering.
    """
    raise NotImplementedError  # Phase 3

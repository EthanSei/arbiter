"""Feature extraction for the probability estimation model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from arbiter.ingestion.base import Contract

FEATURE_VERSION = "0.1.0"
"""Bump this when the feature schema changes. Stored in MarketSnapshot.feature_version
so training can filter by compatible versions."""

NAN = float("nan")


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
    *,
    now: datetime | None = None,
) -> npt.NDArray[np.float64]:
    """Extract a feature vector from a contract and optional context.

    Args:
        contract: The normalized market contract.
        cross_platform_price: YES price from the matched contract on the other platform.
        cross_platform_volume: 24h volume from the matched contract.
        price_history: Historical YES prices (most recent last) for lag features.
        volume_history: Historical 24h volumes (most recent last) for lag features.
        now: Override current time (for testing). Defaults to UTC now.

    Returns:
        1D numpy array matching SPEC.names ordering.
    """
    if now is None:
        now = datetime.now(UTC)

    features = np.full(len(SPEC.names), NAN, dtype=np.float64)

    # --- Market features ---
    features[0] = contract.yes_price
    features[1] = contract.no_price
    features[2] = contract.yes_ask - contract.yes_bid
    features[3] = contract.last_price if contract.last_price is not None else contract.yes_price
    features[4] = np.log1p(contract.volume_24h)
    features[5] = np.log1p(contract.open_interest)

    if contract.expires_at is not None:
        delta = (contract.expires_at - now).total_seconds() / 3600.0
        features[6] = max(0.0, delta)
    # else: stays NaN

    features[7] = contract.yes_price + contract.no_price - 1.0
    features[8] = float(now.weekday())
    features[9] = float(now.hour)

    # --- Cross-platform features ---
    if cross_platform_price is not None:
        features[10] = contract.yes_price - cross_platform_price

    if cross_platform_volume is not None and cross_platform_volume > 0:
        features[11] = contract.volume_24h / cross_platform_volume

    # --- Lag features ---
    if price_history is not None and len(price_history) >= 2:
        features[12] = price_history[-1] - price_history[-2]  # delta_1h
        features[13] = price_history[-1] - price_history[0]  # delta_24h
        features[15] = float(np.std(price_history))  # volatility_24h
    elif price_history is not None and len(price_history) == 1:
        # Single point: deltas and volatility undefined
        pass

    if volume_history is not None and len(volume_history) >= 1:
        mean_vol = float(np.mean(volume_history))
        if mean_vol > 0:
            features[14] = contract.volume_24h / mean_vol

    return features

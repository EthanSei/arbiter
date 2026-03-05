"""FRED surprise history provider."""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
from typing import Any

from arbiter.data.indicators import INDICATORS
from arbiter.data.providers.base import FeatureSet

logger = logging.getLogger(__name__)


class FREDSurpriseProvider:
    """Loads FRED-derived surprise history and computes μ (naive consensus) and σ.

    Cache files at ``{data_dir}/{indicator_id}.json`` contain:
    - observations: list of {date, actual, consensus, surprise}
    - current_consensus: most recent naive consensus value

    The provider computes σ from the surprise values using an optional
    exponential recency weighting controlled by ``halflife``.
    """

    name = "fred"

    def __init__(
        self,
        data_dir: str = "data/features/fred",
        halflife: int | None = None,
        winsorize: bool = True,
    ) -> None:
        self._data_dir = data_dir
        self._halflife = halflife
        self._winsorize = winsorize

    def load(self, indicator_id: str) -> FeatureSet | None:
        path = os.path.join(self._data_dir, f"{indicator_id}.json")
        if not os.path.isfile(path):
            return None

        with open(path) as f:
            data: dict[str, Any] = json.load(f)

        try:
            observations: list[dict[str, Any]] = data.get("observations", [])
            current_consensus: float = data["current_consensus"]
            surprises = [obs["surprise"] for obs in observations]
        except (KeyError, TypeError, ValueError):
            logger.warning("FREDSurpriseProvider: malformed cache for %s", indicator_id)
            return None

        if len(surprises) < 2:
            return None

        config = INDICATORS.get(indicator_id)
        halflife = config.recency_halflife if config is not None else self._halflife
        sigma = compute_sigma(surprises, halflife, winsorize=self._winsorize)

        return FeatureSet(
            provider="fred",
            indicator_id=indicator_id,
            anchor_mu=current_consensus,
            anchor_sigma=sigma,
        )


def compute_sigma(values: list[float], halflife: int | None, *, winsorize: bool = True) -> float:
    """Compute (optionally exponentially-weighted) standard deviation.

    When halflife is None, uses simple population std.
    When halflife is set, applies exponential decay weights where the most
    recent observation has weight 1.0 and observations ``halflife`` steps
    back have weight 0.5.

    When winsorize is True (default), clips outliers beyond ±3σ from the mean
    before computing the weighted variance. This removes COVID-scale spikes that
    inflate σ and push anchor probabilities toward 0.5.
    """
    n = len(values)
    if n < 2:
        return 0.0

    if winsorize:
        # MAD-based clipping: robust against outlier-inflated std.
        # 1.4826 converts MAD to std-equivalent under a normal distribution.
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values)
        if mad > 0:
            cap = 3.0 * 1.4826 * mad
            values = [max(median - cap, min(median + cap, v)) for v in values]

    if halflife is None:
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return math.sqrt(variance)

    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")

    # Exponential weights: most recent = index n-1
    decay = math.log(2) / halflife
    weights = [math.exp(-decay * (n - 1 - i)) for i in range(n)]
    total_w = sum(weights)

    w_mean = sum(w * v for w, v in zip(weights, values, strict=True)) / total_w
    w_var = sum(w * (v - w_mean) ** 2 for w, v in zip(weights, values, strict=True)) / total_w
    return math.sqrt(w_var)

"""BLS CPI sub-component provider."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from arbiter.data.providers.base import FeatureSet

logger = logging.getLogger(__name__)


class BLSComponentProvider:
    """Loads BLS CPI sub-component trends as supplementary features.

    Cache files at ``{data_dir}/{indicator_id}.json`` contain component MoM
    trends (shelter, food, energy, core). These feed into ``FeatureSet.features``
    for the future ADJUST stage — they do NOT contribute to anchor_mu/sigma.
    """

    name = "bls"

    def __init__(self, data_dir: str = "data/features/bls") -> None:
        self._data_dir = data_dir

    def load(self, indicator_id: str) -> FeatureSet | None:
        path = os.path.join(self._data_dir, f"{indicator_id}.json")
        if not os.path.isfile(path):
            return None

        with open(path) as f:
            data: dict[str, Any] = json.load(f)

        try:
            components: dict[str, float] = {
                k: float(v) for k, v in data.get("components", {}).items()
            }
        except (ValueError, TypeError):
            logger.warning("BLSComponentProvider: malformed cache for %s", indicator_id)
            return None

        return FeatureSet(
            provider="bls",
            indicator_id=indicator_id,
            features=components,
        )

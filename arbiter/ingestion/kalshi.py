"""Kalshi API client for fetching prediction market data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from arbiter.ingestion.base import Contract, MarketClient


class KalshiClient(MarketClient):
    """Fetches and normalizes market data from the Kalshi API.

    Uses cursor-based pagination. Prices are normalized from dollar strings
    (or legacy cent integers) to floats in [0, 1].
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        contracts: list[Contract] = []
        cursor = ""
        while True:
            params: dict[str, str | int] = {"limit": limit, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            resp = await self._http.get(f"{self._base_url}/markets", params=params)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

            for m in data.get("markets", []):
                if m.get("status") != "open":
                    continue
                contract = self._parse_market(m)
                if contract is not None:
                    contracts.append(contract)

            cursor = data.get("cursor", "")
            if not cursor:
                break
        return contracts

    async def close(self) -> None:
        pass  # Client is borrowed, not owned

    @staticmethod
    def _parse_market(m: dict[str, Any]) -> Contract | None:
        yes_bid = _to_float(m.get("yes_bid"))
        yes_ask = _to_float(m.get("yes_ask"))
        if yes_bid is None or yes_ask is None:
            return None
        yes_price = (yes_bid + yes_ask) / 2.0

        last_raw = m.get("last_price")
        last_price = _to_float(last_raw) if last_raw is not None else None

        expires_at = None
        close_time = m.get("close_time")
        if close_time:
            expires_at = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))

        return Contract(
            source="kalshi",
            contract_id=str(m["ticker"]),
            title=str(m.get("title", "")),
            category=str(m.get("category", "")),
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            last_price=last_price,
            volume_24h=_to_float(m.get("volume_24h")) or 0.0,
            open_interest=_to_float(m.get("open_interest")) or 0.0,
            expires_at=expires_at,
            url=str(m.get("url", "")),
            status=str(m.get("status", "open")),
        )


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

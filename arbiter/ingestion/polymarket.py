"""Polymarket API client for fetching prediction market data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from arbiter.ingestion.base import Contract, MarketClient

_SPREAD_ESTIMATE = 0.02  # Approximate half-spread for bid/ask estimation


class PolymarketClient(MarketClient):
    """Fetches and normalizes market data from the Polymarket Gamma API.

    Uses offset-based pagination. Rate limited to 60 requests/minute.
    Prices are normalized from decimal strings to floats in [0, 1].
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        gamma_base_url: str = "https://gamma-api.polymarket.com",
    ) -> None:
        self._http = http
        self._gamma_base = gamma_base_url.rstrip("/")

    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        contracts: list[Contract] = []
        offset = 0
        while True:
            params: dict[str, str | int] = {
                "limit": limit,
                "offset": offset,
                "active": "true",
            }
            resp = await self._http.get(f"{self._gamma_base}/markets", params=params)
            resp.raise_for_status()
            markets: list[dict[str, Any]] = resp.json()

            if not markets:
                break

            for m in markets:
                if m.get("closed") or not m.get("accepting_orders", True):
                    continue
                contract = self._parse_market(m)
                if contract is not None:
                    contracts.append(contract)

            if len(markets) < limit:
                break
            offset += len(markets)
        return contracts

    async def close(self) -> None:
        pass  # Client is borrowed, not owned

    @staticmethod
    def _parse_market(m: dict[str, Any]) -> Contract | None:
        tokens: list[dict[str, Any]] = m.get("tokens", [])
        yes_price = _extract_token_price(tokens, "Yes")
        no_price = _extract_token_price(tokens, "No")
        if yes_price is None:
            return None
        if no_price is None:
            no_price = 1.0 - yes_price

        expires_at = None
        end_date = m.get("end_date")
        if end_date:
            expires_at = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))

        slug = str(m.get("market_slug", ""))
        url = f"https://polymarket.com/event/{slug}" if slug else ""

        return Contract(
            source="polymarket",
            contract_id=str(m.get("condition_id", "")),
            title=str(m.get("question", "")),
            category=str(m.get("category", "")),
            yes_price=yes_price,
            no_price=no_price,
            yes_bid=max(0.0, yes_price - _SPREAD_ESTIMATE),
            yes_ask=min(1.0, yes_price + _SPREAD_ESTIMATE),
            last_price=None,  # Gamma API doesn't provide last trade price
            volume_24h=_safe_float(m.get("volume_24hr")),
            open_interest=_safe_float(m.get("liquidity")),
            expires_at=expires_at,
            url=url,
            status="open",
        )


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _extract_token_price(tokens: list[dict[str, Any]], outcome: str) -> float | None:
    for t in tokens:
        if str(t.get("outcome", "")).lower() == outcome.lower():
            try:
                return float(t["price"])
            except (KeyError, ValueError, TypeError):
                return None
    return None

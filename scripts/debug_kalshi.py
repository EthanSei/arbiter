"""Diagnostic: paginate Kalshi markets and show volume distribution per page.

Run with:  python scripts/debug_kalshi.py [--pages N] [--limit N] [--min-vol N]
"""

import argparse
import asyncio
import collections

import httpx


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
VOLUME_FIELD = "volume_24h_fp"


def _to_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


async def main(max_pages: int, limit: int, min_vol: float) -> None:
    async with httpx.AsyncClient(timeout=30.0) as http:
        cursor = ""
        for page_num in range(1, max_pages + 1):
            params = {"limit": limit, "status": "open", "mve_filter": "exclude"}
            if cursor:
                params["cursor"] = cursor

            resp = await http.get(f"{BASE_URL}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            cursor = data.get("cursor", "")

            if not markets:
                print(f"\nPage {page_num}: empty — stopping.")
                break

            # Per-page stats
            vols = [_to_float(m.get(VOLUME_FIELD)) or 0.0 for m in markets]
            qualifying = [v for v in vols if v >= min_vol]
            prefix_counts: dict[str, int] = collections.Counter(
                m.get("ticker", "")[:20] for m in markets
            )
            top_prefixes = ", ".join(
                f"{p!r}×{c}" for p, c in prefix_counts.most_common(3)
            )
            print(
                f"Page {page_num:2d}: {len(markets):4d} markets | "
                f"vol≥{min_vol}: {len(qualifying):4d} | "
                f"max_vol={max(vols):>10.2f} | "
                f"top prefixes: {top_prefixes}"
            )

            if not cursor:
                print("  (no more pages)")
                break

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=15, help="Max pages to fetch")
    parser.add_argument("--limit", type=int, default=1000, help="Markets per page")
    parser.add_argument("--min-vol", type=float, default=5.0, help="Volume threshold")
    args = parser.parse_args()
    asyncio.run(main(args.pages, args.limit, args.min_vol))

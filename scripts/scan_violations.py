"""Scan Kalshi range markets for consistency violations and report statistics.

Fetches all open Kalshi markets, groups MAXMON/MINMON range series, and checks
for stochastic dominance violations.  Use --watch to poll repeatedly and build
a picture of how often violations appear.

Run with:
    python scripts/scan_violations.py                   # single snapshot
    python scripts/scan_violations.py --watch --interval 300   # poll every 5 min
    python scripts/scan_violations.py --watch --interval 60 --duration 3600  # 1 hour
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import UTC, datetime

import httpx

from arbiter.ingestion.kalshi import KalshiClient
from arbiter.ingestion.rate_limiter import RateLimitedClient
from arbiter.scoring.consistency import find_consistency_violations

_FEE_RATE = 0.01


async def fetch_and_scan(client: KalshiClient) -> dict:
    """Fetch markets and return violation summary."""
    t0 = time.monotonic()
    contracts = await client.fetch_markets()
    t_fetch = time.monotonic() - t0

    violations = find_consistency_violations(contracts, fee_rate=_FEE_RATE)

    # Count range-market groups for context
    import re
    from collections import defaultdict

    suffix_re = re.compile(r"-(\d+)$")
    above_re = re.compile(r"MAXMON", re.IGNORECASE)
    below_re = re.compile(r"MINMON", re.IGNORECASE)

    groups: dict[str, list] = defaultdict(list)
    for c in contracts:
        if c.source != "kalshi":
            continue
        m = suffix_re.search(c.contract_id)
        if m is None:
            continue
        base = c.contract_id[: m.start()]
        if above_re.search(base) or below_re.search(base):
            groups[base].append(c)

    # Group details
    group_details = []
    for base, members in sorted(groups.items()):
        traded = [c for c in members if c.volume_24h > 0]
        group_details.append(
            {
                "base": base,
                "size": len(members),
                "traded": len(traded),
            }
        )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_contracts": len(contracts),
        "range_groups": len(groups),
        "range_contracts": sum(len(g) for g in groups.values()),
        "violations": violations,
        "group_details": group_details,
        "fetch_seconds": t_fetch,
    }


def print_snapshot(result: dict, snapshot_num: int = 0) -> None:
    """Pretty-print a single scan result."""
    ts = result["timestamp"][:19]
    header = f"Snapshot #{snapshot_num}" if snapshot_num else "Snapshot"
    print(f"\n{'=' * 70}")
    print(f"  {header}  |  {ts} UTC")
    print(f"{'=' * 70}")
    print(f"  Total contracts:  {result['total_contracts']}")
    print(f"  Range groups:     {result['range_groups']}")
    print(f"  Range contracts:  {result['range_contracts']}")
    print(f"  Fetch time:       {result['fetch_seconds']:.1f}s")

    # Group summary
    if result["group_details"]:
        print(f"\n  {'Group':<45} {'Size':>5} {'Traded':>7}")
        print(f"  {'-' * 45} {'-' * 5} {'-' * 7}")
        for g in result["group_details"]:
            print(f"  {g['base']:<45} {g['size']:>5} {g['traded']:>7}")

    violations = result["violations"]
    if violations:
        print(f"\n  ** {len(violations)} VIOLATION(S) FOUND **\n")
        for v in violations:
            print(f"  Contract: {v.contract.contract_id}")
            print(f"    Title:       {v.contract.title}")
            print(f"    Market:      {v.market_price:.1%}")
            print(f"    Model Prob:  {v.model_probability:.1%}")
            print(f"    EV:          {v.expected_value:+.1%}")
            print(f"    Kelly:       {v.kelly_size:.1%}")
            print(f"    Volume 24h:  {v.contract.volume_24h:.0f}")
            print()
    else:
        print("\n  No violations (all range groups are monotonically consistent)")

    print()


async def run(args: argparse.Namespace) -> None:
    http = httpx.AsyncClient(timeout=30.0)
    rl = RateLimitedClient(http, rpm=20)
    client = KalshiClient(rl, min_volume_24h=0.0)  # include zero-vol to see full groups

    try:
        if not args.watch:
            result = await fetch_and_scan(client)
            print_snapshot(result)
            return

        # Watch mode — poll repeatedly
        start = time.monotonic()
        snapshot_num = 0
        violation_snapshots = 0
        total_violations = 0

        while True:
            snapshot_num += 1
            result = await fetch_and_scan(client)
            print_snapshot(result, snapshot_num)

            n_violations = len(result["violations"])
            total_violations += n_violations
            if n_violations > 0:
                violation_snapshots += 1

            elapsed = time.monotonic() - start
            pct = (violation_snapshots / snapshot_num) * 100

            print(
                f"  Running stats: {snapshot_num} snapshots, "
                f"{violation_snapshots} with violations ({pct:.0f}%), "
                f"{total_violations} total violations, "
                f"{elapsed / 60:.1f} min elapsed"
            )

            if args.duration and elapsed >= args.duration:
                print(f"\n  Duration limit ({args.duration}s) reached. Stopping.")
                break

            await asyncio.sleep(args.interval)

        # Final summary
        print(f"\n{'=' * 70}")
        print("  FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Snapshots:              {snapshot_num}")
        print(f"  With violations:        {violation_snapshots} ({pct:.0f}%)")
        print(f"  Total violations:       {total_violations}")
        print(f"  Duration:               {elapsed / 60:.1f} min")
        print(f"  Poll interval:          {args.interval}s")
        print()

    finally:
        await http.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan Kalshi range markets for consistency violations"
    )
    parser.add_argument(
        "--watch", action="store_true", help="Poll repeatedly instead of single snapshot"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between polls in watch mode (default: 300)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Max duration in seconds for watch mode (0=unlimited)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

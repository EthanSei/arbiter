"""Data collection for model training — historical scraping and live snapshots."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arbiter.db.models import MarketSnapshot, Source
from arbiter.ingestion.base import Contract, MarketClient
from arbiter.models.features import FEATURE_VERSION, SPEC, extract_features

# YES price threshold for determining binary outcome
_YES_WIN_THRESHOLD = 0.99
_NO_WIN_THRESHOLD = 0.01


async def collect_resolved_markets(client: MarketClient) -> list[dict[str, object]]:
    """Fetch historically resolved/settled markets for training data bootstrap.

    Kalshi: status=settled. Polymarket: closed=true.
    Returns a list of dicts with contract fields for downstream use.
    """
    contracts = await client.fetch_markets()
    return [
        {
            "contract_id": c.contract_id,
            "source": c.source,
            "title": c.title,
            "category": c.category,
            "yes_price": c.yes_price,
            "no_price": c.no_price,
            "volume_24h": c.volume_24h,
            "open_interest": c.open_interest,
            "expires_at": c.expires_at,
            "status": c.status,
        }
        for c in contracts
    ]


async def snapshot_live_markets(
    client: MarketClient,
    session: AsyncSession,
    price_change_threshold: float = 0.01,
) -> int:
    """Snapshot current live market data into MarketSnapshot table.

    Only snapshots markets whose price has changed by more than the threshold
    since the last snapshot, to avoid redundant training data.

    Returns the number of snapshots written.
    """
    contracts = await client.fetch_markets()
    written = 0

    for contract in contracts:
        source = Source(contract.source)
        last_price = await _get_last_snapshot_price(session, contract.contract_id)
        if last_price is not None:
            delta = abs(contract.yes_price - last_price)
            if delta <= price_change_threshold:
                continue

        features_vec = extract_features(contract)
        # Convert NaN to None for JSON serialisation
        features_dict: dict[str, object] = {
            name: (None if v != v else float(v))  # v != v is the NaN check
            for name, v in zip(SPEC.names, features_vec, strict=True)
        }

        snapshot = MarketSnapshot(
            source=source,
            contract_id=contract.contract_id,
            title=contract.title,
            category=contract.category,
            features=features_dict,
            feature_version=FEATURE_VERSION,
            snapshot_at=datetime.now(UTC),
        )
        session.add(snapshot)
        written += 1

    if written:
        await session.flush()

    return written


async def backfill_outcomes(session: AsyncSession, client: MarketClient) -> int:
    """Update MarketSnapshot.outcome for markets that have since resolved.

    Fetches current market data from the client, identifies contracts with
    binary outcomes (yes_price ≈ 0 or 1), and backfills unlabeled snapshots.

    Returns the number of snapshots updated.
    """
    contracts = await client.fetch_markets()
    resolved: dict[str, float] = {}
    for c in contracts:
        outcome = _resolve_outcome(c)
        if outcome is not None:
            resolved[c.contract_id] = outcome

    if not resolved:
        return 0

    stmt = select(MarketSnapshot).where(
        MarketSnapshot.outcome.is_(None),
        MarketSnapshot.contract_id.in_(resolved.keys()),
    )
    result = await session.execute(stmt)
    snapshots = result.scalars().all()

    now = datetime.now(UTC)
    for snap in snapshots:
        snap.outcome = resolved[snap.contract_id]
        snap.resolved_at = now

    if snapshots:
        await session.flush()

    return len(snapshots)


async def _get_last_snapshot_price(session: AsyncSession, contract_id: str) -> float | None:
    """Return the yes_price from the most recent snapshot for a contract, or None."""
    stmt = (
        select(MarketSnapshot)
        .where(MarketSnapshot.contract_id == contract_id)
        .order_by(MarketSnapshot.snapshot_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    snap = result.scalar_one_or_none()
    if snap is None or not snap.features:
        return None
    price = snap.features.get("yes_price")
    return float(price) if price is not None else None  # type: ignore[arg-type]


def _resolve_outcome(contract: Contract) -> float | None:
    """Return 1.0 if market resolved YES, 0.0 if NO, None if still active."""
    if contract.yes_price >= _YES_WIN_THRESHOLD:
        return 1.0
    if contract.yes_price <= _NO_WIN_THRESHOLD:
        return 0.0
    return None

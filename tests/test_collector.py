"""Tests for training data collector — snapshots and outcome backfill."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from arbiter.db.models import MarketSnapshot, Source
from arbiter.ingestion.base import Contract
from arbiter.models.features import FEATURE_VERSION
from arbiter.training.collector import (
    backfill_outcomes,
    collect_resolved_markets,
    snapshot_live_markets,
)


def _make_contract(
    contract_id: str = "TEST-01",
    yes_price: float = 0.6,
    source: str = "kalshi",
    status: str = "open",
    expires_at: datetime | None = None,
) -> Contract:
    return Contract(
        source=source,
        contract_id=contract_id,
        title="Will X happen?",
        category="politics",
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        yes_bid=yes_price - 0.01,
        yes_ask=yes_price + 0.01,
        last_price=yes_price,
        volume_24h=10_000.0,
        open_interest=5_000.0,
        expires_at=expires_at or datetime(2025, 1, 1, tzinfo=UTC),
        url="https://example.com",
        status=status,
    )


def _make_client(contracts: list[Contract]) -> AsyncMock:
    client = AsyncMock()
    client.fetch_markets = AsyncMock(return_value=contracts)
    return client


class TestSnapshotLiveMarkets:
    @pytest.mark.asyncio
    async def test_snapshots_written_for_new_contracts(self, db_session):
        """New contracts with no prior snapshot are always snapshotted."""
        contracts = [_make_contract("C1", 0.5), _make_contract("C2", 0.7)]
        client = _make_client(contracts)

        count = await snapshot_live_markets(client, db_session)

        assert count == 2

    @pytest.mark.asyncio
    async def test_skips_unchanged_price(self, db_session):
        """Contract whose price hasn't moved past threshold is not re-snapshotted."""
        contract = _make_contract("C1", 0.5)
        # Pre-seed a snapshot at same price
        existing = MarketSnapshot(
            source=Source.KALSHI,
            contract_id="C1",
            title="Will X happen?",
            features={"yes_price": 0.5},
            feature_version=FEATURE_VERSION,
            snapshot_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        db_session.add(existing)
        await db_session.flush()

        client = _make_client([contract])
        count = await snapshot_live_markets(client, db_session, price_change_threshold=0.01)

        assert count == 0

    @pytest.mark.asyncio
    async def test_snapshots_when_price_exceeds_threshold(self, db_session):
        """Contract whose price moved more than threshold gets a new snapshot."""
        contract = _make_contract("C1", 0.65)
        existing = MarketSnapshot(
            source=Source.KALSHI,
            contract_id="C1",
            title="Will X happen?",
            features={"yes_price": 0.5},
            feature_version=FEATURE_VERSION,
            snapshot_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        db_session.add(existing)
        await db_session.flush()

        client = _make_client([contract])
        count = await snapshot_live_markets(client, db_session, price_change_threshold=0.01)

        assert count == 1

    @pytest.mark.asyncio
    async def test_snapshot_stores_feature_version(self, db_session):
        """Written snapshots use the current FEATURE_VERSION."""
        client = _make_client([_make_contract("C1", 0.5)])
        await snapshot_live_markets(client, db_session)

        from sqlalchemy import select

        stmt = select(MarketSnapshot).where(MarketSnapshot.contract_id == "C1")
        result = await db_session.execute(stmt)
        snap = result.scalar_one()
        assert snap.feature_version == FEATURE_VERSION

    @pytest.mark.asyncio
    async def test_snapshot_stores_features_dict(self, db_session):
        """Written snapshots store a non-empty features JSON dict."""
        client = _make_client([_make_contract("C1", 0.5)])
        await snapshot_live_markets(client, db_session)

        from sqlalchemy import select

        stmt = select(MarketSnapshot).where(MarketSnapshot.contract_id == "C1")
        result = await db_session.execute(stmt)
        snap = result.scalar_one()
        assert snap.features is not None
        assert "yes_price" in snap.features

    @pytest.mark.asyncio
    async def test_returns_count_of_written_snapshots(self, db_session):
        """Return value equals number of snapshots actually written."""
        contracts = [_make_contract("A"), _make_contract("B"), _make_contract("C")]
        client = _make_client(contracts)

        count = await snapshot_live_markets(client, db_session)

        assert count == 3


class TestBackfillOutcomes:
    @pytest.mark.asyncio
    async def test_backfills_resolved_market(self, db_session):
        """Snapshots for a resolved market get outcome=1.0 or 0.0."""
        snap = MarketSnapshot(
            source=Source.KALSHI,
            contract_id="C1",
            title="Test",
            features={},
            feature_version=FEATURE_VERSION,
            outcome=None,
            snapshot_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        db_session.add(snap)
        await db_session.flush()

        # Simulate a resolved contract (status=settled, yes wins)
        resolved_contract = _make_contract("C1", yes_price=1.0, status="settled")
        client = _make_client([resolved_contract])

        count = await backfill_outcomes(db_session, client)

        assert count == 1
        from sqlalchemy import select

        stmt = select(MarketSnapshot).where(MarketSnapshot.contract_id == "C1")
        result = await db_session.execute(stmt)
        updated = result.scalar_one()
        assert updated.outcome == 1.0
        assert updated.resolved_at is not None

    @pytest.mark.asyncio
    async def test_skips_already_labeled_snapshots(self, db_session):
        """Snapshots that already have an outcome are not re-processed."""
        snap = MarketSnapshot(
            source=Source.KALSHI,
            contract_id="C1",
            title="Test",
            features={},
            feature_version=FEATURE_VERSION,
            outcome=1.0,
            snapshot_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        db_session.add(snap)
        await db_session.flush()

        resolved_contract = _make_contract("C1", yes_price=0.0, status="settled")
        client = _make_client([resolved_contract])

        count = await backfill_outcomes(db_session, client)

        assert count == 0

    @pytest.mark.asyncio
    async def test_no_resolved_markets_returns_zero(self, db_session):
        """Returns 0 when no contracts have settled."""
        snap = MarketSnapshot(
            source=Source.KALSHI,
            contract_id="C1",
            title="Test",
            features={},
            feature_version=FEATURE_VERSION,
            outcome=None,
            snapshot_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        db_session.add(snap)
        await db_session.flush()

        # All open, not resolved
        client = _make_client([_make_contract("C1", 0.5, status="open")])
        count = await backfill_outcomes(db_session, client)

        assert count == 0


class TestCollectResolvedMarkets:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        """Returns a list of dicts with contract data."""
        contract = _make_contract("C1", 0.0, status="settled")
        client = _make_client([contract])

        result = await collect_resolved_markets(client)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_includes_contract_fields(self):
        """Each dict includes key contract fields."""
        contract = _make_contract("C1", 0.0, status="settled")
        client = _make_client([contract])

        result = await collect_resolved_markets(client)

        assert result[0]["contract_id"] == "C1"
        assert "yes_price" in result[0]
        assert "source" in result[0]

    @pytest.mark.asyncio
    async def test_empty_when_no_contracts(self):
        """Returns empty list when client returns no contracts."""
        client = _make_client([])
        result = await collect_resolved_markets(client)
        assert result == []

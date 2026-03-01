"""Tests for ScanPipeline — the core scan loop orchestrator."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from arbiter.alerts.base import AlertChannel
from arbiter.db.models import AlertLog, Base, MarketSnapshot, Opportunity
from arbiter.ingestion.base import Contract, MarketClient
from arbiter.models.base import ProbabilityEstimator
from arbiter.scheduler import ScanPipeline
from arbiter.scoring.ev import ScoredOpportunity

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _contract(
    source: str = "kalshi",
    contract_id: str = "TEST-001",
    yes_price: float = 0.40,
    no_price: float = 0.60,
) -> Contract:
    return Contract(
        source=source,
        contract_id=contract_id,
        title="Will X happen?",
        category="test",
        yes_price=yes_price,
        no_price=no_price,
        yes_bid=yes_price - 0.01,
        yes_ask=yes_price + 0.01,
        last_price=yes_price,
        volume_24h=1000.0,
        open_interest=500.0,
        expires_at=None,
        url="https://example.com",
        status="open",
    )


class _OkClient(MarketClient):
    def __init__(self, contracts: list[Contract]) -> None:
        self._contracts = contracts
        self.called = 0

    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        self.called += 1
        return self._contracts

    async def close(self) -> None:
        pass


class _BrokenClient(MarketClient):
    async def fetch_markets(self, *, limit: int = 100) -> list[Contract]:
        raise ConnectionError("simulated network failure")

    async def close(self) -> None:
        pass


class _FixedEstimator(ProbabilityEstimator):
    def __init__(self, prob: float = 0.70) -> None:
        self._prob = prob

    async def estimate(self, contract: Contract) -> float:
        return self._prob


class _RecordingChannel(AlertChannel):
    def __init__(self) -> None:
        self.sent: list[ScoredOpportunity] = []

    async def send(self, opp: ScoredOpportunity) -> None:
        self.sent.append(opp)

    async def close(self) -> None:
        pass


class _BrokenChannel(AlertChannel):
    async def send(self, opp: ScoredOpportunity) -> None:
        raise RuntimeError("channel broke")

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_factory():
    """In-memory SQLite session factory for tests.

    Uses StaticPool so every async session shares the same connection, which
    is required for SQLite :memory: databases to persist data across sessions.
    """
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _pipeline(
    clients: list[MarketClient],
    estimator: ProbabilityEstimator,
    channels: list[AlertChannel],
    db_factory: async_sessionmaker[AsyncSession],
    ev_threshold: float = 0.05,
    fee_rate: float = 0.01,
) -> ScanPipeline:
    return ScanPipeline(
        clients=clients,
        estimator=estimator,
        channels=channels,
        session_factory=db_factory,
        ev_threshold=ev_threshold,
        fee_rate=fee_rate,
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


async def test_init_stores_dependencies(db_factory) -> None:
    clients = [_OkClient([])]
    estimator = _FixedEstimator()
    channels = [_RecordingChannel()]
    p = _pipeline(clients, estimator, channels, db_factory)
    assert p._clients is clients
    assert p._estimator is estimator
    assert p._channels is channels
    assert p._session_factory is db_factory


# ---------------------------------------------------------------------------
# run_cycle — basic behaviour
# ---------------------------------------------------------------------------


async def test_run_cycle_returns_zero_with_no_contracts(db_factory) -> None:
    p = _pipeline([_OkClient([])], _FixedEstimator(), [], db_factory)
    assert await p.run_cycle() == 0


async def test_run_cycle_fetches_all_clients(db_factory) -> None:
    c1 = _OkClient([])
    c2 = _OkClient([])
    p = _pipeline([c1, c2], _FixedEstimator(), [], db_factory)
    await p.run_cycle()
    assert c1.called == 1
    assert c2.called == 1


async def test_run_cycle_continues_after_client_error(db_factory) -> None:
    """A broken client must not abort the cycle; the healthy client's contracts are scored."""
    channel = _RecordingChannel()
    # Contract with large EV
    contract = _contract(yes_price=0.30, no_price=0.70)
    ok = _OkClient([contract])
    broken = _BrokenClient()
    # model_prob=0.70: EV_yes = 0.70 - (0.30+0.01) = 0.39 > threshold
    p = _pipeline([broken, ok], _FixedEstimator(0.70), [channel], db_factory)
    alerts = await p.run_cycle()
    assert alerts >= 1  # At least the yes side alerted


# ---------------------------------------------------------------------------
# run_cycle — EV threshold filtering
# ---------------------------------------------------------------------------


async def test_run_cycle_does_not_alert_below_threshold(db_factory) -> None:
    """Contract with EV just below threshold produces no alerts."""
    channel = _RecordingChannel()
    # yes_price=0.45, model_prob=0.46 → EV_yes = 0.46 - (0.45+0.01) = 0.00 — below 0.05
    p = _pipeline(
        [_OkClient([_contract(yes_price=0.45, no_price=0.55)])],
        _FixedEstimator(0.46),
        [channel],
        db_factory,
        ev_threshold=0.05,
        fee_rate=0.01,
    )
    assert await p.run_cycle() == 0
    assert channel.sent == []


async def test_run_cycle_alerts_above_threshold(db_factory) -> None:
    """Contract with EV above threshold produces exactly one alert (YES side)."""
    channel = _RecordingChannel()
    # yes_price=0.30, model_prob=0.70 → EV_yes = 0.70 - 0.31 = 0.39 > 0.05
    p = _pipeline(
        [_OkClient([_contract(yes_price=0.30, no_price=0.70)])],
        _FixedEstimator(0.70),
        [channel],
        db_factory,
    )
    alerts = await p.run_cycle()
    assert alerts == 1
    assert len(channel.sent) == 1
    assert channel.sent[0].direction == "yes"


# ---------------------------------------------------------------------------
# run_cycle — DB persistence
# ---------------------------------------------------------------------------


async def test_run_cycle_persists_opportunity(db_factory) -> None:
    """A new above-threshold contract is written as an Opportunity row."""
    p = _pipeline(
        [_OkClient([_contract(contract_id="ABC-1", yes_price=0.30)])],
        _FixedEstimator(0.70),
        [],
        db_factory,
    )
    await p.run_cycle()

    async with db_factory() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(Opportunity))).scalars().all()
    assert len(rows) == 1
    assert rows[0].contract_id == "ABC-1"
    assert rows[0].active is True


async def test_run_cycle_creates_alert_log(db_factory) -> None:
    """AlertLog row is created for each channel alert attempt."""
    channel = _RecordingChannel()
    p = _pipeline(
        [_OkClient([_contract(yes_price=0.30)])],
        _FixedEstimator(0.70),
        [channel],
        db_factory,
    )
    await p.run_cycle()

    async with db_factory() as session:
        from sqlalchemy import select

        logs = (await session.execute(select(AlertLog))).scalars().all()
    assert len(logs) == 1
    assert logs[0].success is True


async def test_run_cycle_logs_failed_alert(db_factory) -> None:
    """A channel that raises still produces an AlertLog with success=False."""
    p = _pipeline(
        [_OkClient([_contract(yes_price=0.30)])],
        _FixedEstimator(0.70),
        [_BrokenChannel()],
        db_factory,
    )
    await p.run_cycle()

    async with db_factory() as session:
        from sqlalchemy import select

        logs = (await session.execute(select(AlertLog))).scalars().all()
    assert len(logs) == 1
    assert logs[0].success is False
    assert logs[0].error_message is not None


async def test_run_cycle_creates_market_snapshot(db_factory) -> None:
    """A MarketSnapshot is created for each contract processed."""
    contract = _contract(contract_id="SNAP-001")
    p = _pipeline([_OkClient([contract])], _FixedEstimator(0.50), [], db_factory)
    await p.run_cycle()

    async with db_factory() as session:
        from sqlalchemy import select

        snaps = (await session.execute(select(MarketSnapshot))).scalars().all()
    assert len(snaps) == 1
    assert snaps[0].contract_id == "SNAP-001"
    assert snaps[0].feature_version == "0.1.0"
    assert snaps[0].features is not None


# ---------------------------------------------------------------------------
# run_cycle — state-based deduplication
# ---------------------------------------------------------------------------


async def test_run_cycle_does_not_realert_active_opportunity(db_factory) -> None:
    """Running two cycles for the same contract alerts only once."""
    channel = _RecordingChannel()
    contract = _contract(yes_price=0.30)
    p = _pipeline([_OkClient([contract])], _FixedEstimator(0.70), [channel], db_factory)
    await p.run_cycle()
    await p.run_cycle()
    assert len(channel.sent) == 1  # second cycle: already active, no re-alert


async def test_run_cycle_reactivates_inactive_opportunity(db_factory) -> None:
    """An opportunity that went inactive (EV dropped) re-alerts when EV rises again.

    Use no_price=0.70 so that the NO side is never profitable at any estimator
    value used in this test, isolating the YES-side reactivation logic.
      Cycle 1 (prob=0.70): EV_yes=+0.39, EV_no=-0.41 → YES alerted (1 total)
      Cycle 2 (prob=0.32): EV_yes=+0.01, EV_no=-0.03 → both below → YES deactivated
      Cycle 3 (prob=0.70): EV_yes=+0.39 → YES reactivated → alert (2 total)
    """
    channel = _RecordingChannel()
    # Explicitly set no_price=0.70 so the NO side stays below threshold when
    # model_prob_yes drops to 0.32 (EV_no = 0.68 - 0.71 = -0.03 < 0.05)
    contract = _contract(yes_price=0.30, no_price=0.70)

    p = _pipeline([_OkClient([contract])], _FixedEstimator(0.70), [channel], db_factory)
    await p.run_cycle()  # Cycle 1: alert YES

    p._estimator = _FixedEstimator(0.32)
    await p.run_cycle()  # Cycle 2: both below threshold → YES deactivated

    p._estimator = _FixedEstimator(0.70)
    await p.run_cycle()  # Cycle 3: YES reactivated → re-alert
    assert len(channel.sent) == 2  # first discovery + reactivation


async def test_run_cycle_deactivates_below_threshold_opportunity(db_factory) -> None:
    """An Opportunity drops below threshold and gets marked inactive."""
    # Use no_price=0.70 so NO side never crosses threshold at model_prob=0.32,
    # keeping the deactivation test focused on the YES side only.
    contract = _contract(yes_price=0.30, no_price=0.70)

    # Cycle 1: above threshold
    estimator = _FixedEstimator(0.70)
    p = _pipeline([_OkClient([contract])], estimator, [], db_factory)
    await p.run_cycle()

    # Cycle 2: below threshold
    p._estimator = _FixedEstimator(0.32)
    await p.run_cycle()

    async with db_factory() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(Opportunity))).scalars().all()
    yes_opp = next(r for r in rows if r.direction == "yes")
    assert yes_opp.active is False


# ---------------------------------------------------------------------------
# run_cycle — multiple channels
# ---------------------------------------------------------------------------


async def test_run_cycle_sends_to_all_channels(db_factory) -> None:
    """Alert is dispatched to every channel independently."""
    ch1, ch2 = _RecordingChannel(), _RecordingChannel()
    p = _pipeline(
        [_OkClient([_contract(yes_price=0.30)])],
        _FixedEstimator(0.70),
        [ch1, ch2],
        db_factory,
    )
    await p.run_cycle()
    assert len(ch1.sent) == 1
    assert len(ch2.sent) == 1


async def test_broken_channel_does_not_abort_other_channels(db_factory) -> None:
    """If one channel raises, other channels still receive the alert."""
    good = _RecordingChannel()
    p = _pipeline(
        [_OkClient([_contract(yes_price=0.30)])],
        _FixedEstimator(0.70),
        [_BrokenChannel(), good],
        db_factory,
    )
    await p.run_cycle()
    assert len(good.sent) == 1


# ---------------------------------------------------------------------------
# run_forever
# ---------------------------------------------------------------------------


async def test_run_forever_cancels_cleanly(db_factory) -> None:
    """run_forever stops cleanly when the task is cancelled."""
    p = _pipeline([_OkClient([])], _FixedEstimator(), [], db_factory)
    task = asyncio.create_task(p.run_forever(poll_interval=9999))
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

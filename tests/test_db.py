"""Tests for database models — CRUD operations on all three tables."""

from datetime import UTC, datetime

from sqlalchemy import select

from arbiter.db.models import AlertLog, Direction, MarketSnapshot, Opportunity, Source


async def test_opportunity_uuid_auto_generated(db_session):
    opp = Opportunity(
        source=Source.KALSHI,
        contract_id="UUID-TEST",
        title="UUID test",
        direction=Direction.YES,
        market_price=0.50,
        model_probability=0.60,
        expected_value=0.10,
    )
    db_session.add(opp)
    await db_session.commit()

    result = await db_session.execute(
        select(Opportunity).where(Opportunity.contract_id == "UUID-TEST")
    )
    row = result.scalar_one()
    assert row.id is not None
    assert len(row.id) == 36  # UUID format


async def test_alert_log_uuid_auto_generated(db_session):
    log = AlertLog(opportunity_id="fake-id", channel="discord", success=True)
    db_session.add(log)
    await db_session.commit()

    result = await db_session.execute(select(AlertLog).where(AlertLog.channel == "discord"))
    row = result.scalar_one()
    assert row.id is not None
    assert len(row.id) == 36


async def test_opportunity_timestamps_auto_set(db_session):
    opp = Opportunity(
        source=Source.KALSHI,
        contract_id="TS-TEST",
        title="Timestamp test",
        direction=Direction.YES,
        market_price=0.50,
        model_probability=0.60,
        expected_value=0.10,
    )
    db_session.add(opp)
    await db_session.commit()

    result = await db_session.execute(
        select(Opportunity).where(Opportunity.contract_id == "TS-TEST")
    )
    row = result.scalar_one()
    assert row.discovered_at is not None
    assert row.last_seen_at is not None


async def test_opportunity_with_expires_at(db_session):
    expiry = datetime(2026, 12, 31, tzinfo=UTC)
    opp = Opportunity(
        source=Source.KALSHI,
        contract_id="EXP-TEST",
        title="Expiry test",
        direction=Direction.YES,
        market_price=0.50,
        model_probability=0.60,
        expected_value=0.10,
        expires_at=expiry,
    )
    db_session.add(opp)
    await db_session.commit()

    result = await db_session.execute(
        select(Opportunity).where(Opportunity.contract_id == "EXP-TEST")
    )
    row = result.scalar_one()
    assert row.expires_at is not None


async def test_market_snapshot_without_features(db_session):
    snapshot = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="NO-FEAT",
        title="No features test",
        category="test",
        feature_version="0.1.0",
    )
    db_session.add(snapshot)
    await db_session.commit()

    result = await db_session.execute(
        select(MarketSnapshot).where(MarketSnapshot.contract_id == "NO-FEAT")
    )
    row = result.scalar_one()
    assert row.features is None


async def test_create_and_read_opportunity(db_session):
    opp = Opportunity(
        source=Source.KALSHI,
        contract_id="TICKER-123",
        title="Will it rain tomorrow?",
        direction=Direction.YES,
        market_price=0.45,
        model_probability=0.62,
        expected_value=0.17,
        kelly_size=0.12,
        active=True,
    )
    db_session.add(opp)
    await db_session.commit()

    result = await db_session.execute(
        select(Opportunity).where(Opportunity.contract_id == "TICKER-123")
    )
    row = result.scalar_one()
    assert row.source == Source.KALSHI
    assert row.direction == Direction.YES
    assert row.market_price == 0.45
    assert row.model_probability == 0.62
    assert row.expected_value == 0.17
    assert row.active is True
    assert row.last_alerted_at is None


async def test_opportunity_dedup_fields(db_session):
    """Verify state-based dedup fields work correctly."""
    now = datetime.now(UTC)
    opp = Opportunity(
        source=Source.POLYMARKET,
        contract_id="COND-456",
        title="Will BTC hit 100k?",
        direction=Direction.NO,
        market_price=0.30,
        model_probability=0.25,
        expected_value=-0.05,
        kelly_size=0.0,
        active=True,
        last_alerted_at=now,
        last_seen_at=now,
    )
    db_session.add(opp)
    await db_session.commit()

    result = await db_session.execute(
        select(Opportunity).where(Opportunity.contract_id == "COND-456")
    )
    row = result.scalar_one()
    assert row.last_alerted_at is not None
    assert row.active is True


async def test_create_alert_log(db_session):
    log = AlertLog(
        opportunity_id="fake-opp-id",
        channel="discord",
        success=True,
    )
    db_session.add(log)
    await db_session.commit()

    result = await db_session.execute(
        select(AlertLog).where(AlertLog.opportunity_id == "fake-opp-id")
    )
    row = result.scalar_one()
    assert row.channel == "discord"
    assert row.success is True
    assert row.error_message is None


async def test_alert_log_with_error(db_session):
    log = AlertLog(
        opportunity_id="fake-opp-id",
        channel="sms",
        success=False,
        error_message="Twilio returned 401",
    )
    db_session.add(log)
    await db_session.commit()

    result = await db_session.execute(select(AlertLog).where(AlertLog.channel == "sms"))
    row = result.scalar_one()
    assert row.success is False
    assert row.error_message == "Twilio returned 401"


async def test_create_market_snapshot(db_session):
    snapshot = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="TICKER-789",
        title="Fed rate cut March?",
        category="economics",
        features={"yes_price": 0.65, "volume": 50000},
        feature_version="0.1.0",
    )
    db_session.add(snapshot)
    await db_session.commit()

    result = await db_session.execute(
        select(MarketSnapshot).where(MarketSnapshot.contract_id == "TICKER-789")
    )
    row = result.scalar_one()
    assert row.category == "economics"
    assert row.features["yes_price"] == 0.65
    assert row.feature_version == "0.1.0"
    assert row.outcome is None  # not yet resolved


async def test_snapshot_outcome_backfill(db_session):
    """Simulate backfilling outcome after market resolves."""
    snapshot = MarketSnapshot(
        source=Source.POLYMARKET,
        contract_id="COND-999",
        title="Will ETH flip BTC?",
        category="crypto",
        feature_version="0.1.0",
    )
    db_session.add(snapshot)
    await db_session.commit()

    # Backfill outcome
    result = await db_session.execute(
        select(MarketSnapshot).where(MarketSnapshot.contract_id == "COND-999")
    )
    row = result.scalar_one()
    row.outcome = 0.0  # resolved NO
    row.resolved_at = datetime.now(UTC)
    await db_session.commit()

    result = await db_session.execute(
        select(MarketSnapshot).where(MarketSnapshot.contract_id == "COND-999")
    )
    row = result.scalar_one()
    assert row.outcome == 0.0
    assert row.resolved_at is not None

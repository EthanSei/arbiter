"""Tests for data export utilities — TDD: written before implementation."""

import csv
import io
from datetime import UTC, datetime

from arbiter.db.models import (
    CandlestickBar,
    Direction,
    MarketSnapshot,
    Opportunity,
    OrderBookSnapshot,
    PaperTrade,
    Source,
)
from arbiter.export.dataframes import (
    export_candlestick_bars,
    export_opportunities,
    export_order_book_snapshots,
    export_paper_trades,
    export_snapshots,
    to_csv,
)

# ---------------------------------------------------------------------------
# export_snapshots
# ---------------------------------------------------------------------------


async def test_export_snapshots_returns_dicts(db_session):
    """Insert 2 MarketSnapshot rows, verify export returns 2 dicts with expected keys."""
    s1 = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="SNAP-001",
        title="Will X happen?",
        category="crypto",
        features={"yes_price": 0.40, "no_price": 0.60},
        feature_version="0.1.0",
    )
    s2 = MarketSnapshot(
        source=Source.POLYMARKET,
        contract_id="SNAP-002",
        title="Will Y happen?",
        category="politics",
        features={"yes_price": 0.70, "no_price": 0.30},
        feature_version="0.1.0",
    )
    db_session.add_all([s1, s2])
    await db_session.commit()

    rows = await export_snapshots(db_session)
    assert len(rows) == 2

    expected_keys = {
        "id",
        "source",
        "contract_id",
        "title",
        "category",
        "series_ticker",
        "feature_version",
        "outcome",
        "snapshot_at",
        "resolved_at",
    }
    for row in rows:
        assert expected_keys.issubset(row.keys()), f"Missing keys: {expected_keys - row.keys()}"


async def test_export_snapshots_date_filter(db_session):
    """Insert snapshots at different dates, filter by start_date/end_date."""
    old = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="OLD-001",
        title="Old snapshot",
        category="test",
        feature_version="0.1.0",
        snapshot_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    new = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="NEW-001",
        title="New snapshot",
        category="test",
        feature_version="0.1.0",
        snapshot_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    db_session.add_all([old, new])
    await db_session.commit()

    # Only the new snapshot should match
    rows = await export_snapshots(
        db_session,
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "NEW-001"

    # Only the old snapshot should match
    rows = await export_snapshots(
        db_session,
        end_date=datetime(2025, 12, 31, tzinfo=UTC),
    )
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "OLD-001"

    # Both date bounds together
    rows = await export_snapshots(
        db_session,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert len(rows) == 2


async def test_export_snapshots_category_filter(db_session):
    """Insert snapshots with different categories, filter by category."""
    crypto = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="CAT-CRYPTO",
        title="Crypto snapshot",
        category="crypto",
        feature_version="0.1.0",
    )
    politics = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="CAT-POLITICS",
        title="Politics snapshot",
        category="politics",
        feature_version="0.1.0",
    )
    db_session.add_all([crypto, politics])
    await db_session.commit()

    rows = await export_snapshots(db_session, category="crypto")
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "CAT-CRYPTO"


async def test_export_snapshots_source_filter(db_session):
    """Filter by source='kalshi'."""
    kalshi = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="SRC-KALSHI",
        title="Kalshi snapshot",
        category="test",
        feature_version="0.1.0",
    )
    poly = MarketSnapshot(
        source=Source.POLYMARKET,
        contract_id="SRC-POLY",
        title="Poly snapshot",
        category="test",
        feature_version="0.1.0",
    )
    db_session.add_all([kalshi, poly])
    await db_session.commit()

    rows = await export_snapshots(db_session, source="kalshi")
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "SRC-KALSHI"
    assert rows[0]["source"] == "kalshi"


async def test_export_snapshots_flattens_features(db_session):
    """Features dict keys appear as top-level keys in the exported dict."""
    snapshot = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="FLAT-001",
        title="Flatten test",
        category="test",
        features={"yes_price": 0.40, "no_price": 0.60, "spread": 0.02},
        feature_version="0.1.0",
    )
    db_session.add(snapshot)
    await db_session.commit()

    rows = await export_snapshots(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row["yes_price"] == 0.40
    assert row["no_price"] == 0.60
    assert row["spread"] == 0.02


async def test_export_snapshots_null_features(db_session):
    """Snapshot with features=None should still export cleanly (no feature keys)."""
    snapshot = MarketSnapshot(
        source=Source.KALSHI,
        contract_id="NULL-FEAT",
        title="Null features",
        category="test",
        feature_version="0.1.0",
    )
    db_session.add(snapshot)
    await db_session.commit()

    rows = await export_snapshots(db_session)
    assert len(rows) == 1
    # Should have standard keys but no extra feature keys
    assert "yes_price" not in rows[0]


# ---------------------------------------------------------------------------
# export_opportunities
# ---------------------------------------------------------------------------


async def test_export_opportunities(db_session):
    """Insert Opportunity rows, verify export returns all columns."""
    now = datetime.now(UTC)
    opp = Opportunity(
        source=Source.KALSHI,
        contract_id="OPP-001",
        title="Will X happen?",
        direction=Direction.YES,
        strategy_name="EVStrategy",
        market_price=0.40,
        model_probability=0.70,
        expected_value=0.30,
        kelly_size=0.20,
        active=True,
        last_seen_at=now,
    )
    db_session.add(opp)
    await db_session.commit()

    rows = await export_opportunities(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row["contract_id"] == "OPP-001"
    assert row["direction"] == "yes"
    assert row["strategy_name"] == "EVStrategy"
    assert row["market_price"] == 0.40
    assert row["model_probability"] == 0.70
    assert row["expected_value"] == 0.30
    assert row["kelly_size"] == 0.20
    assert row["active"] is True


async def test_export_opportunities_active_only(db_session):
    """Filter active_only=True returns only active opportunities."""
    now = datetime.now(UTC)
    active = Opportunity(
        source=Source.KALSHI,
        contract_id="ACTIVE-001",
        title="Active opp",
        direction=Direction.YES,
        market_price=0.40,
        model_probability=0.70,
        expected_value=0.30,
        kelly_size=0.20,
        active=True,
        last_seen_at=now,
    )
    inactive = Opportunity(
        source=Source.KALSHI,
        contract_id="INACTIVE-001",
        title="Inactive opp",
        direction=Direction.NO,
        market_price=0.80,
        model_probability=0.50,
        expected_value=-0.10,
        kelly_size=0.0,
        active=False,
        last_seen_at=now,
    )
    db_session.add_all([active, inactive])
    await db_session.commit()

    rows = await export_opportunities(db_session, active_only=True)
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "ACTIVE-001"

    # Without filter, both returned
    all_rows = await export_opportunities(db_session)
    assert len(all_rows) == 2


async def test_export_opportunities_start_date(db_session):
    """Filter by start_date on discovered_at."""
    old = Opportunity(
        source=Source.KALSHI,
        contract_id="OLD-OPP",
        title="Old opportunity",
        direction=Direction.YES,
        market_price=0.40,
        model_probability=0.70,
        expected_value=0.30,
        kelly_size=0.20,
        active=True,
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
        last_seen_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    new = Opportunity(
        source=Source.KALSHI,
        contract_id="NEW-OPP",
        title="New opportunity",
        direction=Direction.YES,
        market_price=0.40,
        model_probability=0.70,
        expected_value=0.30,
        kelly_size=0.20,
        active=True,
        discovered_at=datetime(2026, 6, 1, tzinfo=UTC),
        last_seen_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    db_session.add_all([old, new])
    await db_session.commit()

    rows = await export_opportunities(
        db_session,
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "NEW-OPP"


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


def test_to_csv_format():
    """Verify CSV output has proper headers and data rows."""
    rows = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"},
    ]
    result = to_csv(rows)
    reader = csv.reader(io.StringIO(result))
    lines = list(reader)
    assert len(lines) == 3  # header + 2 data rows
    assert lines[0] == ["name", "age", "city"]
    assert lines[1] == ["Alice", "30", "NYC"]
    assert lines[2] == ["Bob", "25", "LA"]


def test_to_csv_empty():
    """Empty input returns empty string."""
    assert to_csv([]) == ""


# ---------------------------------------------------------------------------
# export_paper_trades
# ---------------------------------------------------------------------------


async def test_export_paper_trades(db_session):
    """Insert PaperTrade rows, verify export returns all columns."""
    trade = PaperTrade(
        source=Source.KALSHI,
        contract_id="PT-001",
        direction=Direction.YES,
        strategy_name="EVStrategy",
        entry_price=0.40,
        quantity=10.0,
        model_probability=0.70,
        expected_value=0.29,
    )
    db_session.add(trade)
    await db_session.commit()

    rows = await export_paper_trades(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row["contract_id"] == "PT-001"
    assert row["direction"] == "yes"
    assert row["strategy_name"] == "EVStrategy"
    assert row["entry_price"] == 0.40
    assert row["pnl"] is None


async def test_export_paper_trades_settled_only(db_session):
    """Filter settled_only=True returns only settled trades."""
    now = datetime.now(UTC)
    open_trade = PaperTrade(
        source=Source.KALSHI,
        contract_id="PT-OPEN",
        direction=Direction.YES,
        strategy_name="EVStrategy",
        entry_price=0.40,
        quantity=10.0,
        model_probability=0.70,
        expected_value=0.29,
    )
    settled_trade = PaperTrade(
        source=Source.KALSHI,
        contract_id="PT-SETTLED",
        direction=Direction.NO,
        strategy_name="ConsistencyStrategy",
        entry_price=0.60,
        quantity=5.0,
        model_probability=0.50,
        expected_value=0.18,
        exit_price=1.0,
        exited_at=now,
        pnl=2.0,
        outcome=0.0,
    )
    db_session.add_all([open_trade, settled_trade])
    await db_session.commit()

    rows = await export_paper_trades(db_session, settled_only=True)
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "PT-SETTLED"

    all_rows = await export_paper_trades(db_session)
    assert len(all_rows) == 2


# ---------------------------------------------------------------------------
# export_order_book_snapshots
# ---------------------------------------------------------------------------


async def test_export_order_book_snapshots(db_session):
    """Export OrderBookSnapshot rows as list of dicts."""
    snap = OrderBookSnapshot(
        source=Source.KALSHI,
        contract_id="KXCPI-T3.0",
        series_ticker="KXCPI",
        event_ticker="KXCPI-26MAR",
        bids=[{"price": 0.55, "quantity": 100}],
        asks=[{"price": 0.57, "quantity": 150}],
    )
    db_session.add(snap)
    await db_session.commit()

    rows = await export_order_book_snapshots(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row["contract_id"] == "KXCPI-T3.0"
    assert row["series_ticker"] == "KXCPI"
    assert row["source"] == "kalshi"
    assert row["bids"] == [{"price": 0.55, "quantity": 100}]
    assert row["asks"] == [{"price": 0.57, "quantity": 150}]


async def test_export_order_book_snapshots_filter_by_contract(db_session):
    """Filter by contract_id."""
    for ticker in ["KXCPI-T3.0", "KXCPI-T3.5"]:
        snap = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id=ticker,
            bids=[],
            asks=[],
        )
        db_session.add(snap)
    await db_session.commit()

    rows = await export_order_book_snapshots(db_session, contract_id="KXCPI-T3.0")
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "KXCPI-T3.0"


# ---------------------------------------------------------------------------
# export_candlestick_bars
# ---------------------------------------------------------------------------


async def test_export_candlestick_bars(db_session):
    """Export CandlestickBar rows as list of dicts."""
    bar = CandlestickBar(
        source=Source.KALSHI,
        contract_id="KXCPI-T3.0",
        series_ticker="KXCPI",
        period_start=datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC),
        period_interval=60,
        open=0.55,
        high=0.60,
        low=0.52,
        close=0.58,
        volume=1200,
    )
    db_session.add(bar)
    await db_session.commit()

    rows = await export_candlestick_bars(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row["contract_id"] == "KXCPI-T3.0"
    assert row["series_ticker"] == "KXCPI"
    assert row["open"] == 0.55
    assert row["high"] == 0.60
    assert row["low"] == 0.52
    assert row["close"] == 0.58
    assert row["volume"] == 1200
    assert row["period_interval"] == 60


async def test_export_candlestick_bars_pagination(db_session):
    """Limit and offset should paginate results."""
    for i in range(5):
        bar = CandlestickBar(
            source=Source.KALSHI,
            contract_id=f"KXCPI-T{i}",
            series_ticker="KXCPI",
            period_start=datetime(2026, 3, 7, i, 0, 0, tzinfo=UTC),
            period_interval=60,
            open=0.50,
            high=0.55,
            low=0.48,
            close=0.52,
            volume=100,
        )
        db_session.add(bar)
    await db_session.commit()

    # First page
    rows = await export_candlestick_bars(db_session, limit=2)
    assert len(rows) == 2

    # Second page
    rows2 = await export_candlestick_bars(db_session, limit=2, offset=2)
    assert len(rows2) == 2

    # All rows
    all_rows = await export_candlestick_bars(db_session)
    assert len(all_rows) == 5


async def test_export_snapshots_pagination(db_session):
    """Limit and offset should paginate snapshot results."""
    for i in range(5):
        snap = MarketSnapshot(
            source=Source.KALSHI,
            contract_id=f"SNAP-{i}",
            title=f"Snap {i}",
            category="test",
            feature_version="0.1.0",
        )
        db_session.add(snap)
    await db_session.commit()

    rows = await export_snapshots(db_session, limit=3)
    assert len(rows) == 3

    rows2 = await export_snapshots(db_session, limit=3, offset=3)
    assert len(rows2) == 2


async def test_export_order_book_snapshots_pagination(db_session):
    """Limit and offset should paginate order book snapshot results."""
    for i in range(4):
        snap = OrderBookSnapshot(
            source=Source.KALSHI,
            contract_id=f"OB-{i}",
            bids=[],
            asks=[],
        )
        db_session.add(snap)
    await db_session.commit()

    rows = await export_order_book_snapshots(db_session, limit=2)
    assert len(rows) == 2


async def test_export_opportunities_pagination(db_session):
    """Limit and offset should paginate opportunity results."""
    for i in range(5):
        opp = Opportunity(
            source=Source.KALSHI,
            contract_id=f"OPP-{i}",
            title=f"Opp {i}",
            direction=Direction.YES,
            strategy_name="test",
            market_price=0.50,
            model_probability=0.60,
            expected_value=0.05,
            kelly_size=0.10,
        )
        db_session.add(opp)
    await db_session.commit()

    rows = await export_opportunities(db_session, limit=3)
    assert len(rows) == 3

    rows2 = await export_opportunities(db_session, limit=3, offset=3)
    assert len(rows2) == 2


async def test_export_paper_trades_pagination(db_session):
    """Limit and offset should paginate paper trade results."""
    for i in range(5):
        trade = PaperTrade(
            source=Source.KALSHI,
            contract_id=f"PT-{i}",
            direction=Direction.YES,
            strategy_name="test",
            entry_price=0.50,
            quantity=10,
            model_probability=0.60,
            expected_value=0.05,
        )
        db_session.add(trade)
    await db_session.commit()

    rows = await export_paper_trades(db_session, limit=2)
    assert len(rows) == 2

    rows2 = await export_paper_trades(db_session, limit=2, offset=2)
    assert len(rows2) == 2

    all_rows = await export_paper_trades(db_session)
    assert len(all_rows) == 5


async def test_export_candlestick_bars_filter_by_contract(db_session):
    """Filter by contract_id."""
    for ticker in ["KXCPI-T3.0", "KXCPI-T3.5"]:
        bar = CandlestickBar(
            source=Source.KALSHI,
            contract_id=ticker,
            series_ticker="KXCPI",
            period_start=datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC),
            period_interval=60,
            open=0.50,
            high=0.55,
            low=0.48,
            close=0.52,
            volume=100,
        )
        db_session.add(bar)
    await db_session.commit()

    rows = await export_candlestick_bars(db_session, contract_id="KXCPI-T3.0")
    assert len(rows) == 1
    assert rows[0]["contract_id"] == "KXCPI-T3.0"

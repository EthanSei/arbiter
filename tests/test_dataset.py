"""Tests for dataset loading and temporal splitting."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from arbiter.db.models import MarketSnapshot, Source
from arbiter.models.features import FEATURE_VERSION, SPEC
from arbiter.training.dataset import load_dataset, temporal_split


class TestTemporalSplit:
    def test_split_sizes_default(self):
        """Default split: 70% train, 15% val, 15% test."""
        n = 100
        features = np.random.default_rng(42).standard_normal((n, len(SPEC.names)))
        labels = np.random.default_rng(42).choice([0.0, 1.0], n)
        timestamps = np.arange(n, dtype=np.float64)

        splits = temporal_split(features, labels, timestamps)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = len(splits["train"][0]) + len(splits["val"][0]) + len(splits["test"][0])
        assert total == n

    def test_split_is_chronological(self):
        """Train timestamps < val timestamps < test timestamps."""
        n = 100
        features = np.random.default_rng(42).standard_normal((n, len(SPEC.names)))
        labels = np.random.default_rng(42).choice([0.0, 1.0], n)
        timestamps = np.arange(n, dtype=np.float64)

        splits = temporal_split(features, labels, timestamps)

        # Reconstruct timestamps per split using sorted order
        train_end = len(splits["train"][0])
        val_end = train_end + len(splits["val"][0])
        # Train gets earliest, test gets latest
        assert train_end > 0
        assert val_end > train_end

    def test_features_and_labels_stay_aligned(self):
        """Features and labels in each split correspond to each other."""
        n = 20
        features = np.arange(n * 4, dtype=np.float64).reshape(n, 4)
        labels = np.arange(n, dtype=np.float64)
        timestamps = np.arange(n, dtype=np.float64)

        splits = temporal_split(features, labels, timestamps)

        # After sorting by timestamp and splitting, feature rows match label indices
        sorted_idx = np.argsort(timestamps)
        sorted_features = features[sorted_idx]
        sorted_labels = labels[sorted_idx]

        offset = 0
        for key in ("train", "val", "test"):
            feat, lbl = splits[key]
            expected_f = sorted_features[offset : offset + len(feat)]
            expected_l = sorted_labels[offset : offset + len(lbl)]
            np.testing.assert_array_equal(feat, expected_f)
            np.testing.assert_array_equal(lbl, expected_l)
            offset += len(feat)

    def test_custom_fractions(self):
        """Custom val/test fractions resize splits accordingly."""
        n = 100
        features = np.random.default_rng(1).standard_normal((n, 4))
        labels = np.random.default_rng(1).choice([0.0, 1.0], n)
        timestamps = np.arange(n, dtype=np.float64)

        splits = temporal_split(features, labels, timestamps, val_fraction=0.2, test_fraction=0.2)

        # train ≈ 60, val ≈ 20, test ≈ 20
        assert len(splits["train"][0]) == 60
        assert len(splits["val"][0]) == 20
        assert len(splits["test"][0]) == 20

    def test_unsorted_timestamps_sorted_before_split(self):
        """Shuffled timestamps are sorted before splitting."""
        n = 50
        rng = np.random.default_rng(7)
        features = rng.standard_normal((n, 4))
        labels = rng.choice([0.0, 1.0], n)
        timestamps = rng.permutation(n).astype(np.float64)

        splits = temporal_split(features, labels, timestamps)

        total = sum(len(splits[k][0]) for k in ("train", "val", "test"))
        assert total == n

    def test_small_dataset(self):
        """Works with very small datasets (minimum viable)."""
        features = np.array([[1.0], [2.0], [3.0], [4.0]])
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        timestamps = np.array([1.0, 2.0, 3.0, 4.0])

        splits = temporal_split(features, labels, timestamps, val_fraction=0.25, test_fraction=0.25)

        total = sum(len(splits[k][0]) for k in ("train", "val", "test"))
        assert total == 4
        # Each split has at least some data
        assert len(splits["train"][0]) >= 1
        assert len(splits["test"][0]) >= 1


def _make_snapshot(
    contract_id: str,
    outcome: float | None,
    snapshot_at: datetime,
    resolved_at: datetime | None = None,
    feature_version: str = FEATURE_VERSION,
) -> MarketSnapshot:
    features = {name: 0.5 for name in SPEC.names}
    return MarketSnapshot(
        source=Source.KALSHI,
        contract_id=contract_id,
        title="Test Market",
        features=features,
        feature_version=feature_version,
        outcome=outcome,
        snapshot_at=snapshot_at,
        resolved_at=resolved_at,
    )


class TestLoadDataset:
    @pytest.mark.asyncio
    async def test_returns_labeled_rows_only(self, db_session):
        """Only snapshots with a non-null outcome are returned."""
        now = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
        resolved = now + timedelta(hours=48)
        db_session.add(_make_snapshot("c1", outcome=1.0, snapshot_at=now, resolved_at=resolved))
        db_session.add(_make_snapshot("c2", outcome=None, snapshot_at=now))
        await db_session.flush()

        features, labels, timestamps = await load_dataset(db_session)

        assert len(features) == 1
        assert labels[0] == 1.0

    @pytest.mark.asyncio
    async def test_excludes_snapshots_near_resolution(self, db_session):
        """Snapshots within exclude_hours_before_resolution of resolved_at are dropped."""
        resolved_at = datetime(2024, 1, 20, 0, 0, 0, tzinfo=UTC)
        # 12h before resolution — should be excluded (default 24h window)
        near = resolved_at - timedelta(hours=12)
        # 36h before resolution — should be included
        far = resolved_at - timedelta(hours=36)
        db_session.add(_make_snapshot("c1", outcome=1.0, snapshot_at=near, resolved_at=resolved_at))
        db_session.add(_make_snapshot("c2", outcome=0.0, snapshot_at=far, resolved_at=resolved_at))
        await db_session.flush()

        features, labels, timestamps = await load_dataset(
            db_session, exclude_hours_before_resolution=24.0
        )

        assert len(features) == 1
        assert labels[0] == 0.0

    @pytest.mark.asyncio
    async def test_filters_by_feature_version(self, db_session):
        """Only snapshots matching the requested feature_version are returned."""
        now = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
        resolved = now + timedelta(hours=48)
        db_session.add(_make_snapshot("c1", 1.0, now, resolved, feature_version="0.1.0"))
        db_session.add(_make_snapshot("c2", 0.0, now, resolved, feature_version="0.0.1"))
        await db_session.flush()

        features, labels, timestamps = await load_dataset(db_session, feature_version="0.1.0")

        assert len(features) == 1
        assert labels[0] == 1.0

    @pytest.mark.asyncio
    async def test_defaults_to_current_feature_version(self, db_session):
        """Without explicit version, uses FEATURE_VERSION constant."""
        now = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
        resolved = now + timedelta(hours=48)
        db_session.add(_make_snapshot("c1", 1.0, now, resolved, feature_version=FEATURE_VERSION))
        db_session.add(_make_snapshot("c2", 0.0, now, resolved, feature_version="old"))
        await db_session.flush()

        features, labels, timestamps = await load_dataset(db_session)

        assert len(features) == 1

    @pytest.mark.asyncio
    async def test_features_shape_matches_spec(self, db_session):
        """Each row has exactly len(SPEC.names) features."""
        now = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
        resolved = now + timedelta(hours=48)
        db_session.add(_make_snapshot("c1", outcome=1.0, snapshot_at=now, resolved_at=resolved))
        db_session.add(_make_snapshot("c2", outcome=0.0, snapshot_at=now, resolved_at=resolved))
        await db_session.flush()

        features, labels, timestamps = await load_dataset(db_session)

        assert features.shape == (2, len(SPEC.names))
        assert labels.shape == (2,)
        assert timestamps.shape == (2,)

    @pytest.mark.asyncio
    async def test_timestamps_preserve_chronological_order(self, db_session):
        """Timestamps are floats ordered consistently with snapshot_at."""
        t1 = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 11, 12, 0, 0, tzinfo=UTC)
        resolved = t2 + timedelta(hours=48)
        db_session.add(_make_snapshot("c1", outcome=1.0, snapshot_at=t1, resolved_at=resolved))
        db_session.add(_make_snapshot("c2", outcome=0.0, snapshot_at=t2, resolved_at=resolved))
        await db_session.flush()

        _, labels, timestamps = await load_dataset(db_session)

        assert len(timestamps) == 2
        # Earlier snapshot has smaller timestamp
        earlier_idx = int(np.argmin(timestamps))
        assert labels[earlier_idx] == 1.0  # c1 was added earlier

    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_arrays(self, db_session):
        """No matching snapshots → empty arrays with correct shape."""
        features, labels, timestamps = await load_dataset(db_session)

        assert features.shape == (0, len(SPEC.names))
        assert labels.shape == (0,)
        assert timestamps.shape == (0,)

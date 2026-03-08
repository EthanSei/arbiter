# ── Setup & Quality ───────────────────────────────────────────────────────────
.PHONY: install dev install-data clean lint format test typecheck check integrate

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

install-data:
	pip install -e ".[data]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/ -v

typecheck:
	mypy arbiter/

check: lint test

integrate:
	pytest -m integration -v tests/test_integration.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ── Run ───────────────────────────────────────────────────────────────────────
.PHONY: run run-stdout

run:
	python -m arbiter

run-stdout:
	python -m arbiter --stdout

# ── EV Strategy ──────────────────────────────────────────────────────────────
# Workflow: ev-data → ev-train → ev-backtest → run
.PHONY: ev-data ev-train ev-backtest ev-analyze

ev-data:
	python scripts/collect_training_data.py

ev-train:
	python -m arbiter.training.train --data-path data/training.csv

ev-backtest:
	python scripts/backtest_ev.py

ev-analyze:
	python scripts/analyze_mispricings.py

# ── Anchor Strategy ──────────────────────────────────────────────────────────
# Workflow: anchor-data → anchor-fit → anchor-eval → run
.PHONY: anchor-data anchor-fit anchor-eval anchor-scan

anchor-data:
	python scripts/fetch_fred_data.py
	python scripts/fetch_bls_data.py
	python scripts/fetch_candlestick_prices.py

anchor-fit:
	python scripts/fit_anchor_calibrators.py

anchor-eval:
	python scripts/evaluate_anchor_calibration.py

anchor-scan:
	python scripts/scan_violations.py

# ── Maker Strategy ───────────────────────────────────────────────────────────
# Workflow: maker-data → run notebook (all data saved to data/*.parquet)
.PHONY: maker-data maker-candles maker-orderbooks maker-snapshots

maker-data: maker-snapshots maker-orderbooks maker-candles

maker-candles:
	python scripts/backfill_candles.py

maker-orderbooks:
	python scripts/collect_orderbooks.py

maker-snapshots:
	python scripts/export_snapshots.py

# ── Utilities ─────────────────────────────────────────────────────────────────
.PHONY: backfill-outcomes analyze-categories debug-kalshi reset-db

backfill-outcomes:
	python scripts/backfill_outcomes.py

analyze-categories:
	python scripts/analyze_categories.py

debug-kalshi:
	python scripts/debug_kalshi.py

reset-db:
	python scripts/reset_db.py

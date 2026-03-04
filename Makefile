.PHONY: install dev install-data lint format test typecheck check clean run run-stdout train integrate collect-data backtest fetch-fred fetch-bls fetch-data

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

# Run all checks (lint + test)
check: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

run:
	python -m arbiter

run-stdout:
	python -m arbiter --stdout

train:
	python -m arbiter.training.train --data-path data/training.csv

collect-data:
	python scripts/collect_training_data.py

backtest:
	python scripts/backtest_ev.py

integrate:
	pytest -m integration -v tests/test_integration.py

# Fetch external data (requires FRED_API_KEY / BLS_API_KEY in env or .env)
fetch-fred:
	python scripts/fetch_fred_data.py

fetch-bls:
	python scripts/fetch_bls_data.py

fetch-data: fetch-fred fetch-bls

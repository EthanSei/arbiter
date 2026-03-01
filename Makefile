.PHONY: install dev lint format test typecheck check clean run train integrate

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

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

train:
	python -m arbiter.training.train

integrate:
	pytest -m integration -v tests/test_integration.py

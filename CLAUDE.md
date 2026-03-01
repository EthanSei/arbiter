# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arbiter is an automated prediction market scanner that identifies mispriced event contracts across Kalshi and Polymarket. It runs as a persistent async Python process that polls market APIs, estimates true probabilities using a LightGBM model trained on historical market data, computes expected value (including execution costs), and sends alerts via Discord/SMS.

## Tech Stack & Conventions

- **Python 3.12**, async/await throughout (httpx for HTTP)
- **pyproject.toml** (hatchling) for dependency management
- **SQLAlchemy** (async) for ORM — Postgres via asyncpg in production, SQLite via aiosqlite for tests only
- **LightGBM** + scikit-learn for the probability estimation model with isotonic calibration
- **pydantic-settings** for configuration from `.env` (see `.env.example`)
- **ruff** for linting and formatting (line-length 100, target py311)

## Build & Run Commands

```bash
# Setup
make dev                    # Install with dev dependencies
make install                # Install production only

# Development
make test                   # Run all tests
make lint                   # Lint with ruff
make format                 # Format with ruff
make typecheck              # Type check with mypy
make check                  # Lint + test

# Run a single test
pytest tests/test_scoring.py::TestKellyCriterion::test_positive_edge

# Run the scanner
make run                    # python -m arbiter

# Train the model
make train                  # python -m arbiter.training.train
```

## Architecture

Pipeline: **Ingest → Match Cross-Platform → Extract Features → LightGBM Estimate → EV Score → Filter/Dedup → Alert → Log**

Each poll cycle also snapshots market data to `MarketSnapshot` for continuous training data collection.

### Package layout

- **arbiter/ingestion/** — Async API clients for Kalshi and Polymarket. `Contract` frozen dataclass is the common data type. `MarketClient` ABC defines the interface. `RateLimitedClient` wraps httpx with token-bucket rate limiting. `ContractMatcher` matches contracts across platforms.
- **arbiter/models/** — `ProbabilityEstimator` ABC. `LGBMEstimator` loads trained LightGBM model, extracts features via `features.py`, applies isotonic calibration. Falls back to market midpoint when no model loaded.
- **arbiter/scoring/** — `compute_ev()` computes expected value for both YES and NO sides, subtracting execution costs (`fee_rate`). `kelly_criterion()` for position sizing. `ScoredOpportunity` dataclass wraps Contract with computed fields.
- **arbiter/alerts/** — `AlertChannel` ABC. `DiscordChannel` (httpx webhook POST) and `SMSChannel` (raw httpx POST to Twilio API). Both degrade gracefully when credentials missing.
- **arbiter/db/** — SQLAlchemy models: `Opportunity` (state-based dedup with `active`/`last_alerted_at`/`last_seen_at`), `AlertLog`, `MarketSnapshot` (features JSON with `feature_version`, nullable `outcome` backfilled on resolution).
- **arbiter/training/** — Offline training pipeline: historical data collector, temporal train/val/test splits, LightGBM training with calibration evaluation (Brier score, ECE).
- **arbiter/scheduler.py** — `ScanPipeline` with bare asyncio loop (not APScheduler). Dynamic cycle interval.
- **arbiter/health.py** — Health endpoint on `:8080` for Cloud Run liveness probes.

### Key design patterns

- All external API calls are async via httpx, wrapped with `RateLimitedClient`
- State-based deduplication: `Opportunity.active` flag + `last_alerted_at` instead of time-windowed
- EV calculation subtracts execution costs (spread + platform fees) before threshold comparison
- Training data uses temporal splits (not random), excludes near-expiry snapshots, and versions features
- Alert channels degrade gracefully — skip with warning when credentials missing

## Infrastructure

- **Database:** Supabase Postgres (free tier). `DATABASE_URL` env var.
- **Deployment:** GCP Cloud Run. Container built from `Dockerfile`. Env vars injected by Cloud Run / Secret Manager.
- **Model weights:** `models/arbiter_lgbm.pkl` bundled in container or fetched from GCS.

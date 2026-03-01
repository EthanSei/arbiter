# arbiter

Automated prediction market scanner that identifies mispriced event contracts across [Kalshi](https://kalshi.com) and [Polymarket](https://polymarket.com).

Each scan cycle fetches live markets from both platforms, estimates true probabilities using a LightGBM model trained on historical data, computes expected value net of execution costs, and fires Discord alerts when edge exceeds a configurable threshold. Market snapshots are persisted to Postgres for continuous model training.

## Architecture

```
Ingest (Kalshi + Polymarket)
  → Cross-platform contract matching
  → Feature extraction
  → LightGBM probability estimate
  → EV scoring (net of fees)
  → State-based deduplication
  → Discord alert
  → Postgres snapshot
```

Runs as a persistent async Python process on **GCP Cloud Run** backed by **Supabase Postgres**.

---

## Prerequisites

- Python 3.12
- A [Supabase](https://supabase.com) project (free tier is sufficient)
- A [GCP](https://cloud.google.com) project with Cloud Run and Artifact Registry enabled
- A Discord webhook URL for alerts
- (Optional) Trained model weights at `models/arbiter_lgbm.pkl`

---

## Local Development

### 1. Install dependencies

```bash
make dev
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | Supabase connection string (`postgresql+asyncpg://...`) |
| `DISCORD_WEBHOOK_URL` | Yes | Discord channel webhook for alerts |
| `KALSHI_API_BASE` | No | Defaults to public Kalshi API |
| `POLYMARKET_GAMMA_BASE` | No | Defaults to public Polymarket Gamma API |
| `EV_THRESHOLD` | No | Minimum expected value to alert (default: `0.05`) |
| `FEE_RATE` | No | Platform fee rate subtracted from EV (default: `0.01`) |
| `POLL_INTERVAL_SECONDS` | No | Seconds between scan cycles (default: `300`) |
| `KELLY_FRACTION` | No | Fractional Kelly position sizing (default: `0.25`) |
| `MODEL_WEIGHTS_PATH` | No | Path to trained LightGBM `.pkl` file (default: `models/arbiter_lgbm.pkl`) |
| `HEALTH_PORT` | No | Port for health check endpoint (default: `8080`) |

Get your Supabase connection string from **Project Settings → Database → Connection string → URI**, then replace the `postgres://` scheme with `postgresql+asyncpg://`.

### 3. Run locally

```bash
make run
```

The scanner starts polling immediately. A health endpoint is available at `http://localhost:8080/health`.

### 4. Train the model (optional)

If you have collected market snapshots in the database:

```bash
make train
```

This writes trained weights to `models/arbiter_lgbm.pkl`. Without a model file the scanner falls back to using the market midpoint as the probability estimate.

---

## Testing

```bash
make test        # run all tests
make lint        # ruff check
make typecheck   # mypy
make check       # lint + test
```

Tests use an in-memory SQLite database — no Postgres connection required.

---

## Deployment (GCP Cloud Run + Supabase)

### 1. Authenticate

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable required services

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### 3. Create an Artifact Registry repository

```bash
gcloud artifacts repositories create arbiter \
  --repository-format=docker \
  --location=us-central1
```

### 4. Build and push the image

```bash
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/arbiter/arbiter:latest

gcloud auth configure-docker $REGION-docker.pkg.dev

docker build --target production -t $IMAGE .
docker push $IMAGE
```

The Dockerfile runs the full test suite before producing the production image — if tests fail, the build fails.

### 5. Store secrets in Secret Manager

```bash
echo -n "postgresql+asyncpg://..." | \
  gcloud secrets create DATABASE_URL --data-file=-

echo -n "https://discord.com/api/webhooks/..." | \
  gcloud secrets create DISCORD_WEBHOOK_URL --data-file=-
```

### 6. Deploy to Cloud Run

```bash
gcloud run deploy arbiter \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --min-instances 1 \
  --port 8080 \
  --set-secrets="DATABASE_URL=DATABASE_URL:latest,DISCORD_WEBHOOK_URL=DISCORD_WEBHOOK_URL:latest" \
  --set-env-vars="EV_THRESHOLD=0.05,POLL_INTERVAL_SECONDS=300,FEE_RATE=0.01" \
  --no-allow-unauthenticated
```

Set `--min-instances 1` to prevent cold starts from missing scan cycles.

### 7. Verify

```bash
# Check logs
gcloud run services logs read arbiter --region $REGION

# Check health
SERVICE_URL=$(gcloud run services describe arbiter --region $REGION --format='value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" $SERVICE_URL/health
```

A healthy response looks like:

```json
{
  "status": "healthy",
  "last_cycle_completed_at": "2025-01-01T12:00:00+00:00",
  "last_cycle_duration_seconds": 4.2,
  "markets_scanned": 312,
  "opportunities_found": 2,
  "errors": []
}
```

---

## Supabase Setup

Arbiter creates its own tables on first run via SQLAlchemy (`init_db()`). No manual migrations needed.

To inspect data:

```sql
-- Active opportunities above EV threshold
SELECT source, title, direction, market_price, model_probability, expected_value, kelly_size
FROM opportunity
WHERE active = true
ORDER BY expected_value DESC;

-- Alert delivery history
SELECT o.title, al.channel, al.success, al.error_message, al.sent_at
FROM alert_log al
JOIN opportunity o ON al.opportunity_id = o.id
ORDER BY al.sent_at DESC
LIMIT 50;

-- Training data snapshots
SELECT source, contract_id, feature_version, captured_at
FROM market_snapshot
ORDER BY captured_at DESC
LIMIT 20;
```

---

## Configuration Reference

All settings are read from environment variables (or `.env` locally). See [`.env.example`](.env.example) for the full list with defaults.

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost/postgres` | Must use `asyncpg` driver |
| `DISCORD_WEBHOOK_URL` | _(empty)_ | Alerts are skipped if blank |
| `EV_THRESHOLD` | `0.05` | Must be > 0 |
| `FEE_RATE` | `0.01` | Must be ≥ 0 |
| `POLL_INTERVAL_SECONDS` | `300` | Must be > 0 |
| `KELLY_FRACTION` | `0.25` | Must be in (0, 1] |
| `MAX_MARKETS_PER_POLL` | `200` | Per-platform market cap |
| `HEALTH_PORT` | `8080` | Cloud Run expects port 8080 |
| `MODEL_WEIGHTS_PATH` | `models/arbiter_lgbm.pkl` | Falls back to midpoint if missing |
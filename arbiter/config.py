"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database (Supabase Postgres)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

    # Kalshi
    kalshi_api_base: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Polymarket
    polymarket_clob_base: str = "https://clob.polymarket.com"
    polymarket_gamma_base: str = "https://gamma-api.polymarket.com"

    # Alerts — Discord
    discord_webhook_url: str = ""

    # Alerts — SMS (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""

    # Anthropic (future — not used in V1)
    anthropic_api_key: str = ""

    # Scanning parameters
    poll_interval_seconds: int = Field(default=300, gt=0)
    ev_threshold: float = Field(default=0.05, ge=0)
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    max_markets_per_poll: int = Field(default=200, gt=0)
    fee_rate: float = Field(default=0.01, ge=0)

    # Health check
    health_port: int = 8080

    # Model weights
    model_weights_path: str = "models/arbiter_lgbm.pkl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

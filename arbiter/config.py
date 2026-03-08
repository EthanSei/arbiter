"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database (Supabase Postgres)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

    # Kalshi
    kalshi_api_base: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Polymarket
    polymarket_enabled: bool = False
    polymarket_clob_base: str = "https://clob.polymarket.com"
    polymarket_gamma_base: str = "https://gamma-api.polymarket.com"

    # Alerts — Discord
    discord_webhook_url: str = ""

    # Data API Keys
    fred_api_key: str = ""
    bls_api_key: str = ""

    # Alerts — SMS (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""

    # Anthropic (future — not used in V1)
    anthropic_api_key: str = ""

    # Targeted series (comma-separated Kalshi series tickers)
    # Economics: CPI, CPI YoY, Core CPI YoY, Payrolls, Jobless Claims, GDP, Fed Funds
    # Crypto: BTC daily, ETH daily
    # Equities: S&P 500, Nasdaq 100
    kalshi_target_series: str = (
        "KXCPI,KXCPIYOY,KXCPICOREYOY,KXPAYROLLS,KXJOBLESSCLAIMS,"
        "KXGDP,KXFED,"
        "KXBTCD,KXETHD,"
        "KXINXU,KXNASDAQ100U"
    )

    # Scanning parameters
    poll_interval_seconds: int = Field(default=300, gt=0)
    ev_threshold: float = Field(default=0.05, gt=0)
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    max_markets_per_poll: int = Field(default=10_000, gt=0)
    kalshi_max_empty_pages: int = Field(default=10, gt=0)
    min_volume_24h: float = Field(default=5.0, ge=0)
    fee_rate: float = Field(default=0.01, ge=0)  # legacy flat rate, ignored when fee_model=kalshi
    fee_model: str = Field(default="kalshi")  # "kalshi" (parabolic) or "flat"

    # Trading safety
    paper_trade_only: bool = True

    # Data collection
    data_collection_rpm: int = Field(default=60, gt=0)
    orderbook_top_n: int = Field(default=20, gt=0)

    # Health check
    health_port: int = 8080

    # Model weights
    model_weights_path: str = "models/arbiter_lgbm.pkl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

"""Configuration from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ShopSettings:
    database_url: str
    smtp_host: str
    payment_api_key: str
    currency: str = "USD"


def load_settings() -> ShopSettings:
    """Read DATABASE_URL, SMTP_HOST, PAYMENT_API_KEY and SHOP_CURRENCY from the environment."""
    return ShopSettings(
        database_url=os.environ["DATABASE_URL"],
        smtp_host=os.environ["SMTP_HOST"],
        payment_api_key=os.environ["PAYMENT_API_KEY"],
        currency=os.environ.get("SHOP_CURRENCY", "USD"),
    )

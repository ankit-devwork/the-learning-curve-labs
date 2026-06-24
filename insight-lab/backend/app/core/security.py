"""Security helpers — production detection, client IP, CORS validation."""

from __future__ import annotations

import os

from fastapi import Request

from app.core.config import settings
from app.core.yaml_config import get_yaml_config


def is_production() -> bool:
    """True when either yaml app.env or APP_ENV indicates production."""
    yaml_env = get_yaml_config().app.env.lower()
    app_env = settings.app_env.lower()
    prod_values = {"production", "prod"}
    return yaml_env in prod_values or app_env in prod_values


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64] or "unknown"
    if request.client and request.client.host:
        return request.client.host[:64]
    return "unknown"


def validate_cors_origins_at_startup() -> None:
    """Reject obviously unsafe CORS config in production."""
    if not is_production():
        return
    raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins:
        raise RuntimeError("CORS_ALLOW_ORIGINS must be set in production")
    for origin in origins:
        if origin == "*":
            raise RuntimeError("CORS_ALLOW_ORIGINS must not use wildcard with credentials")
        if not origin.startswith("https://"):
            raise RuntimeError(f"Production CORS origin must use HTTPS: {origin}")

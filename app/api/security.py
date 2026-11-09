"""API authentication and rate limiting configuration."""

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Require a valid API key when API-key authentication is enabled."""
    if not settings.api_key_enabled:
        return

    configured_key = settings.api_key
    if not api_key or not configured_key or not secrets.compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def validate_security_config() -> None:
    """Fail fast when production security is not configured safely."""
    if settings.is_production and (not settings.api_key_enabled or not settings.api_key):
        raise ValueError(
            "Production requires API_KEY_ENABLED=true and a non-empty API_KEY"
        )
    if settings.rate_limit_per_minute < 1:
        raise ValueError("RATE_LIMIT_PER_MINUTE must be greater than zero")

"""Tests for API authentication and production security validation."""

import pytest
from fastapi import HTTPException

from app.api.security import validate_security_config, verify_api_key
from app.config import settings


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_is_optional_when_authentication_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "api_key_enabled", False)
    monkeypatch.setattr(settings, "api_key", None)

    assert await verify_api_key(None) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_rejects_missing_or_invalid_values(monkeypatch):
    monkeypatch.setattr(settings, "api_key_enabled", True)
    monkeypatch.setattr(settings, "api_key", "expected-secret")

    with pytest.raises(HTTPException) as missing:
        await verify_api_key(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as invalid:
        await verify_api_key("wrong-secret")
    assert invalid.value.status_code == 401


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_accepts_matching_value(monkeypatch):
    monkeypatch.setattr(settings, "api_key_enabled", True)
    monkeypatch.setattr(settings, "api_key", "expected-secret")

    assert await verify_api_key("expected-secret") is None


@pytest.mark.unit
def test_production_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "api_key_enabled", False)
    monkeypatch.setattr(settings, "api_key", None)

    with pytest.raises(ValueError, match="Production requires API_KEY_ENABLED=true"):
        validate_security_config()


@pytest.mark.unit
def test_rate_limit_must_be_positive(monkeypatch):
    monkeypatch.setattr(settings, "environment", "local")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)

    with pytest.raises(ValueError, match="RATE_LIMIT_PER_MINUTE"):
        validate_security_config()

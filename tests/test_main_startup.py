"""Boot-time guard tests — refuse to start in production with auth off
or SEC EDGAR identity unset.

Without these guards, a misconfigured Railway deploy would either:
- trust `X-User-Id` as-is (DISABLE_AUTH=true / CLERK_SECRET_KEY unset) and
  leak every user's sessions/watchlist/briefings, or
- send all SEC EDGAR requests under a generic identity, getting the app's
  IP rate-limited or banned.

Two layers of coverage:
- Direct unit tests on the guard functions — fast, exhaustive.
- Integration tests via `TestClient(app)` — exercises the real `lifespan`
  startup path, proving uvicorn would fail in the same condition.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    check_production_auth_config,
    check_production_sec_config,
    set_sec_identity,
)


@pytest.fixture
def prod_env(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)


def test_passes_outside_production(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    check_production_auth_config()


def test_passes_with_clerk_configured_in_production(prod_env, monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    check_production_auth_config()


def test_refuses_when_clerk_unset_in_production(prod_env, monkeypatch):
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="CLERK_SECRET_KEY"):
        check_production_auth_config()


def test_refuses_when_disable_auth_in_production(prod_env, monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("DISABLE_AUTH", "true")
    with pytest.raises(RuntimeError, match="DISABLE_AUTH"):
        check_production_auth_config()


def test_escape_hatch_overrides_guard(prod_env, monkeypatch):
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.setenv("DISABLE_AUTH", "true")
    monkeypatch.setenv("ANALYST_ALLOW_DISABLED_AUTH", "1")
    check_production_auth_config()


def test_prod_env_value_accepted(monkeypatch):
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError):
        check_production_auth_config()


def test_env_case_insensitive(monkeypatch):
    monkeypatch.setenv("ENV", "PRODUCTION")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError):
        check_production_auth_config()


# ---------------------------------------------------------------------------
# Integration: real FastAPI lifespan via TestClient
# ---------------------------------------------------------------------------


def test_app_refuses_to_start_in_prod_without_clerk(monkeypatch):
    """Mirror the manual smoke test: `ENV=production CLERK_SECRET_KEY=` must abort startup."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="CLERK_SECRET_KEY"):
        with TestClient(app):
            pass


def test_app_refuses_to_start_in_prod_with_disable_auth(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("DISABLE_AUTH", "true")
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="DISABLE_AUTH"):
        with TestClient(app):
            pass


def test_app_starts_in_prod_with_clerk_configured(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_app_starts_in_dev_without_clerk(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# SEC_HEADER startup guard
# ---------------------------------------------------------------------------


def test_sec_guard_passes_outside_production(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("SEC_HEADER", raising=False)
    check_production_sec_config()


def test_sec_guard_passes_with_sec_header_in_production(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SEC_HEADER", "Real Person real@example.com")
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    check_production_sec_config()


def test_sec_guard_refuses_when_sec_header_unset_in_production(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("SEC_HEADER", raising=False)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="SEC_HEADER"):
        check_production_sec_config()


def test_sec_guard_refuses_default_placeholder_in_production(monkeypatch):
    """The default placeholder must NOT pass in production — it would get the app banned."""
    from api.main import _DEFAULT_SEC_HEADER

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SEC_HEADER", _DEFAULT_SEC_HEADER)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="SEC_HEADER"):
        check_production_sec_config()


def test_sec_guard_escape_hatch_overrides(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("SEC_HEADER", raising=False)
    monkeypatch.setenv("ANALYST_ALLOW_DISABLED_AUTH", "1")
    check_production_sec_config()


# ---------------------------------------------------------------------------
# set_sec_identity — configures edgartools once at startup
# ---------------------------------------------------------------------------


def test_set_sec_identity_uses_env_value(monkeypatch):
    monkeypatch.setenv("SEC_HEADER", "Custom Person custom@example.com")
    with patch("api.main.set_identity") as mock_set_identity:
        set_sec_identity()
    mock_set_identity.assert_called_once_with("Custom Person custom@example.com")


def test_set_sec_identity_falls_back_to_default(monkeypatch):
    """When SEC_HEADER is unset (dev/local), the documented placeholder is used."""
    from api.main import _DEFAULT_SEC_HEADER

    monkeypatch.delenv("SEC_HEADER", raising=False)
    with patch("api.main.set_identity") as mock_set_identity:
        set_sec_identity()
    mock_set_identity.assert_called_once_with(_DEFAULT_SEC_HEADER)


# ---------------------------------------------------------------------------
# Integration: real lifespan with SEC guard
# ---------------------------------------------------------------------------


def test_app_refuses_to_start_in_prod_without_sec_header(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.delenv("SEC_HEADER", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="SEC_HEADER"):
        with TestClient(app):
            pass


def test_app_starts_in_prod_with_sec_header_configured(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("SEC_HEADER", "Real Person real@example.com")
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

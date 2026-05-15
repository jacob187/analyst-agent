"""Boot-time auth guard tests — refuse to start in production with auth off.

Without this guard, a misconfigured Railway deploy (DISABLE_AUTH=true or
CLERK_SECRET_KEY unset) would trust `X-User-Id` as-is and leak every user's
sessions/watchlist/briefings.

Two layers of coverage:
- Direct unit tests on `check_production_auth_config` — fast, exhaustive.
- Integration tests via `TestClient(app)` — exercises the real `lifespan`
  startup path, proving uvicorn would fail in the same condition.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, check_production_auth_config


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

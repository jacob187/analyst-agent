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

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.db as db_module
from api.main import (
    app,
    check_production_auth_config,
    check_production_sec_config,
    resolve_log_level,
    set_sec_identity,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """The lifespan runs init_db(); keep it off the real data/chats.db."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "startup.db")
    monkeypatch.setattr(db_module, "_db", None)


@pytest.fixture
def prod_env(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)


def test_passes_outside_production(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    check_production_auth_config()


_TEST_ISSUER = "https://test.clerk.test"


def test_passes_with_clerk_configured_in_production(prod_env, monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("CLERK_ISSUER_URL", _TEST_ISSUER)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    check_production_auth_config()


def test_refuses_when_clerk_issuer_unset_in_production(prod_env, monkeypatch):
    # Without the issuer no token can be verified: every signed-in user
    # would get 503s, so fail at boot instead.
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.delenv("CLERK_ISSUER_URL", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="CLERK_ISSUER_URL"):
        check_production_auth_config()


def test_escape_hatch_overrides_missing_issuer(prod_env, monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.delenv("CLERK_ISSUER_URL", raising=False)
    monkeypatch.setenv("ANALYST_ALLOW_DISABLED_AUTH", "1")
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
    monkeypatch.setenv("CLERK_ISSUER_URL", _TEST_ISSUER)
    # Production also requires a real SEC identity; set it here so the test
    # doesn't silently depend on a developer's .env.
    monkeypatch.setenv("SEC_HEADER", "Real Person real@example.com")
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_app_starts_in_dev_without_clerk(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def _aiosqlite_threads() -> list[threading.Thread]:
    # aiosqlite names nothing; its worker runs `_connection_worker_thread`.
    return [
        t for t in threading.enumerate()
        if getattr(getattr(t, "_target", None), "__name__", "") == "_connection_worker_thread"
    ]


def test_failed_db_init_closes_the_connection(tmp_path, monkeypatch):
    """A corrupt DB fails startup; the connection opened for it must be closed,
    or its non-daemon worker thread keeps uvicorn from exiting."""
    monkeypatch.setenv("ENV", "development")
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(os.urandom(4096))
    monkeypatch.setattr(db_module, "DB_PATH", garbage)
    before = set(_aiosqlite_threads())

    with pytest.raises(Exception, match="not a database"):
        with TestClient(app):
            pass

    assert db_module._db is None
    leaked = [t for t in _aiosqlite_threads() if t not in before]
    for t in leaked:
        t.join(timeout=2)
    assert not [t for t in leaked if t.is_alive()]


def test_schema_failure_after_connect_closes_the_connection(tmp_path, monkeypatch):
    """The connection opens fine but a schema statement fails (a `sessions`
    view can't be indexed): the lifespan must still close it."""
    import sqlite3

    monkeypatch.setenv("ENV", "development")
    path = tmp_path / "view.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE VIEW sessions AS SELECT 1 AS x")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db_module, "DB_PATH", path)

    with pytest.raises(sqlite3.OperationalError, match="views may not be indexed"):
        with TestClient(app):
            pass

    assert db_module._db is None


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
    monkeypatch.setenv("CLERK_ISSUER_URL", _TEST_ISSUER)
    monkeypatch.delenv("SEC_HEADER", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("ANALYST_ALLOW_DISABLED_AUTH", raising=False)
    with pytest.raises(RuntimeError, match="SEC_HEADER"):
        with TestClient(app):
            pass


def test_app_starts_in_prod_with_sec_header_configured(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("CLERK_ISSUER_URL", _TEST_ISSUER)
    monkeypatch.setenv("SEC_HEADER", "Real Person real@example.com")
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# LOG_LEVEL / EDGAR_LOG_LEVEL — an invalid value must not crash startup
# ---------------------------------------------------------------------------


def test_resolve_log_level_unset_uses_default(monkeypatch, caplog):
    monkeypatch.delenv("EDGAR_LOG_LEVEL", raising=False)
    caplog.set_level(logging.WARNING, logger="api.main")
    assert resolve_log_level("EDGAR_LOG_LEVEL", "WARNING") == logging.WARNING
    assert caplog.records == []


@pytest.mark.parametrize("raw, expected", [
    ("error", logging.ERROR),
    ("DEBUG", logging.DEBUG),
    (" Info ", logging.INFO),
])
def test_resolve_log_level_applies_valid_value(monkeypatch, caplog, raw, expected):
    monkeypatch.setenv("EDGAR_LOG_LEVEL", raw)
    caplog.set_level(logging.WARNING, logger="api.main")
    assert resolve_log_level("EDGAR_LOG_LEVEL", "WARNING") == expected
    assert caplog.records == []


@pytest.mark.parametrize("raw", ["verbose", "", "10"])
def test_resolve_log_level_invalid_falls_back_and_warns_once(monkeypatch, caplog, raw):
    monkeypatch.setenv("EDGAR_LOG_LEVEL", raw)
    caplog.set_level(logging.WARNING, logger="api.main")
    assert resolve_log_level("EDGAR_LOG_LEVEL", "WARNING") == logging.WARNING
    warnings = [r for r in caplog.records if r.name == "api.main"]
    assert len(warnings) == 1
    assert "EDGAR_LOG_LEVEL" in warnings[0].getMessage()


_PRINT_LEVELS = "print(logging.getLogger().level, logging.getLogger('edgar').level)"


def _import_main_with_env(*, prelude: str = "", **overrides: str) -> subprocess.CompletedProcess:
    """Import api.main in a fresh interpreter (module-level logging setup runs
    once per process, so reloading it in-process would swap `app` under other
    tests) and print the resulting root and edgar logger levels."""
    env = {**os.environ, **overrides}
    return subprocess.run(
        [sys.executable, "-c", f"import logging, os; {prelude}import api.main; {_PRINT_LEVELS}"],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


def test_app_imports_with_invalid_log_levels():
    result = _import_main_with_env(LOG_LEVEL="loud", EDGAR_LOG_LEVEL="verbose")
    assert result.returncode == 0, result.stderr
    root_level, edgar_level = map(int, result.stdout.split()[-2:])
    assert (root_level, edgar_level) == (logging.INFO, logging.WARNING)
    assert "LOG_LEVEL='loud'" in result.stderr
    assert "EDGAR_LOG_LEVEL='verbose'" in result.stderr


def test_app_import_applies_valid_log_levels():
    result = _import_main_with_env(LOG_LEVEL="debug", EDGAR_LOG_LEVEL="error")
    assert result.returncode == 0, result.stderr
    root_level, edgar_level = map(int, result.stdout.split()[-2:])
    assert (root_level, edgar_level) == (logging.DEBUG, logging.ERROR)


def test_log_levels_set_in_dotenv_apply(monkeypatch):
    """`.env` is the documented local config; its levels must be loaded before
    logging is configured. The .env file loader is stubbed (the real one reads
    the repo-root .env) to set what a .env would, without overriding the shell."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("EDGAR_LOG_LEVEL", raising=False)
    stub = (
        "import dotenv; "
        "dotenv.load_dotenv = lambda *a, **k: ("
        "os.environ.setdefault('LOG_LEVEL', 'debug'), "
        "os.environ.setdefault('EDGAR_LOG_LEVEL', 'error')); "
    )
    result = _import_main_with_env(prelude=stub)
    assert result.returncode == 0, result.stderr
    root_level, edgar_level = map(int, result.stdout.split()[-2:])
    assert (root_level, edgar_level) == (logging.DEBUG, logging.ERROR)

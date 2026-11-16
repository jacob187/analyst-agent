"""Filings endpoints pick a model the caller can actually run.

A hosted deploy whose operator holds a key only for a provider other than the
registry default's must still generate analyses for signed-in users on a
lendable model of that provider. The providers are picked from the registry. Real temp SQLite, the real SECDataRetrieval over
an in-memory EDGAR (tests/sec_fakes.py) and a fake chat model at the
`create_llm` boundary that records which model id each analysis ran on.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import api.db as db_module
from agents.model_registry import get_all_models, get_model
from api.dependencies import PROVIDER_ENV_VARS
from api.main import app
from tests.conftest import byok_header, clear_operator_keys, flagged_default_model, models_where
from tests.sec_fakes import CountingChatModel, FakeEdgar, eightk, tenk, tenq
from tests.test_dependencies import (  # noqa: F401  (clerk_env / rsa_keypair are fixtures)
    _make_token,
    clerk_env,
    rsa_keypair,
)

TICKER = "FBCK"
PATH = f"/api/company/{TICKER}/filings"
STREAM_PATH = f"{PATH}/stream"
USER = "user_modelfallback"
TENK_RISK = "TENK-RISK: supply chain concentration and antitrust exposure."
TENQ_RISK = "There have been no material changes to the risk factors previously disclosed."


def _operator_provider():
    """A non-default provider with both lendable and non-lendable models."""
    default_provider = flagged_default_model().provider
    for provider in sorted({m.provider for m in get_all_models()} - {default_provider}):
        lendable = {m.operator_lendable for m in models_where(provider=provider)}
        if lendable == {True, False}:
            return provider
    pytest.skip(
        "no provider besides the default's has both lendable and non-lendable models",
        allow_module_level=True,
    )


# The only provider the operator holds a key for, and another one a caller brings.
OPERATOR_PROVIDER = _operator_provider()
OPERATOR_ENV = PROVIDER_ENV_VARS[OPERATOR_PROVIDER]
BYOK_PROVIDER = next(p for p in PROVIDER_ENV_VARS if p != OPERATOR_PROVIDER)


@pytest.fixture
def filings_env(tmp_path, monkeypatch):
    """Temp DB + fake EDGAR + a recording fake LLM. Yields the list of
    `(model_id, api_key)` pairs `create_llm` was called with."""
    clear_operator_keys(monkeypatch)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("FILING_FRESHNESS_TTL_SECONDS", raising=False)

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "filings.db")
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.run(db_module.init_db())

    edgar = FakeEdgar()
    edgar.publish(tenk("0000320193-25-000101", "2025-11-01", TENK_RISK))
    edgar.publish(tenq("0000320193-26-000111", "2026-02-01", TENQ_RISK, mda="TENQ-MDA: services grew."))
    edgar.publish(eightk("0000320193-26-000121", "2026-01-30", earnings=True))
    edgar.install(monkeypatch)

    used: list[tuple[str, str | None]] = []
    llm = CountingChatModel()

    def fake_create_llm(model_id, api_key, *args, **kwargs):
        used.append((model_id, api_key))
        return llm

    monkeypatch.setattr("agents.llm_factory.create_llm", fake_create_llm)
    yield used
    asyncio.run(db_module.close_db())


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def signed_in(clerk_env, rsa_keypair):
    priv, _ = rsa_keypair
    return {"X-Clerk-Session-Token": _make_token(priv, sub=USER)}


def _usage(user_id: str) -> int:
    return asyncio.run(db_module.get_llm_usage(user_id))


def _stream(client, headers) -> list[dict]:
    events = []
    with client.stream("GET", STREAM_PATH, headers=headers) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _section_statuses(events) -> set[str]:
    return {
        e["status"] for e in events
        if e.get("type") == "progress" and e["step"] != "edgar_fetch"
    }


def _operator_model(*, lendable: bool):
    return models_where(provider=OPERATOR_PROVIDER, operator_lendable=lendable)[0]


# ── AC1: no explicit model → a lendable model the operator key can run ───────


def test_signed_in_gets_a_lendable_model_of_the_only_configured_provider(
    client, filings_env, signed_in, monkeypatch,
):
    monkeypatch.setenv(OPERATOR_ENV, "env-key")

    resp = client.get(PATH, headers=signed_in)

    assert resp.status_code == 200
    body = resp.json()
    assert "risk_10k" in body["tenk"] and "risk_10q" in body["tenq"]  # generated, not needs_key
    assert filings_env, "no analysis ran"
    for model_id, api_key in filings_env:
        model = get_model(model_id)
        assert model.provider == OPERATOR_PROVIDER
        assert model.operator_lendable is True
        assert api_key == "env-key"
    # The operator paid: one budget unit per generated section.
    assert _usage(USER) == len(filings_env)


def test_stream_gets_a_lendable_model_of_the_only_configured_provider(
    client, filings_env, signed_in, monkeypatch,
):
    monkeypatch.setenv(OPERATOR_ENV, "env-key")

    events = _stream(client, signed_in)

    statuses = _section_statuses(events)
    assert "done" in statuses
    assert "needs_key" not in statuses
    assert filings_env
    assert {get_model(m).provider for m, _ in filings_env} == {OPERATOR_PROVIDER}
    assert all(get_model(m).operator_lendable for m, _ in filings_env)
    assert _usage(USER) == len(filings_env)


def test_env_default_model_id_does_not_count_as_an_explicit_request(
    client, filings_env, signed_in, monkeypatch,
):
    # DEFAULT_MODEL_ID names a model the operator has no key for; with no
    # X-Model-Id the caller still lands on a model they can run.
    monkeypatch.setenv(OPERATOR_ENV, "env-key")
    monkeypatch.setenv("DEFAULT_MODEL_ID", flagged_default_model().id)

    assert client.get(PATH, headers=signed_in).status_code == 200
    assert filings_env
    assert {get_model(m).provider for m, _ in filings_env} == {OPERATOR_PROVIDER}


def test_byok_for_another_provider_is_used_without_an_explicit_model(
    client, filings_env, signed_in,
):
    resp = client.get(PATH, headers={**signed_in, **byok_header(BYOK_PROVIDER, "their-key")})

    assert resp.status_code == 200
    assert filings_env
    assert {get_model(m).provider for m, _ in filings_env} == {BYOK_PROVIDER}
    assert {k for _, k in filings_env} == {"their-key"}
    assert _usage(USER) == 0  # BYOK: the operator doesn't pay


# ── AC2: explicit request for a model the operator won't lend ────────────────


def test_explicit_non_lendable_model_without_byok_needs_key(
    client, filings_env, signed_in, monkeypatch,
):
    monkeypatch.setenv(OPERATOR_ENV, "env-key")
    headers = {**signed_in, "X-Model-Id": _operator_model(lendable=False).id}

    resp = client.get(PATH, headers=headers)
    assert resp.status_code == 200
    assert set(resp.json()["tenk"]) == {"metadata"}

    assert _section_statuses(_stream(client, headers)) == {"needs_key"}
    assert filings_env == []
    assert _usage(USER) == 0


def test_explicit_lendable_model_is_honoured(client, filings_env, signed_in, monkeypatch):
    monkeypatch.setenv(OPERATOR_ENV, "env-key")
    # Not the first lendable one, so the explicit choice is what's observed.
    asked = models_where(provider=OPERATOR_PROVIDER, operator_lendable=True)[-1]

    assert client.get(PATH, headers={**signed_in, "X-Model-Id": asked.id}).status_code == 200
    assert {m for m, _ in filings_env} == {asked.id}


# ── AC3: anonymous callers never spend the operator key ──────────────────────


def test_anonymous_never_generates_on_the_operator_key(client, filings_env, clerk_env, monkeypatch):
    monkeypatch.setenv(OPERATOR_ENV, "env-key")

    resp = client.get(PATH)
    assert resp.status_code == 200
    assert set(resp.json()["tenk"]) == {"metadata"}

    assert _section_statuses(_stream(client, {})) == {"needs_key"}
    assert filings_env == []


def test_anonymous_with_no_keys_anywhere_needs_key(client, filings_env, clerk_env):
    assert _section_statuses(_stream(client, {})) == {"needs_key"}
    assert filings_env == []


def test_anonymous_byok_generates_on_their_own_key_only(client, filings_env, clerk_env, monkeypatch):
    monkeypatch.setenv(OPERATOR_ENV, "env-key")

    resp = client.get(PATH, headers=byok_header(BYOK_PROVIDER, "their-key"))

    assert resp.status_code == 200
    assert filings_env
    assert {k for _, k in filings_env} == {"their-key"}

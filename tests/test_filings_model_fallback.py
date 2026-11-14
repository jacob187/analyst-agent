"""Filings endpoints pick a model the caller can actually run.

A hosted deploy whose operator holds only a Google key must still generate
analyses for signed-in users on a lendable Google model, even though the
registry default is Anthropic. Real temp SQLite, the real SECDataRetrieval over
an in-memory EDGAR (tests/sec_fakes.py) and a fake chat model at the
`create_llm` boundary that records which model id each analysis ran on.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import api.db as db_module
from agents.model_registry import get_all_models, get_default_model, get_model
from agents.sec_workflow import analysis_cache
from api.main import app
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


@pytest.fixture
def filings_env(tmp_path, monkeypatch):
    """Temp DB + fake EDGAR + a recording fake LLM. Yields the list of
    `(model_id, api_key)` pairs `create_llm` was called with."""
    for var in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEFAULT_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FILING_FRESHNESS_TTL_SECONDS", raising=False)

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "filings.db")
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.run(db_module.init_db())

    for cache in (
        analysis_cache._shared_retrievers,
        analysis_cache._latest_filings,
        analysis_cache._latest_locks,
        analysis_cache._analysis_locks,
    ):
        cache.clear()
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


def _google_model(*, lendable: bool):
    return next(
        m for m in get_all_models()
        if m.provider == "google_genai" and m.operator_lendable is lendable
    )


def test_registry_default_is_not_runnable_on_a_google_only_operator():
    # The premise of the fallback: the default's provider isn't the one the
    # operator has a key for.
    assert get_default_model().provider != "google_genai"


# ── AC1: no explicit model → a lendable model the operator key can run ───────


def test_signed_in_gets_a_lendable_google_model_when_only_google_is_configured(
    client, filings_env, signed_in, monkeypatch,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")

    resp = client.get(PATH, headers=signed_in)

    assert resp.status_code == 200
    body = resp.json()
    assert "risk_10k" in body["tenk"] and "risk_10q" in body["tenq"]  # generated, not needs_key
    assert filings_env, "no analysis ran"
    for model_id, api_key in filings_env:
        model = get_model(model_id)
        assert model.provider == "google_genai"
        assert model.operator_lendable is True
        assert api_key == "env-google"
    # The operator paid: one budget unit per generated section.
    assert _usage(USER) == len(filings_env)


def test_stream_gets_a_lendable_google_model_when_only_google_is_configured(
    client, filings_env, signed_in, monkeypatch,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")

    events = _stream(client, signed_in)

    statuses = _section_statuses(events)
    assert "done" in statuses
    assert "needs_key" not in statuses
    assert filings_env
    assert {get_model(m).provider for m, _ in filings_env} == {"google_genai"}
    assert all(get_model(m).operator_lendable for m, _ in filings_env)
    assert _usage(USER) == len(filings_env)


def test_env_default_model_id_does_not_count_as_an_explicit_request(
    client, filings_env, signed_in, monkeypatch,
):
    # DEFAULT_MODEL_ID names a model the operator has no key for; with no
    # X-Model-Id the caller still lands on a model they can run.
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")
    monkeypatch.setenv("DEFAULT_MODEL_ID", get_default_model().id)

    assert client.get(PATH, headers=signed_in).status_code == 200
    assert filings_env
    assert {get_model(m).provider for m, _ in filings_env} == {"google_genai"}


def test_byok_for_another_provider_is_used_without_an_explicit_model(
    client, filings_env, signed_in,
):
    resp = client.get(PATH, headers={**signed_in, "X-Openai-Api-Key": "their-openai"})

    assert resp.status_code == 200
    assert filings_env
    assert {get_model(m).provider for m, _ in filings_env} == {"openai"}
    assert {k for _, k in filings_env} == {"their-openai"}
    assert _usage(USER) == 0  # BYOK: the operator doesn't pay


# ── AC2: explicit request for a model the operator won't lend ────────────────


def test_explicit_non_lendable_model_without_byok_needs_key(
    client, filings_env, signed_in, monkeypatch,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")
    headers = {**signed_in, "X-Model-Id": _google_model(lendable=False).id}

    resp = client.get(PATH, headers=headers)
    assert resp.status_code == 200
    assert set(resp.json()["tenk"]) == {"metadata"}

    assert _section_statuses(_stream(client, headers)) == {"needs_key"}
    assert filings_env == []
    assert _usage(USER) == 0


def test_explicit_lendable_model_is_honoured(client, filings_env, signed_in, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")
    # Not the first lendable one, so the explicit choice is what's observed.
    asked = [m for m in get_all_models() if m.provider == "google_genai" and m.operator_lendable][-1]

    assert client.get(PATH, headers={**signed_in, "X-Model-Id": asked.id}).status_code == 200
    assert {m for m, _ in filings_env} == {asked.id}


# ── AC3: anonymous callers never spend the operator key ──────────────────────


def test_anonymous_never_generates_on_the_operator_key(client, filings_env, clerk_env, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")

    resp = client.get(PATH)
    assert resp.status_code == 200
    assert set(resp.json()["tenk"]) == {"metadata"}

    assert _section_statuses(_stream(client, {})) == {"needs_key"}
    assert filings_env == []


def test_anonymous_with_no_keys_anywhere_needs_key(client, filings_env, clerk_env):
    assert _section_statuses(_stream(client, {})) == {"needs_key"}
    assert filings_env == []


def test_anonymous_byok_generates_on_their_own_key_only(client, filings_env, clerk_env, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google")

    resp = client.get(PATH, headers={"X-Openai-Api-Key": "their-openai"})

    assert resp.status_code == 200
    assert filings_env
    assert {k for _, k in filings_env} == {"their-openai"}

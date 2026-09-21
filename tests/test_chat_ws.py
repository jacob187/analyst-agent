"""WebSocket-hardening tests for /ws/chat/{ticker}.

Verifies:
- Oversized auth frames close with policy_violation (1008).
- Auth frames missing within the timeout close cleanly.
- Malformed JSON auth payloads close with 1008.
- `_safe_send` closes the socket when send_json hangs past the send timeout.

The route's full happy path is covered elsewhere (those tests require a live
LLM and Tavily). The hardening tests here only need to reach the rejection
paths, which run before any agent setup.
"""

import asyncio
import json
import sqlite3
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.main import app
from api.routes import chat


@pytest.fixture
def client():
    return TestClient(app)


class TestAuthFrameSizeCap:
    def test_oversized_auth_closed_with_1008(self, client, monkeypatch):
        # Cap at 1 KB so the test payload doesn't have to be huge.
        monkeypatch.setattr(chat, "WS_AUTH_MAX_BYTES", 1024)

        oversized = '{"type":"auth","junk":"' + "x" * 4096 + '"}'

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(oversized)
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008

    def test_normal_sized_auth_passes_size_check(self, client, monkeypatch):
        # With a generous cap and an invalid (but small) auth payload, the
        # connection should fail past the size check on a different reason
        # (missing type) — proving the size gate didn't fire.
        monkeypatch.setattr(chat, "WS_AUTH_MAX_BYTES", 16384)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text('{"type":"not_auth"}')
            err = ws.receive_json()
            assert err["type"] == "error"
            # Server closes; the next receive raises.
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


class TestAuthFrameTimeout:
    def test_no_auth_frame_within_timeout_closes(self, client, monkeypatch):
        monkeypatch.setattr(chat, "WS_AUTH_TIMEOUT_SECONDS", 0.3)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            # Send nothing. Server should close after ~0.3s.
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


class TestInvalidAuthJson:
    def test_invalid_json_closes_with_1008(self, client, monkeypatch):
        monkeypatch.setattr(chat, "WS_AUTH_MAX_BYTES", 16384)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text('{"type":"auth", invalid json')
            err = ws.receive_json()
            assert err["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008

    def test_non_object_auth_closes_with_1008(self, client, monkeypatch):
        monkeypatch.setattr(chat, "WS_AUTH_MAX_BYTES", 16384)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text('["not", "an", "object"]')
            err = ws.receive_json()
            assert err["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


class TestSafeSendTimeout:
    @pytest.mark.asyncio
    async def test_safe_send_closes_on_send_timeout(self, monkeypatch):
        """_safe_send must close the socket if send_json hangs past the timeout."""
        monkeypatch.setattr(chat, "WS_SEND_TIMEOUT_SECONDS", 0.1)

        sends_seen: list[dict] = []
        close_calls: list[int] = []

        class HangingWebSocket:
            async def send_json(self, data):
                sends_seen.append(data)
                # Hang past the send timeout
                await asyncio.sleep(1.0)

            async def close(self, code: int = 1000):
                close_calls.append(code)

        ws = HangingWebSocket()
        ok = await chat._safe_send(ws, {"type": "system", "message": "hello"})

        assert ok is False
        assert sends_seen == [{"type": "system", "message": "hello"}]
        assert close_calls == [1008]

    @pytest.mark.asyncio
    async def test_safe_send_returns_true_on_fast_send(self, monkeypatch):
        monkeypatch.setattr(chat, "WS_SEND_TIMEOUT_SECONDS", 1.0)

        class FastWebSocket:
            def __init__(self):
                self.sent: list[dict] = []

            async def send_json(self, data):
                self.sent.append(data)

            async def close(self, code: int = 1000):
                pass

        ws = FastWebSocket()
        ok = await chat._safe_send(ws, {"type": "ok"})
        assert ok is True
        assert ws.sent == [{"type": "ok"}]


# ── Anonymous chat persists nothing ────────────────────────────────────────


class _FakeAgent:
    """Stand-in for the LangGraph agent: streams a single response event."""

    def stream(self, *args, **kwargs):
        async def _gen():
            yield {"type": "response", "message": "Analysis complete."}
        return _gen()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point api.db at a throwaway SQLite file with the schema initialised, so a
    full WS exchange can run and the test can count persisted rows directly."""
    import api.db as db_module
    db_path = tmp_path / "chat.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.get_event_loop().run_until_complete(db_module.init_db())
    yield db_path
    asyncio.get_event_loop().run_until_complete(db_module.close_db())


def _patch_agent(monkeypatch):
    """Replace LLM + agent construction so no real model is created."""
    monkeypatch.setattr(chat, "create_llm_pair", lambda *a, **k: (MagicMock(), MagicMock()))
    monkeypatch.setattr(
        "agents.graph.analyst_graph.create_sec_qa_agent",
        lambda *a, **k: _FakeAgent(),
    )


def _drain_until(ws, target_type):
    while True:
        ev = ws.receive_json()
        if ev["type"] == target_type:
            return ev


def _row_count(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# BYOK keys for every provider so the default model's provider key always
# resolves; no user_id → anonymous.
_BYOK_AUTH = {
    "type": "auth",
    "google_api_key": "byok-key",
    "openai_api_key": "byok-key",
    "anthropic_api_key": "byok-key",
}


class TestAnonChatNoPersistence:
    """Anonymous (no user_id) chat with a BYOK key streams a response but writes
    zero rows — persistence is sign-in-only."""

    def test_anon_chat_streams_but_persists_nothing(self, client, temp_db, monkeypatch):
        _patch_agent(monkeypatch)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps(_BYOK_AUTH))  # no user_id → anonymous
            ok = _drain_until(ws, "auth_success")
            assert ok["session_id"] is None
            assert ok["resumed"] is False

            ws.send_text(json.dumps({"type": "query", "message": "hi"}))
            resp = _drain_until(ws, "response")
            assert resp["message"] == "Analysis complete."

        assert _row_count(temp_db, "sessions") == 0
        assert _row_count(temp_db, "messages") == 0

    def test_signed_in_chat_persists_session(self, client, temp_db, monkeypatch):
        # Contrast: a Clerk-format id (Clerk disabled in tests → trusted) writes a
        # session row, proving the test would catch an anon-persistence regression.
        # Assert on the session row (awaited), not messages (fire-and-forget task).
        _patch_agent(monkeypatch)
        auth = {**_BYOK_AUTH, "user_id": "user_contrasttest"}

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps(auth))
            ok = _drain_until(ws, "auth_success")
            assert ok["session_id"]  # a real session id

            ws.send_text(json.dumps({"type": "query", "message": "hi"}))
            _drain_until(ws, "response")

        assert _row_count(temp_db, "sessions") == 1


# ── Anonymous free trial ─────────────────────────────────────────────────────


class TestAnonFreeTrial:
    """Anonymous visitors with no BYOK key may still chat using the
    designated free-trial model, capped per-IP at ANON_FREE_QUERIES/day."""

    @pytest.fixture(autouse=True)
    def _isolate_quota(self, monkeypatch):
        from api.rate_limit import _rest_timestamps
        # Clerk enabled (no token) → ordinary anon env-key resolution is
        # blocked, so success here can only come from the free-trial carve-out.
        monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
        _rest_timestamps.clear()
        yield
        _rest_timestamps.clear()

    def test_anon_no_key_uses_free_trial_model(self, client, temp_db, monkeypatch):
        _patch_agent(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps({"type": "auth"}))  # no keys, no user_id
            ok = _drain_until(ws, "auth_success")
            assert ok["free_trial"] is True
            assert ok["free_trial_queries"] == chat.ANON_FREE_QUERIES

            ws.send_text(json.dumps({"type": "query", "message": "hi"}))
            resp = _drain_until(ws, "response")
            assert resp["message"] == "Analysis complete."

    def test_anon_free_trial_quota_exhausted(self, client, temp_db, monkeypatch):
        _patch_agent(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")
        monkeypatch.setattr(chat, "ANON_FREE_QUERIES", 1)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps({"type": "auth"}))
            _drain_until(ws, "auth_success")

            ws.send_text(json.dumps({"type": "query", "message": "hi"}))
            _drain_until(ws, "response")

            ws.send_text(json.dumps({"type": "query", "message": "hi again"}))
            err = _drain_until(ws, "error")
            assert "Free trial limit reached" in err["message"]

    def test_anon_without_operator_key_still_requires_byok(self, client, temp_db, monkeypatch):
        # No GOOGLE_API_KEY configured → no free trial to fall back to.
        _patch_agent(monkeypatch)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps({"type": "auth"}))
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "API key required" in err["message"]

    def test_anon_requesting_non_trial_model_still_requires_byok(self, client, temp_db, monkeypatch):
        # The free trial only covers the one designated light model — picking
        # a different model without a key is still refused.
        _patch_agent(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-anthropic-key")

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps({"type": "auth", "model_id": "claude-opus-5"}))
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "API key required" in err["message"]

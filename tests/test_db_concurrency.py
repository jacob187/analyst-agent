"""Concurrent use of the shared aiosqlite connection in api/db.py.

Marker: eval_unit (no API keys, no network). Real SQLite in a temp dir.

api.db shares ONE connection across every coroutine. SQLite refuses a COMMIT
while another statement on that connection is still mid-flight — e.g. the
`INSERT ... RETURNING` in increment_llm_usage between its execute and its
fetch — with "cannot commit transaction - SQL statements in progress".
"""

import asyncio
import gc
import logging
import sqlite3
import time
from contextlib import closing

import pytest

import api.db as db_module

USER = "user_concurrency"


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "concurrency.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield tmp_path / "concurrency.db"
    await db_module.close_db()


async def _long_raw_read(session_id: str) -> int:
    """A reader that keeps a cursor open across awaits between fetches."""
    db = await db_module.get_db()
    seen = 0
    async with db.execute(
        "SELECT id FROM messages WHERE session_id = ? UNION ALL SELECT id FROM sessions",
        (session_id,),
    ) as cur:
        while await cur.fetchone() is not None:
            seen += 1
            await asyncio.sleep(0)
    return seen


@pytest.mark.eval_unit
class TestConcurrentWritesOnSharedConnection:
    async def test_interleaved_reads_and_writes_never_fail_a_commit(self, fresh_db):
        session_id = await db_module.create_session("AAPL", USER)
        n_messages, n_charges, n_reads = 60, 60, 30

        coros = []
        for i in range(max(n_messages, n_charges, n_reads)):
            if i < n_messages:
                coros.append(db_module.save_message(session_id, "user", f"m{i}"))
            if i < n_charges:
                coros.append(db_module.increment_llm_usage(USER))
            if i < n_reads:
                coros.append(_long_raw_read(session_id))
                coros.append(db_module.get_session_messages(session_id))
                coros.append(db_module.get_llm_usage(USER))

        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True), timeout=20
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"{len(errors)} DB ops failed; first: {errors[0]!r}"

        assert len(await db_module.get_session_messages(session_id)) == n_messages
        assert await db_module.get_llm_usage(USER) == n_charges


@pytest.mark.eval_unit
class TestFailedWriteLeavesNoPartialRows:
    async def test_failed_multi_statement_write_is_rolled_back(self, fresh_db):
        """create_session inserts companies + user_companies before sessions. If
        the sessions insert fails, the earlier rows must not ride along on the
        next operation's commit."""
        db = await db_module.get_db()
        await db.execute("DROP TABLE sessions")
        await db.commit()

        with pytest.raises(Exception):
            await db_module.create_session("ORPH", USER)
        await db_module.ensure_company("OTHER")  # commits

        assert await db_module.get_company("ORPH") is None
        assert await db_module.get_company("OTHER") is not None


def _contended_ops(session_id: str) -> list:
    return [
        op
        for i in range(20)
        for op in (
            db_module.save_message(session_id, "user", f"m{i}"),
            db_module.increment_llm_usage(USER),
            db_module.get_llm_usage(USER),
        )
    ]


@pytest.mark.eval_unit
class TestLockAcrossEventLoops:
    def test_contended_ops_work_on_consecutive_loops(self, tmp_path, monkeypatch):
        """The benchmark and the test suite open/close the DB on fresh loops; a
        lock bound to an earlier loop must not break contended use on a new one."""
        monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "loops.db")
        monkeypatch.setattr(db_module, "_db", None)

        async def run_once() -> tuple[int, int]:
            await db_module.init_db()
            session_id = await db_module.get_or_create_session("AAPL", USER)
            await asyncio.wait_for(asyncio.gather(*_contended_ops(session_id)), timeout=20)
            counts = (
                len(await db_module.get_session_messages(session_id)),
                await db_module.get_llm_usage(USER),
            )
            await db_module.close_db()
            return counts

        assert asyncio.run(run_once()) == (20, 20)
        assert asyncio.run(run_once()) == (40, 40)

    def test_connection_kept_open_across_loops(self, tmp_path, monkeypatch):
        """Same connection, no close in between: the TestClient-based tests
        init the DB on one loop and serve requests on another."""
        monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "loops2.db")
        monkeypatch.setattr(db_module, "_db", None)

        async def setup() -> str:
            await db_module.init_db()
            return await db_module.get_or_create_session("AAPL", USER)

        async def contend(session_id: str) -> None:
            await asyncio.wait_for(asyncio.gather(*_contended_ops(session_id)), timeout=20)

        session_id = asyncio.run(setup())
        asyncio.run(contend(session_id))
        asyncio.run(contend(session_id))

        async def finish() -> int:
            usage = await db_module.get_llm_usage(USER)
            await db_module.close_db()
            return usage

        assert asyncio.run(finish()) == 40


@pytest.mark.eval_unit
class TestSpawnWrite:
    async def test_failed_background_write_is_logged_at_error(self, fresh_db, caplog):
        db = await db_module.get_db()
        await db.execute("DROP TABLE messages")
        await db.commit()

        caplog.set_level(logging.ERROR)
        task = db_module.spawn_write(
            db_module.save_message("sess-123", "user", "secret content"),
            op="save_message", session_id="sess-123",
        )
        await asyncio.wait([task])
        await asyncio.sleep(0)  # let the done-callback run
        del task
        gc.collect()

        errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == "api.db"]
        assert len(errors) == 1
        assert "sess-123" in errors[0].getMessage()
        assert "save_message" in errors[0].getMessage()
        assert "secret content" not in errors[0].getMessage()
        assert not any("never retrieved" in r.getMessage() for r in caplog.records)
        assert db_module._background_writes == set()

    async def test_successful_background_write_logs_nothing(self, fresh_db, caplog):
        session_id = await db_module.create_session("AAPL", USER)
        caplog.set_level(logging.WARNING)
        task = db_module.spawn_write(
            db_module.save_message(session_id, "user", "hi"),
            op="save_message", session_id=session_id,
        )
        await task
        await asyncio.sleep(0)
        assert caplog.records == []
        assert len(await db_module.get_session_messages(session_id)) == 1


def _stored_messages(db_path, session_id: str) -> list[tuple[str]]:
    """Read straight from the file, so the check can't reopen api.db's connection."""
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", (session_id,)
        ).fetchall()


@pytest.mark.eval_unit
class TestCloseDbDrainsBackgroundWrites:
    async def test_pending_write_lands_before_close(self, fresh_db):
        session_id = await db_module.create_session("AAPL", USER)
        task = db_module.spawn_write(
            db_module.save_message(session_id, "assistant", "last words"),
            op="save_message", session_id=session_id,
        )

        await db_module.close_db()

        assert task.done() and not task.cancelled()
        assert db_module._db is None
        assert _stored_messages(fresh_db, session_id) == [("last words",)]

    async def test_write_past_timeout_does_not_block_close_and_is_logged(
        self, fresh_db, monkeypatch, caplog
    ):
        monkeypatch.setenv("DB_CLOSE_DRAIN_TIMEOUT_SECONDS", "0.2")
        session_id = await db_module.create_session("AAPL", USER)

        async def stalled_write() -> None:
            # Holds the connection lock, as a write stuck mid-operation would.
            async with db_module._conn() as db:
                await asyncio.sleep(30)
                await db.execute(
                    "INSERT INTO messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
                    ("m-stalled", session_id, "assistant", "never"),
                )
                await db.commit()

        task = db_module.spawn_write(stalled_write(), op="stalled_write", session_id=session_id)
        await asyncio.sleep(0)  # let it take the lock
        caplog.set_level(logging.WARNING, logger="api.db")

        start = time.monotonic()
        await asyncio.wait_for(db_module.close_db(), timeout=5)

        assert time.monotonic() - start < 2
        assert task.cancelled()
        assert db_module._db is None
        assert db_module._background_writes == set()
        unfinished = [r.getMessage() for r in caplog.records if r.name == "api.db"]
        assert any("stalled_write" in m and session_id in m for m in unfinished), unfinished
        assert _stored_messages(fresh_db, session_id) == []

    @pytest.mark.parametrize("raw", ["soon", "-1"])
    def test_invalid_drain_timeout_falls_back_to_default(self, monkeypatch, caplog, raw):
        monkeypatch.setenv("DB_CLOSE_DRAIN_TIMEOUT_SECONDS", raw)
        caplog.set_level(logging.WARNING, logger="api.db")
        assert db_module._drain_timeout_seconds() == db_module.DEFAULT_DRAIN_TIMEOUT_SECONDS
        assert any("DB_CLOSE_DRAIN_TIMEOUT_SECONDS" in r.getMessage() for r in caplog.records)

    def test_write_bound_to_another_loop_is_skipped(self, tmp_path, monkeypatch):
        """conftest closes the DB from a fresh loop after TestClient requests ran
        on others; their leftover writes must not be awaited from the wrong loop."""
        monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "other_loop.db")
        monkeypatch.setattr(db_module, "_db", None)
        # Long enough that waiting on the foreign task would trip wait_for below.
        monkeypatch.setenv("DB_CLOSE_DRAIN_TIMEOUT_SECONDS", "30")

        other = asyncio.new_event_loop()

        async def spawn_on_other() -> asyncio.Task:
            return db_module.spawn_write(
                asyncio.sleep(3600), op="other_loop_write", session_id="s-other"
            )

        task = other.run_until_complete(spawn_on_other())
        try:
            async def init_and_close() -> None:
                await db_module.init_db()
                await asyncio.wait_for(db_module.close_db(), timeout=5)

            asyncio.run(init_and_close())

            assert db_module._db is None
            assert task not in db_module._background_writes
            assert not task.done()
        finally:
            task.cancel()
            other.run_until_complete(asyncio.gather(task, return_exceptions=True))
            other.close()

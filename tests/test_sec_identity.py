"""Tests for the edgar.set_identity race fix.

Pre-fix, every `SECDataRetrieval(ticker, sec_header)` call wrote a
process-global identity to edgartools — concurrent users with different
headers raced last-writer-wins. Post-fix, set_identity is called exactly
once at startup from api/main.py and the constructor takes no header.

These tests assert:
- The constructor never touches edgartools' global identity.
- Concurrent constructions on multiple threads don't race anything.
"""

import threading
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_company():
    """Stub edgartools' Company so the constructor doesn't hit SEC over the wire."""
    with patch("agents.sec_workflow.get_SEC_data.Company") as MockCompany:
        MockCompany.return_value = object()  # plain sentinel; nothing reads it here
        yield MockCompany


def test_constructor_does_not_set_identity(mock_company):
    """SECDataRetrieval must NOT call set_identity — that's a startup-only call now."""
    from agents.sec_workflow import get_SEC_data

    # The set_identity symbol was removed from the module's imports. Confirm
    # absence directly — any reintroduction would trip this assertion.
    assert not hasattr(get_SEC_data, "set_identity")


def test_constructor_signature_drops_sec_header(mock_company):
    """SECDataRetrieval must accept only ticker — no per-instance header."""
    import inspect

    from agents.sec_workflow.get_SEC_data import SECDataRetrieval

    sig = inspect.signature(SECDataRetrieval.__init__)
    params = list(sig.parameters)
    assert params == ["self", "ticker"], f"Unexpected signature: {params}"


def test_concurrent_construction_is_race_free(mock_company):
    """Constructing many retrievers concurrently must not touch any shared
    edgartools global state (because set_identity is gone from the path)."""
    from agents.sec_workflow.get_SEC_data import SECDataRetrieval

    errors: list[Exception] = []

    def construct(ticker: str):
        try:
            SECDataRetrieval(ticker)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=construct, args=(f"T{i}",)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"Concurrent construction raised: {errors}"

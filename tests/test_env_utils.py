"""agents.env_utils: numeric env settings fall back (with one warning) on bad values."""

import logging

import pytest

from agents import env_utils
from agents.env_utils import env_float, env_int

_NAME = "ANALYST_TEST_ENV_NUMBER"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(_NAME, raising=False)
    monkeypatch.setattr(env_utils, "_warned", set())


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "agents.env_utils" and _NAME in r.getMessage()]


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_is_the_default_without_warning(monkeypatch, caplog, raw):
    if raw is not None:
        monkeypatch.setenv(_NAME, raw)
    with caplog.at_level(logging.WARNING):
        assert env_float(_NAME, 2.5) == 2.5
        assert env_int(_NAME, 7) == 7
    assert _warnings(caplog) == []


def test_valid_values_parse(monkeypatch):
    monkeypatch.setenv(_NAME, " 3 ")
    assert env_float(_NAME, 1.0) == 3.0
    assert env_int(_NAME, 1) == 3


@pytest.mark.parametrize("raw", ["soon", "8k", "nan", "inf", "-inf"])
def test_float_rejects_non_numbers_and_non_finite(monkeypatch, caplog, raw):
    monkeypatch.setenv(_NAME, raw)
    with caplog.at_level(logging.WARNING):
        assert env_float(_NAME, 5.0) == 5.0
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize("raw", ["8k", "1.5", "nan"])
def test_int_rejects_non_integers(monkeypatch, caplog, raw):
    monkeypatch.setenv(_NAME, raw)
    with caplog.at_level(logging.WARNING):
        assert env_int(_NAME, 9) == 9
    assert len(_warnings(caplog)) == 1


def test_minimum_is_inclusive(monkeypatch, caplog):
    monkeypatch.setenv(_NAME, "0")
    assert env_float(_NAME, 5.0, minimum=0) == 0.0
    monkeypatch.setenv(_NAME, "-1")
    with caplog.at_level(logging.WARNING):
        assert env_float(_NAME, 5.0, minimum=0) == 5.0
        assert env_int(_NAME, 4, minimum=0) == 4
    assert _warnings(caplog)


def test_warns_once_per_value(monkeypatch, caplog):
    monkeypatch.setenv(_NAME, "soon")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            env_float(_NAME, 5.0)
        monkeypatch.setenv(_NAME, "later")
        env_float(_NAME, 5.0)
    assert [r.getMessage() for r in _warnings(caplog)] == [
        f"Ignoring invalid {_NAME}='soon'; using 5.0",
        f"Ignoring invalid {_NAME}='later'; using 5.0",
    ]

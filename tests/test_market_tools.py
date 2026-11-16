"""Market tools are zero-argument: the tool function itself is the callable."""

import pytest

from agents.tools import market_tools

_TOOL_FUNCS = {
    "get_market_overview": market_tools._tool_market_overview,
    "get_macro_indicators": market_tools._tool_macro_indicators,
}


@pytest.fixture
def tools():
    return {t.name: t for t in market_tools.create_market_tools()}


def test_tools_wrap_their_functions_directly(tools):
    assert set(tools) == set(_TOOL_FUNCS)
    for name, func in _TOOL_FUNCS.items():
        assert tools[name].func is func


@pytest.mark.parametrize("name", list(_TOOL_FUNCS))
def test_tool_runs_and_ignores_supplied_arguments(tools, monkeypatch, name):
    monkeypatch.setattr(market_tools, _TOOL_FUNCS[name].__name__, lambda: "fixture-output")
    rebuilt = {t.name: t for t in market_tools.create_market_tools()}

    assert rebuilt[name].invoke({"query": "ignored"}) == "fixture-output"

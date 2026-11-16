"""`DaysArgs.days` accepts whatever a model sends and always yields a usable window.

The value feeds ``datetime('now', '-N days')`` in the briefing query, which
returns NULL (so: no rows) for negative or absurd N, and an exception out of
the validator aborts the whole ReAct turn instead of reaching the model.
"""

import json

import pytest

from agents.tools.tool_args import DEFAULT_BRIEFING_DAYS, DaysArgs

MAX_DAYS = 365


@pytest.mark.parametrize("sent", [1, 7, 30, MAX_DAYS, "14", 7.9, True])
def test_in_range_values_pass_through_as_ints(sent):
    assert DaysArgs(days=sent).days == int(sent)


@pytest.mark.parametrize(
    "sent, expected",
    [(0, 1), (-7, 1), (-(10**12), 1), (366, MAX_DAYS), (10**12, MAX_DAYS), ("9999", MAX_DAYS)],
)
def test_out_of_range_integers_are_clamped(sent, expected):
    assert DaysArgs(days=sent).days == expected


@pytest.mark.parametrize(
    "sent",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
        "inf",
        "abc",
        "",
        None,
        [],
        {},
        "9" * 5000,
    ],
)
def test_garbage_falls_back_to_the_default_without_raising(sent):
    assert DaysArgs(days=sent).days == DEFAULT_BRIEFING_DAYS


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN", "1e999"])
def test_json_infinity_literals_from_a_model_do_not_raise(literal):
    # json.loads turns these into float inf/nan, which int() rejects with
    # OverflowError/ValueError.
    args = json.loads(f'{{"days": {literal}}}')

    assert DaysArgs(**args).days == DEFAULT_BRIEFING_DAYS


def test_missing_days_uses_the_default():
    assert DaysArgs().days == DEFAULT_BRIEFING_DAYS

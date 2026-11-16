"""Shared argument schemas for the agent's ``StructuredTool``s.

Tools used to be single-input ``Tool``s taking one string, so a model that
emitted a structured call (``{"ticker": "NVDA", "days": 7}``) tripped
"Too many arguments to single-input tool". These schemas describe the real
parameters, ignore anything extra, and keep the graph's ``tool.ainvoke("")``
and bare-string call shapes working (LangChain validates a string against the
first field and hands it to the tool positionally).
"""

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

DEFAULT_BRIEFING_DAYS = 7
MIN_DAYS = 1
MAX_DAYS = 365
DEFAULT_8K_ITEM = "2.02"

# Names a model may use for the free-text input of a search-style tool.
_QUERY_ALIASES = ("topic", "q", "search_query", "question", "input")


def _lenient_str(value: Any) -> str:
    return "" if value is None else str(value)


def _lenient_days(value: Any) -> int:
    """Coerce to a lookback window within MIN_DAYS..MAX_DAYS; never raises.

    A validator exception other than ValueError would escape pydantic and abort
    the agent turn, and SQLite's ``datetime('now', '-N days')`` yields NULL
    (no rows) for negative or absurdly large N.
    """
    try:
        days = int(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_BRIEFING_DAYS
    return max(MIN_DAYS, min(days, MAX_DAYS))


class ToolArgs(BaseModel):
    """Base schema: unknown arguments are dropped instead of rejected."""

    model_config = ConfigDict(extra="ignore")


class NoArgs(ToolArgs):
    """Ticker-bound tool with nothing to choose; any supplied arguments are ignored."""


class QueryArgs(ToolArgs):
    """Search-style tool reading one free-text input."""

    query: Annotated[str, BeforeValidator(_lenient_str)] = Field(
        default="", description="Search query or research topic."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_query_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "query" not in data:
            for alias in _QUERY_ALIASES:
                if alias in data:
                    return {**data, "query": data[alias]}
        return data


class DaysArgs(ToolArgs):
    """Lookback window in days; non-numeric values fall back to the default."""

    days: Annotated[int, BeforeValidator(_lenient_days)] = Field(
        default=DEFAULT_BRIEFING_DAYS, description="Number of days to look back."
    )


class EightKItemArgs(ToolArgs):
    """8-K item number, e.g. '2.02'."""

    item: Annotated[str, BeforeValidator(_lenient_str)] = Field(
        default=DEFAULT_8K_ITEM,
        description="8-K item number, e.g. '2.02' (earnings) or '5.02' (leadership changes).",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_query_alias(cls, data: Any) -> Any:
        # A model that sends the item as a search-style ``query`` must not
        # silently get the default item.
        if isinstance(data, dict) and "item" not in data and "query" in data:
            return {**data, "item": data["query"]}
        return data

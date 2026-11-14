"""Architecture registry: name -> async builder.

A builder is ``async (ticker: str, model_id: str) -> agent`` where ``agent`` has
an async generator ``stream(inputs)`` (``inputs = {"messages": [HumanMessage]}``)
yielding event dicts with a ``type`` of ``node``, ``tool``, ``thinking``,
``token`` or ``response``. The last ``response`` event's ``message`` is the
answer. This is the contract ``PlanningAgent.stream`` already satisfies.

Builders read provider keys from the environment themselves (see
``common.provider_api_key``) and must not print them.

To add an architecture, write a builder here and add it to ``ARCHITECTURES``
(``--graph <name>`` picks from this mapping). Import heavy agent modules inside
the builder so summarize/report/judge don't load the agent stack.
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable

from evals.benchmark.common import BenchmarkError, provider_api_key, resolve_model

Builder = Callable[[str, str], Awaitable[Any]]


async def build_current(ticker: str, model_id: str) -> Any:
    """The production planning agent (router -> planner/executor -> synthesizer)."""
    from agents.graph.analyst_graph import create_planning_agent
    from agents.llm_factory import create_llm_pair

    model = resolve_model(model_id)
    llm, synthesizer_llm = create_llm_pair(model.id, provider_api_key(model))
    return create_planning_agent(
        ticker,
        llm,
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip() or None,
        synthesizer_llm=synthesizer_llm,
    )


async def build_skills(ticker: str, model_id: str) -> Any:
    """One tool-calling agent guided by on-demand skills (``agents/skills/``)."""
    from agents.graph import build_agent
    from agents.llm_factory import create_llm_pair

    model = resolve_model(model_id)
    llm, synthesizer_llm = create_llm_pair(model.id, provider_api_key(model))
    return build_agent(
        "skills",
        ticker,
        llm,
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip() or None,
        synthesizer_llm=synthesizer_llm,
    )


ARCHITECTURES: dict[str, Builder] = {
    "current": build_current,
    "skills": build_skills,
}


def get_builder(name: str) -> Builder:
    try:
        return ARCHITECTURES[name]
    except KeyError:
        raise BenchmarkError(
            f"Unknown architecture {name!r}. Available: {', '.join(sorted(ARCHITECTURES))}"
        ) from None

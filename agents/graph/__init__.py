"""Chat agent architectures and the factory that picks one.

``AGENT_ARCHITECTURE`` selects which agent ``build_agent`` returns:

- ``current`` (default): the router -> planner/workers -> synthesizer graph
  (``analyst_graph.create_planning_agent``).
- ``skills``: one tool-calling agent guided by on-demand skills
  (``skills_agent.create_skills_agent``).

Agent modules are imported inside ``build_agent`` so importing this package
stays cheap.
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, Optional

ARCHITECTURES = ("current", "skills")
DEFAULT_ARCHITECTURE = "current"


def agent_architecture() -> str:
    """The architecture named by ``AGENT_ARCHITECTURE`` (default ``current``).

    Raises ValueError on an unknown name rather than silently running a
    different agent than the one configured.
    """
    name = (os.getenv("AGENT_ARCHITECTURE") or DEFAULT_ARCHITECTURE).strip().lower()
    if name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown AGENT_ARCHITECTURE {name!r}; expected one of {', '.join(ARCHITECTURES)}"
        )
    return name


def build_agent(
    architecture: str,
    ticker: str,
    llm: Any,
    tavily_api_key: Optional[str] = None,
    synthesizer_llm: Any = None,
    user_id: str | None = None,
    search_budget: Optional[Callable[[], Awaitable[bool]]] = None,
) -> Any:
    """Build the chat agent for ``architecture``; arguments as ``create_planning_agent``."""
    if architecture == "current":
        # Looked up on the module at call time so tests that patch
        # ``analyst_graph.create_sec_qa_agent`` still take effect.
        from agents.graph import analyst_graph

        return analyst_graph.create_sec_qa_agent(
            ticker, llm,
            tavily_api_key=tavily_api_key,
            synthesizer_llm=synthesizer_llm,
            user_id=user_id,
            search_budget=search_budget,
        )
    if architecture == "skills":
        from agents.graph.skills_agent import create_skills_agent

        return create_skills_agent(
            ticker, llm,
            tavily_api_key=tavily_api_key,
            synthesizer_llm=synthesizer_llm,
            user_id=user_id,
            search_budget=search_budget,
        )
    raise ValueError(
        f"Unknown agent architecture {architecture!r}; expected one of {', '.join(ARCHITECTURES)}"
    )

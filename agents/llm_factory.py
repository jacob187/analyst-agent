"""LLM factory — creates LangChain chat models from model registry IDs.

Wraps LangChain's `init_chat_model` with a ThinkingConfig abstraction
that normalizes provider-specific thinking/reasoning parameters.

Usage:
    from agents.llm_factory import create_llm, create_llm_pair

    # Single LLM
    llm = create_llm("gemini-3-flash-preview", api_key="AIza...")

    # Tool-calling LLM + synthesizer pair
    llm, synthesizer = create_llm_pair("gpt-4o-mini", api_key="sk-...")
"""

import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from agents.model_registry import get_default_model, get_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider → API key kwarg mapping for init_chat_model
# ---------------------------------------------------------------------------

_PROVIDER_KEY_KWARG: dict[str, str] = {
    "google_genai": "google_api_key",
    "openai": "api_key",
    "anthropic": "anthropic_api_key",
}

# Thinking budget tiers for Anthropic (token counts per level)
_ANTHROPIC_THINKING_BUDGETS: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}


@dataclass(frozen=True)
class ThinkingConfig:
    """Provider-agnostic thinking/reasoning configuration.

    Attributes:
        enabled: Whether to enable thinking/reasoning mode.
        level: Intensity level — maps to provider-specific params.
            Google: thinking_level ("low"/"medium"/"high")
            Anthropic: budget_tokens via _ANTHROPIC_THINKING_BUDGETS
            OpenAI: reasoning_effort ("low"/"medium"/"high")
    """

    enabled: bool = False
    level: str = "medium"


def _build_thinking_kwargs(
    provider: str, model_id: str, config: ThinkingConfig | None
) -> dict:
    """Translate ThinkingConfig into provider-specific kwargs.

    Each provider has a different wire format for thinking/reasoning.
    This function is the single place that knows about those differences.
    """
    if config is None or not config.enabled:
        return {}

    if provider == "google_genai":
        return {"thinking_level": config.level, "include_thoughts": True}

    if provider == "anthropic":
        budget = _ANTHROPIC_THINKING_BUDGETS.get(config.level, 4096)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    if provider == "openai":
        return {"reasoning_effort": config.level}

    return {}


def create_llm(
    model_id: str,
    api_key: str,
    thinking: ThinkingConfig | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model from a registry model ID.

    Uses init_chat_model for provider dispatch — no if/elif chains.
    Falls back to the default model if model_id isn't in the registry.

    Args:
        model_id: Model ID from models.json (e.g. "gemini-3-flash-preview").
        api_key: The API key for the model's provider.
        thinking: Optional thinking/reasoning config. Ignored if the model
            isn't thinking-capable.

    Returns:
        A BaseChatModel instance ready for use with LangGraph.
    """
    from langchain.chat_models import init_chat_model

    model = get_model(model_id)
    if model is None:
        default = get_default_model()
        logger.warning(
            "Model '%s' not found in registry, falling back to '%s'",
            model_id,
            default.id,
        )
        model = default

    # Only apply thinking kwargs if the model supports it
    effective_thinking = thinking if model.thinking_capable else None
    thinking_kwargs = _build_thinking_kwargs(model.provider, model.id, effective_thinking)

    # Resolve the provider-specific API key kwarg name
    key_kwarg_name = _PROVIDER_KEY_KWARG.get(model.provider, "api_key")

    return init_chat_model(
        f"{model.provider}:{model.id}",
        temperature=0,
        **{key_kwarg_name: api_key},
        **thinking_kwargs,
    )


def create_llm_pair(
    model_id: str,
    api_key: str,
) -> tuple[BaseChatModel, BaseChatModel]:
    """Create the standard two-LLM pair for the agent graph.

    Returns (main_llm, synthesizer_llm) where:
    - main_llm: No thinking — safe for tool calls and structured output.
    - synthesizer_llm: Thinking enabled (if model supports it) — used
      for the synthesis node that produces the final response.

    If the model doesn't support thinking, both LLMs are identical.
    """
    main_llm = create_llm(model_id, api_key)

    model = get_model(model_id)
    if model is not None and model.thinking_capable:
        synthesizer_llm = create_llm(
            model_id, api_key, thinking=ThinkingConfig(enabled=True, level="medium")
        )
    else:
        synthesizer_llm = main_llm

    return main_llm, synthesizer_llm

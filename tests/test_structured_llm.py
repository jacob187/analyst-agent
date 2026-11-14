"""`structured_llm` avoids forced tool calling on Anthropic models.

Newer Claude models (always-on adaptive thinking) reject
``tool_choice`` "tool"/"any", which is how ``with_structured_output`` works by
default for Anthropic. These tests build real chat-model objects (no network)
and inspect what each would send.
"""

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from agents.llm_factory import structured_llm
from agents.model_registry import get_all_models


class _Verdict(BaseModel):
    winner: str


def _bound_kwargs(runnable) -> dict:
    """kwargs the structured-output chain binds onto the chat model."""
    first = runnable.first if hasattr(runnable, "first") else runnable
    return getattr(first, "kwargs", {})


def test_anthropic_models_use_native_structured_output_not_forced_tools():
    anthropic_ids = [m.id for m in get_all_models() if m.provider == "anthropic"]
    assert anthropic_ids
    for model_id in anthropic_ids:
        llm = ChatAnthropic(model=model_id, api_key="test-key")
        kwargs = _bound_kwargs(structured_llm(llm, _Verdict))
        assert "output_config" in kwargs, model_id
        assert "tool_choice" not in kwargs, model_id


def test_anthropic_models_get_an_explicit_output_limit_above_the_4096_fallback():
    from agents.llm_factory import ThinkingConfig, create_llm

    for model in [m for m in get_all_models() if m.provider == "anthropic"]:
        for thinking in (None, ThinkingConfig(enabled=True, level="medium")):
            llm = create_llm(model.id, "test-key", thinking=thinking)
            assert llm.max_tokens >= 16_000, (model.id, thinking)


def test_other_providers_keep_their_default_method():
    google_id = next(m.id for m in get_all_models() if m.provider == "google_genai")
    llm = ChatGoogleGenerativeAI(model=google_id, google_api_key="test-key")
    kwargs = _bound_kwargs(structured_llm(llm, _Verdict))
    assert "output_config" not in kwargs

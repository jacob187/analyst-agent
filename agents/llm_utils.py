"""Reusable utilities for extracting thinking/reasoning from LLM responses.

Gemini with `include_thoughts=True` returns `content` as a list of typed blocks:
  [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]

This module provides a single, consistent interface for any code path that
needs to separate thinking from text — briefings, graph nodes, future features.

Usage:
    from agents.llm_utils import parse_llm_response

    result = parse_llm_response(llm_response)
    print(result.text)      # The actual content
    print(result.thinking)  # The chain-of-thought reasoning (may be empty)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Parsed LLM response with thinking separated from text content.

    Attributes:
        text: The actual text output from the LLM.
        thinking: The chain-of-thought reasoning (empty string if none).
    """
    text: str
    thinking: str


def parse_llm_response(response) -> LLMResponse:
    """Parse an LLM response into separate text and thinking components.

    Handles three content formats:
    1. str — plain text, no thinking (standard responses)
    2. list[dict] — structured blocks from Gemini with include_thoughts=True
       Block types: "thinking"/"reasoning" (CoT) and "text" (actual output)
    3. Anything else — converted to string, no thinking

    Works with both raw LangChain AIMessage objects (has `.content`) and
    plain strings/lists.

    Args:
        response: LLM response object (AIMessage) or raw content value.

    Returns:
        LLMResponse with .text and .thinking fields.
    """
    # Unwrap AIMessage-like objects to get the raw content
    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        return LLMResponse(text=content, thinking="")

    if isinstance(content, list):
        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
            elif block_type in ("thinking", "reasoning"):
                thought = block.get(block_type, "")
                if thought:
                    thinking_parts.append(thought)

        return LLMResponse(
            text="".join(text_parts),
            thinking="\n\n".join(thinking_parts),
        )

    return LLMResponse(text=str(content), thinking="")


def extract_text(response) -> str:
    """Convenience function — extract just the text content from an LLM response.

    Drop-in replacement for inline text extraction scattered across the codebase.
    """
    return parse_llm_response(response).text


def extract_thinking(response) -> str:
    """Convenience function — extract just the thinking/reasoning from an LLM response."""
    return parse_llm_response(response).thinking

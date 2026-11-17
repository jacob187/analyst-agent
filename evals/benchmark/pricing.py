"""Cost estimation from per-million-token prices.

A missing price (or token counts we could not observe) means cost is unknown
(``None``), never 0.
"""

from __future__ import annotations

from agents.model_registry import ModelDef
from evals.benchmark.schemas import Pricing

_MTOK = 1_000_000


def pricing_from_model(model: ModelDef) -> Pricing:
    """Snapshot a registry model's price fields."""
    return Pricing(
        input_price_per_mtok=model.input_price_per_mtok,
        output_price_per_mtok=model.output_price_per_mtok,
        prices_as_of=model.prices_as_of,
    )


def estimate_cost(
    pricing: Pricing | None,
    input_tokens: int,
    output_tokens: int,
    llm_calls_without_usage: int = 0,
) -> float | None:
    """USD estimate, or None when a price or any call's token usage is missing."""
    if pricing is None:
        return None
    if pricing.input_price_per_mtok is None or pricing.output_price_per_mtok is None:
        return None
    if llm_calls_without_usage > 0:
        return None
    return (
        input_tokens * pricing.input_price_per_mtok
        + output_tokens * pricing.output_price_per_mtok
    ) / _MTOK


def cost_text(value: float | None) -> str:
    """A USD amount for display; None is "unknown", never $0."""
    return "unknown" if value is None else f"${value:.4f}"


def usage_text(input_tokens: int | None, output_tokens: int | None, calls_without_usage: int | None) -> str:
    """Judge token usage for display; flags totals that are only a lower bound."""
    if input_tokens is None:
        return "usage not recorded"
    caveat = " (some calls reported no usage)" if calls_without_usage else ""
    return f"{input_tokens} in / {output_tokens} out tokens{caveat}"

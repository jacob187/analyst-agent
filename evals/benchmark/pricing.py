"""Cost estimation from per-million-token prices.

A missing price (or token counts we could not observe) means cost is unknown
(``None``), never 0.
"""

from __future__ import annotations

from typing import Any

from evals.benchmark.schemas import Pricing

_MTOK = 1_000_000


def pricing_from_model(model: Any) -> Pricing:
    """Snapshot a registry ``ModelDef``'s price fields.

    Read with ``getattr`` so this works before and after the price fields exist
    on ``ModelDef``.
    """
    as_of = getattr(model, "prices_as_of", None)
    return Pricing(
        input_price_per_mtok=getattr(model, "input_price_per_mtok", None),
        output_price_per_mtok=getattr(model, "output_price_per_mtok", None),
        prices_as_of=None if as_of is None else str(as_of),
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

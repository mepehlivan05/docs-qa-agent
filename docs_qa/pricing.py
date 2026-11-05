"""Cost per request from the API's token usage.

Prices are USD per million tokens (Anthropic list prices, Nov 2026). Cache
writes use the 5-minute rate (1.25x input). Update the table if prices change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01),
}

# A fallback model may serve a refused request; price unknown models as the
# most expensive listed one so reported costs are never understated.
_DEFAULT = max(PRICES.values(), key=lambda p: p.output)


def cost_usd(model: str, usage) -> float:
    price = PRICES.get(model, _DEFAULT)
    tokens = {
        "input": usage.input_tokens or 0,
        "output": usage.output_tokens or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", None) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", None) or 0,
    }
    return (
        tokens["input"] * price.input
        + tokens["output"] * price.output
        + tokens["cache_read"] * price.cache_read
        + tokens["cache_write"] * price.cache_write
    ) / 1_000_000

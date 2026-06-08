"""Estimate the money saved by cache hits, from a small per-model price table."""

from __future__ import annotations

from dataclasses import dataclass, field

from semcache.models import Stats

# Approximate published prices in USD per 1M tokens (input, output). These are
# deliberately conservative defaults and fully overridable per cache.
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "default": (1.00, 3.00),
}


@dataclass
class PriceTable:
    """Per-model input/output token prices in USD per 1M tokens.

    The blended cost assumes an ``output_ratio`` fraction of the saved tokens
    were output tokens (defaults to 50/50), which is what makes a single
    ``tokens`` count translate into a dollar figure.
    """

    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(_DEFAULT_PRICES))
    output_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.output_ratio <= 1.0:
            raise ValueError("output_ratio must be between 0 and 1")
        if "default" not in self.prices:
            self.prices = {**self.prices, "default": _DEFAULT_PRICES["default"]}

    def blended_price_per_token(self, model: str | None) -> float:
        """Blended USD price per single token for ``model`` (falls back to default)."""
        input_price, output_price = self.prices.get(model or "default", self.prices["default"])
        per_million = input_price * (1.0 - self.output_ratio) + output_price * self.output_ratio
        return per_million / 1_000_000.0

    def cost_for(self, tokens: int, model: str | None) -> float:
        """USD cost of ``tokens`` tokens for ``model``."""
        return tokens * self.blended_price_per_token(model)


def estimate_savings(stats: Stats, prices: PriceTable | None = None) -> float:
    """Estimate dollars saved from ``stats.tokens_saved`` at default pricing.

    For per-model accuracy, accumulate ``cost_saved`` as hits occur (the cache
    does this with the originating model); this helper is a quick aggregate
    estimate against a single blended rate.
    """
    table = prices or PriceTable()
    return table.cost_for(stats.tokens_saved, None)

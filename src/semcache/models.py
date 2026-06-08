"""Pydantic models for cache entries and aggregate statistics."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class CacheEntry(BaseModel):
    """A single stored cache entry.

    ``vector`` is the (L2-normalized) embedding of ``key``; ``value`` is the
    cached result that semantically-similar prompts should resolve to.
    """

    key: str
    vector: list[float]
    value: str
    created_at: float = 0.0
    last_used_at: float = 0.0
    hits: int = 0
    tokens: int = 0
    model: str | None = None
    meta: dict[str, str] = Field(default_factory=dict)


class Stats(BaseModel):
    """Aggregate cache statistics over the lifetime of a :class:`SemanticCache`."""

    lookups: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    entries: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were served from cache, in ``[0.0, 1.0]``."""
        if self.lookups == 0:
            return 0.0
        return self.hits / self.lookups

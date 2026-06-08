"""semcache — zero-infra semantic cache for LLM (or any text->text) calls.

Cache on *meaning*, not exact string match: embed the prompt and return a
cached value when a semantically similar prompt (cosine similarity >=
threshold) was seen before. No proxy, no server, no account.

Everything ships offline and deterministic via :class:`HashingEmbedder`, so the
demo and the whole test suite run with zero network access.
"""

from __future__ import annotations

from semcache.cache import SemanticCache
from semcache.decorator import semantic_cache
from semcache.embedding import Embedder, HashingEmbedder, cosine
from semcache.models import CacheEntry, Stats
from semcache.savings import PriceTable, estimate_savings

__version__ = "0.1.0"

__all__ = [
    "CacheEntry",
    "Embedder",
    "HashingEmbedder",
    "PriceTable",
    "SemanticCache",
    "Stats",
    "__version__",
    "cosine",
    "estimate_savings",
    "semantic_cache",
]

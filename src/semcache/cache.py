"""The semantic cache itself.

:class:`SemanticCache` stores embedded prompts and serves a cached value when a
new prompt is semantically similar enough (cosine >= ``threshold``). It supports
an exact-match fast path, LRU eviction at ``max_size``, optional TTL expiry (via
an injectable clock for deterministic tests), and namespace isolation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from semcache.embedding import Embedder, HashingEmbedder, cosine
from semcache.models import CacheEntry, Stats
from semcache.savings import PriceTable


@dataclass(frozen=True)
class LookupResult:
    """Outcome of a cache lookup: whether it hit, the value, and the similarity."""

    hit: bool
    value: str | None
    similarity: float


class SemanticCache:
    """A meaning-aware cache for ``str -> str`` results.

    Args:
        threshold: minimum cosine similarity for a semantic hit (0..1).
        max_size: maximum stored entries before LRU eviction (``None`` = unbounded).
        ttl_seconds: entry lifetime; ``None`` disables expiry.
        embedder: any :class:`Embedder`; defaults to the offline
            :class:`HashingEmbedder`.
        namespace: logical partition; entries in different namespaces never match.
        time_fn: clock used for ``created_at``/TTL; injectable for deterministic tests.
        prices: price table used to accumulate ``cost_saved`` on hits.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        max_size: int | None = 1000,
        ttl_seconds: float | None = None,
        embedder: Embedder | None = None,
        namespace: str = "default",
        time_fn: Callable[[], float] | None = None,
        prices: PriceTable | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.namespace = namespace
        self._time_fn = time_fn or _default_clock
        self._prices = prices or PriceTable()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = Stats()

    # -- internals --------------------------------------------------------

    def _now(self) -> float:
        return self._time_fn()

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now - entry.created_at) >= self.ttl_seconds

    def _purge_expired(self, now: float) -> None:
        if self.ttl_seconds is None:
            return
        expired = [k for k, e in self._entries.items() if self._is_expired(e, now)]
        for key in expired:
            del self._entries[key]
            self._stats.expirations += 1

    # -- public API -------------------------------------------------------

    def lookup(self, prompt: str) -> LookupResult:
        """Find the best match for ``prompt`` without mutating statistics.

        Honours TTL (expired entries are skipped) and the exact-match fast path,
        but does **not** count the lookup or update LRU order — use :meth:`get`
        for the stateful path.
        """
        now = self._now()
        exact = self._entries.get(prompt)
        if exact is not None and not self._is_expired(exact, now):
            return LookupResult(hit=True, value=exact.value, similarity=1.0)

        query = self.embedder.embed(prompt)
        best_value: str | None = None
        best_sim = 0.0
        for entry in self._entries.values():
            if self._is_expired(entry, now):
                continue
            sim = cosine(query, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_value = entry.value
        if best_value is not None and best_sim >= self.threshold:
            return LookupResult(hit=True, value=best_value, similarity=best_sim)
        return LookupResult(hit=False, value=None, similarity=best_sim)

    def get(self, prompt: str) -> str | None:
        """Return a cached value for ``prompt`` or ``None``; updates stats/LRU/TTL."""
        now = self._now()
        self._purge_expired(now)
        self._stats.lookups += 1

        # Exact-match fast path.
        exact = self._entries.get(prompt)
        if exact is not None:
            self._record_hit(prompt, exact, now)
            return exact.value

        query = self.embedder.embed(prompt)
        best_key: str | None = None
        best_sim = 0.0
        for key, entry in self._entries.items():
            sim = cosine(query, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key is not None and best_sim >= self.threshold:
            entry = self._entries[best_key]
            self._record_hit(best_key, entry, now)
            return entry.value

        self._stats.misses += 1
        return None

    def _record_hit(self, key: str, entry: CacheEntry, now: float) -> None:
        entry.hits += 1
        entry.last_used_at = now
        self._entries.move_to_end(key)
        self._stats.hits += 1
        self._stats.tokens_saved += entry.tokens
        self._stats.cost_saved += self._prices.cost_for(entry.tokens, entry.model)

    def set(
        self,
        prompt: str,
        value: str,
        *,
        tokens: int = 0,
        model: str | None = None,
        meta: dict[str, str] | None = None,
    ) -> CacheEntry:
        """Store ``value`` under ``prompt``, embedding it for future similarity hits."""
        now = self._now()
        self._purge_expired(now)
        entry = CacheEntry(
            key=prompt,
            vector=self.embedder.embed(prompt),
            value=value,
            created_at=now,
            last_used_at=now,
            tokens=tokens,
            model=model,
            meta=meta or {},
        )
        if prompt in self._entries:
            del self._entries[prompt]
        self._entries[prompt] = entry
        self._evict_if_needed()
        self._stats.entries = len(self._entries)
        return entry

    def _evict_if_needed(self) -> None:
        if self.max_size is None:
            return
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
            self._stats.evictions += 1

    def clear(self) -> None:
        """Remove all entries (statistics are preserved)."""
        self._entries.clear()
        self._stats.entries = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, prompt: str) -> bool:
        entry = self._entries.get(prompt)
        if entry is None:
            return False
        return not self._is_expired(entry, self._now())

    @property
    def stats(self) -> Stats:
        """A snapshot of current statistics (live entry count refreshed)."""
        self._stats.entries = len(self._entries)
        return self._stats

    @property
    def entries(self) -> list[CacheEntry]:
        """Stored entries in LRU order (oldest first, most-recently-used last)."""
        return list(self._entries.values())


def _default_clock() -> float:
    import time

    return time.monotonic()

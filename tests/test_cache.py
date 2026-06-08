"""Tests for the SemanticCache behaviour."""

from __future__ import annotations

import itertools

import pytest

from semcache.cache import SemanticCache


def make_clock():
    counter = itertools.count(start=1.0, step=1.0)
    return lambda: next(counter)


def test_exact_match_hit():
    cache = SemanticCache(threshold=0.9)
    cache.set("hello world", "greeting")
    assert cache.get("hello world") == "greeting"


def test_miss_on_empty_cache():
    cache = SemanticCache()
    assert cache.get("anything") is None


def test_paraphrase_above_threshold_hits():
    cache = SemanticCache(threshold=0.5)
    cache.set("how do I reset my password", "reset instructions")
    assert cache.get("how can I reset the password") == "reset instructions"


def test_unrelated_prompt_misses():
    cache = SemanticCache(threshold=0.5)
    cache.set("how do I reset my password", "reset instructions")
    assert cache.get("tell me about quantum physics") is None


def test_threshold_boundary_high_blocks_paraphrase():
    cache = SemanticCache(threshold=0.999)
    cache.set("how do I reset my password", "reset")
    # Near-identical but not exact -> blocked by the very high threshold.
    assert cache.get("how can I reset the password") is None


def test_threshold_zero_matches_anything_nonzero():
    cache = SemanticCache(threshold=0.0)
    cache.set("how do I reset my password", "reset")
    # Even a weakly-related prompt clears a zero threshold if cosine > 0.
    assert cache.get("how do I reset my account") == "reset"


def test_lookup_does_not_mutate_stats():
    cache = SemanticCache()
    cache.set("hello", "world")
    result = cache.lookup("hello")
    assert result.hit is True
    assert result.value == "world"
    assert result.similarity == pytest.approx(1.0)
    assert cache.stats.lookups == 0


def test_lookup_reports_similarity_on_miss():
    cache = SemanticCache(threshold=0.99)
    cache.set("how do I reset my password", "x")
    result = cache.lookup("how can I reset the password")
    assert result.hit is False
    assert 0.0 < result.similarity < 0.99


def test_set_overwrites_existing_key():
    cache = SemanticCache()
    cache.set("key", "first")
    cache.set("key", "second")
    assert cache.get("key") == "second"
    assert len(cache) == 1


def test_clear_removes_entries():
    cache = SemanticCache()
    cache.set("a", "1")
    cache.set("b", "2")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_contains_operator():
    cache = SemanticCache()
    cache.set("present", "yes")
    assert "present" in cache
    assert "absent" not in cache


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        SemanticCache(threshold=1.5)


def test_invalid_max_size_raises():
    with pytest.raises(ValueError):
        SemanticCache(max_size=0)


def test_invalid_ttl_raises():
    with pytest.raises(ValueError):
        SemanticCache(ttl_seconds=0)


# -- LRU eviction ---------------------------------------------------------


def test_lru_eviction_at_max_size():
    cache = SemanticCache(max_size=2)
    cache.set("alpha apple", "1")
    cache.set("bravo banana", "2")
    cache.set("charlie cherry", "3")
    assert len(cache) == 2
    assert cache.stats.evictions == 1
    # Oldest ("alpha apple") evicted first.
    assert cache.get("alpha apple") is None


def test_lru_get_refreshes_recency():
    cache = SemanticCache(max_size=2, threshold=1.0)
    cache.set("alpha apple", "1")
    cache.set("bravo banana", "2")
    cache.get("alpha apple")  # touch alpha -> now most recent
    cache.set("charlie cherry", "3")  # should evict bravo, not alpha
    assert cache.get("alpha apple") == "1"
    assert cache.get("bravo banana") is None


def test_unbounded_cache_never_evicts():
    cache = SemanticCache(max_size=None)
    for i in range(50):
        cache.set(f"prompt number {i}", str(i))
    assert len(cache) == 50
    assert cache.stats.evictions == 0


# -- TTL ------------------------------------------------------------------


def test_ttl_expiry_via_injected_clock():
    cache = SemanticCache(ttl_seconds=5, time_fn=make_clock())
    cache.set("hello", "world")  # created_at = 1
    # subsequent gets advance the clock; by t=6+ the entry is expired.
    assert cache.get("hello") == "world"  # t=2, fresh
    for _ in range(10):
        cache.get("hello")
    assert cache.get("hello") is None


def test_ttl_not_expired_within_window():
    clock = itertools.count(start=100.0, step=1.0)
    cache = SemanticCache(ttl_seconds=1000, time_fn=lambda: next(clock))
    cache.set("hello", "world")
    assert cache.get("hello") == "world"


def test_expired_entry_counts_as_miss():
    times = iter([1.0, 2.0, 100.0, 100.0])
    cache = SemanticCache(ttl_seconds=5, time_fn=lambda: next(times))
    cache.set("hello", "world")  # t=1
    # get at t=2 not reached because purge uses t=100 -> wait, set consumed t=1.
    # Next get: now=t=2 (fresh)
    assert cache.get("hello") == "world"
    # Next get: now=t=100 -> expired, purged, miss.
    assert cache.get("hello") is None
    assert cache.stats.misses == 1


def test_contains_respects_ttl():
    times = iter([1.0, 100.0])
    cache = SemanticCache(ttl_seconds=5, time_fn=lambda: next(times))
    cache.set("hello", "world")  # t=1
    assert "hello" not in cache  # checked at t=100 -> expired


# -- namespaces -----------------------------------------------------------


def test_namespace_isolation():
    clock = make_clock()
    a = SemanticCache(namespace="tenant-a", time_fn=clock)
    b = SemanticCache(namespace="tenant-b", time_fn=clock)
    a.set("shared prompt", "answer-a")
    assert a.get("shared prompt") == "answer-a"
    assert b.get("shared prompt") is None


def test_entries_property_in_lru_order():
    cache = SemanticCache(threshold=1.0)
    cache.set("first one", "1")
    cache.set("second one", "2")
    keys = [e.key for e in cache.entries]
    assert keys == ["first one", "second one"]

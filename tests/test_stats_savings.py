"""Tests for statistics math and savings estimation."""

from __future__ import annotations

import itertools

import pytest

from semcache.cache import SemanticCache
from semcache.models import Stats
from semcache.savings import PriceTable, estimate_savings


def test_hit_rate_zero_when_no_lookups():
    assert Stats().hit_rate == 0.0


def test_hit_rate_math():
    s = Stats(lookups=4, hits=3, misses=1)
    assert s.hit_rate == pytest.approx(0.75)


def test_hit_rate_all_hits():
    s = Stats(lookups=5, hits=5)
    assert s.hit_rate == pytest.approx(1.0)


def test_stats_track_lookups_hits_misses():
    cache = SemanticCache(threshold=0.5)
    cache.set("how do I reset my password", "x", tokens=100)
    cache.get("how can I reset the password")  # hit
    cache.get("completely different topic here")  # miss
    s = cache.stats
    assert s.lookups == 2
    assert s.hits == 1
    assert s.misses == 1
    assert s.hit_rate == pytest.approx(0.5)


def test_tokens_saved_accumulate_on_hits():
    cache = SemanticCache(threshold=0.5)
    cache.set("how do I reset my password", "x", tokens=120)
    cache.get("how can I reset the password")
    cache.get("how do I reset my password")  # exact hit
    assert cache.stats.tokens_saved == 240


def test_cost_saved_accumulates():
    cache = SemanticCache(threshold=1.0)
    cache.set("hello", "world", tokens=1_000_000, model="gpt-4o")
    cache.get("hello")
    # blended 50/50 of (2.50, 10.00) per 1M = 6.25 per 1M tokens.
    assert cache.stats.cost_saved == pytest.approx(6.25)


def test_miss_does_not_accumulate_tokens():
    cache = SemanticCache(threshold=0.99)
    cache.set("how do I reset my password", "x", tokens=100)
    cache.get("a totally unrelated sentence about cats")
    assert cache.stats.tokens_saved == 0


def test_price_table_blended_price():
    table = PriceTable()
    # gpt-4o-mini: (0.15, 0.60) -> 50/50 blend = 0.375 per 1M.
    assert table.blended_price_per_token("gpt-4o-mini") == pytest.approx(0.375 / 1_000_000)


def test_price_table_falls_back_to_default():
    table = PriceTable()
    assert table.cost_for(1000, "unknown-model") == table.cost_for(1000, None)


def test_price_table_output_ratio():
    table = PriceTable(output_ratio=1.0)
    # All output tokens for gpt-4o -> 10.00 per 1M.
    assert table.cost_for(1_000_000, "gpt-4o") == pytest.approx(10.0)


def test_price_table_input_only():
    table = PriceTable(output_ratio=0.0)
    assert table.cost_for(1_000_000, "gpt-4o") == pytest.approx(2.5)


def test_price_table_override():
    table = PriceTable(prices={"custom": (100.0, 100.0), "default": (1.0, 3.0)})
    assert table.cost_for(1_000_000, "custom") == pytest.approx(100.0)


def test_price_table_invalid_ratio_raises():
    with pytest.raises(ValueError):
        PriceTable(output_ratio=2.0)


def test_estimate_savings_from_stats():
    s = Stats(tokens_saved=2_000_000)
    saved = estimate_savings(s)
    # default model blend (1.0, 3.0) -> 2.0 per 1M -> 4.0 for 2M.
    assert saved == pytest.approx(4.0)


def test_estimate_savings_zero():
    assert estimate_savings(Stats()) == 0.0


def test_evictions_counted_in_stats():
    cache = SemanticCache(max_size=1)
    cache.set("alpha apple", "1")
    cache.set("bravo banana", "2")
    assert cache.stats.evictions == 1


def test_expirations_counted_in_stats():
    times = itertools.count(start=1.0, step=50.0)
    cache = SemanticCache(ttl_seconds=10, time_fn=lambda: next(times))
    cache.set("hello", "world")
    cache.get("hello")  # next tick -> expired -> purged
    assert cache.stats.expirations >= 1


def test_entries_count_reflects_size():
    cache = SemanticCache()
    cache.set("one apple", "1")
    cache.set("two banana", "2")
    assert cache.stats.entries == 2

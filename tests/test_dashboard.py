"""Tests for the rich dashboard rendering."""

from __future__ import annotations

import io

from rich.console import Console

from semcache.cache import SemanticCache
from semcache.dashboard import entries_table, render, stats_table
from semcache.models import Stats


def test_stats_table_builds():
    table = stats_table(Stats(lookups=3, hits=2, misses=1, tokens_saved=100, cost_saved=0.5))
    assert table.row_count == 8


def test_entries_table_respects_limit():
    cache = SemanticCache(threshold=1.0)
    for i in range(20):
        cache.set(f"prompt number {i}", str(i))
    table = entries_table(cache.entries, limit=5)
    assert table.row_count == 5


def test_render_writes_output():
    cache = SemanticCache(threshold=0.5)
    cache.set("how do I reset my password", "answer", tokens=100)
    cache.get("how can I reset the password")
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    render(cache, console=console)
    text = buffer.getvalue()
    assert "statistics" in text.lower()
    assert "Hit rate" in text


def test_entries_table_empty():
    table = entries_table([])
    assert table.row_count == 0

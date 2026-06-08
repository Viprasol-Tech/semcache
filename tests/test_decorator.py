"""Tests for the semantic_cache decorator."""

from __future__ import annotations

from semcache.cache import SemanticCache
from semcache.decorator import semantic_cache


def test_decorator_wraps_and_returns_value():
    @semantic_cache(threshold=0.5)
    def fake_llm(prompt: str) -> str:
        return f"answer to: {prompt}"

    assert fake_llm("hello there friend") == "answer to: hello there friend"


def test_decorator_exposes_cache():
    @semantic_cache(threshold=0.5)
    def fake_llm(prompt: str) -> str:
        return "x"

    assert isinstance(fake_llm.cache, SemanticCache)


def test_decorator_calls_func_only_on_miss():
    calls: list[str] = []

    @semantic_cache(threshold=0.5)
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "cached answer"

    fake_llm("how do I reset my password")  # miss -> calls func
    fake_llm("how can I reset the password")  # paraphrase -> HIT, no call
    assert len(calls) == 1


def test_decorator_counts_hits():
    @semantic_cache(threshold=0.5)
    def fake_llm(prompt: str) -> str:
        return "answer"

    fake_llm("how do I reset my password")
    fake_llm("how can I reset the password")
    assert fake_llm.cache.stats.hits == 1
    assert fake_llm.cache.stats.misses == 1


def test_decorator_exact_repeat_hits():
    calls: list[str] = []

    @semantic_cache(threshold=0.9)
    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "answer"

    fake_llm("identical prompt")
    fake_llm("identical prompt")
    assert len(calls) == 1
    assert fake_llm.cache.stats.hits == 1


def test_decorator_preserves_metadata():
    @semantic_cache(threshold=0.5)
    def fake_llm(prompt: str) -> str:
        """Docstring preserved."""
        return "x"

    assert fake_llm.__name__ == "fake_llm"
    assert fake_llm.__doc__ == "Docstring preserved."


def test_decorator_shared_store():
    store = SemanticCache(threshold=0.5)

    @semantic_cache(store=store)
    def first(prompt: str) -> str:
        return "from-first"

    @semantic_cache(store=store)
    def second(prompt: str) -> str:
        return "from-second"

    first("how do I reset my password")
    # second sees a paraphrase already cached by first via the shared store.
    assert second("how can I reset the password") == "from-first"


def test_decorator_custom_key():
    @semantic_cache(threshold=0.9, key=lambda *a, **k: str(k["q"]))
    def fake_llm(*, q: str) -> str:
        return f"got {q}"

    assert fake_llm(q="hello") == "got hello"
    assert fake_llm.cache.stats.lookups == 1


def test_decorator_records_model_and_tokens():
    @semantic_cache(threshold=1.0, model="gpt-4o", tokens=lambda s: 500)
    def fake_llm(prompt: str) -> str:
        return "answer text"

    fake_llm("a prompt")
    entry = fake_llm.cache.entries[0]
    assert entry.model == "gpt-4o"
    assert entry.tokens == 500


def test_decorator_default_token_estimate():
    @semantic_cache(threshold=1.0)
    def fake_llm(prompt: str) -> str:
        return "x" * 40

    fake_llm("a prompt")
    # ~4 chars/token -> 10 tokens.
    assert fake_llm.cache.entries[0].tokens == 10

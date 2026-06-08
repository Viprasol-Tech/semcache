"""The :func:`semantic_cache` decorator — wrap any ``func(prompt, ...) -> str``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, Protocol, cast

from semcache.cache import SemanticCache
from semcache.embedding import Embedder

P = ParamSpec("P")


class CachedCallable(Protocol[P]):
    """A wrapped function that also exposes its backing :class:`SemanticCache`."""

    cache: SemanticCache

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str: ...


def semantic_cache(
    store: SemanticCache | None = None,
    *,
    threshold: float = 0.9,
    embedder: Embedder | None = None,
    key: Callable[..., str] | None = None,
    tokens: Callable[[str], int] | None = None,
    model: str | None = None,
) -> Callable[[Callable[P, str]], CachedCallable[P]]:
    """Cache a ``func(prompt: str, ...) -> str`` callable on prompt *meaning*.

    On each call the prompt (first positional arg, or whatever ``key`` extracts)
    is looked up; on a hit the cached string is returned without calling
    ``func``; on a miss ``func`` runs and its result is stored. The backing
    store is attached to the wrapper as ``wrapped.cache``.

    Args:
        store: an existing cache to share; a new one is created if omitted.
        threshold/embedder: used only when ``store`` is omitted.
        key: extract the cache key from the call args (default: first arg).
        tokens: estimate tokens saved from the *result* (for savings accounting).
        model: model name recorded with stored entries (for cost accounting).
    """
    cache = store if store is not None else SemanticCache(threshold=threshold, embedder=embedder)
    extract_key = key or _first_arg
    count_tokens = tokens or _estimate_tokens

    def decorator(func: Callable[P, str]) -> CachedCallable[P]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
            prompt = extract_key(*args, **kwargs)
            cached = cache.get(prompt)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            cache.set(prompt, result, tokens=count_tokens(result), model=model)
            return result

        wrapped = cast(CachedCallable[P], wrapper)
        wrapped.cache = cache
        return wrapped

    return decorator


def _first_arg(*args: object, **kwargs: object) -> str:
    """Default key extractor: the first positional argument, stringified."""
    if args:
        return str(args[0])
    if kwargs:
        return str(next(iter(kwargs.values())))
    raise TypeError("semantic_cache: cannot derive a cache key from no arguments")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when no estimator is given."""
    return max(1, len(text) // 4)

"""Embedders and cosine similarity.

The default :class:`HashingEmbedder` is fully **offline and deterministic**: it
hashes word tokens into a fixed-dimension vector and L2-normalizes the result.
Prompts that share words land near each other in vector space, so paraphrases
score a high cosine similarity — without any network call or model download.

To use a real embedder (OpenAI, sentence-transformers, ...) implement the
:class:`Embedder` protocol; see the README "plug in a real embedder" section.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-length list of floats."""

    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        """Return the embedding vector for ``text``."""
        ...


def _tokens(text: str) -> list[str]:
    """Lowercase and tokenize ``text`` on word boundaries."""
    return _TOKEN_RE.findall(text.lower())


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """Map a token to a (deterministic) dimension index and a +/-1 sign."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashingEmbedder:
    """Deterministic, dependency-free hashing embedder.

    Each token is hashed into one of ``dim`` buckets with a stable +/-1 sign and
    accumulated; the vector is then L2-normalized. Optionally, adjacent-word
    bigrams contribute a smaller ``ngram_weight`` so word *order* nudges
    similarity without dominating it.
    """

    def __init__(self, dim: int = 256, *, ngram_weight: float = 0.5) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if ngram_weight < 0:
            raise ValueError("ngram_weight must be non-negative")
        self.dim = dim
        self.ngram_weight = ngram_weight

    def embed(self, text: str) -> list[float]:
        """Embed ``text`` into an L2-normalized vector of length ``dim``."""
        vec = [0.0] * self.dim
        tokens = _tokens(text)
        for token in tokens:
            index, sign = _bucket(token, self.dim)
            vec[index] += sign
        if self.ngram_weight > 0:
            for a, b in itertools.pairwise(tokens):
                index, sign = _bucket(f"{a}_{b}", self.dim)
                vec[index] += sign * self.ngram_weight
        norm = math.sqrt(sum(component * component for component in vec))
        if norm == 0.0:
            return vec
        return [component / norm for component in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` if either vector is all-zero. Raises ``ValueError`` on a
    length mismatch.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

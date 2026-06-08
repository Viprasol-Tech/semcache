"""Tests for the hashing embedder and cosine similarity."""

from __future__ import annotations

import math

import pytest

from semcache.embedding import Embedder, HashingEmbedder, cosine


def test_embedder_is_deterministic():
    emb = HashingEmbedder()
    a = emb.embed("the quick brown fox")
    b = emb.embed("the quick brown fox")
    assert a == b


def test_embedding_has_fixed_dimension():
    emb = HashingEmbedder(dim=128)
    assert len(emb.embed("hello world")) == 128
    assert len(emb.embed("")) == 128


def test_default_dimension_is_256():
    assert len(HashingEmbedder().embed("anything")) == 256


def test_embedding_is_l2_normalized():
    vec = HashingEmbedder().embed("normalize me please")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0)


def test_empty_text_is_zero_vector():
    vec = HashingEmbedder().embed("   ")
    assert all(x == 0.0 for x in vec)


def test_lowercasing_is_applied():
    emb = HashingEmbedder()
    assert emb.embed("Hello World") == emb.embed("hello world")


def test_punctuation_is_stripped():
    emb = HashingEmbedder()
    assert emb.embed("hello, world!") == emb.embed("hello world")


def test_shared_words_give_high_cosine():
    emb = HashingEmbedder()
    a = emb.embed("how do I reset my password")
    b = emb.embed("how can I reset the password")
    assert cosine(a, b) > 0.5


def test_unrelated_text_gives_low_cosine():
    emb = HashingEmbedder()
    a = emb.embed("how do I reset my password")
    b = emb.embed("penguins waddle across antarctic ice")
    assert cosine(a, b) < 0.3


def test_self_cosine_is_one():
    vec = HashingEmbedder().embed("identical vector test")
    assert cosine(vec, vec) == pytest.approx(1.0)


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_negative_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_ngram_weight_affects_word_order():
    emb = HashingEmbedder(ngram_weight=1.0)
    a = emb.embed("dog bites man")
    b = emb.embed("man bites dog")
    # Same tokens but different order: bigrams differ, so not identical.
    assert a != b


def test_zero_ngram_weight_ignores_order():
    emb = HashingEmbedder(ngram_weight=0.0)
    a = emb.embed("dog bites man")
    b = emb.embed("man bites dog")
    assert a == b


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_negative_ngram_weight_raises():
    with pytest.raises(ValueError):
        HashingEmbedder(ngram_weight=-1.0)


def test_hashing_embedder_satisfies_protocol():
    assert isinstance(HashingEmbedder(), Embedder)


def test_more_shared_words_means_higher_similarity():
    emb = HashingEmbedder()
    base = emb.embed("the cat sat on the mat")
    close = emb.embed("the cat sat on a mat")
    far = emb.embed("the cat ran")
    assert cosine(base, close) > cosine(base, far)

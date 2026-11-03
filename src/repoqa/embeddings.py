"""Embedders: a deterministic offline `HashingEmbedder` and an OpenAI-backed one."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from functools import partial
from typing import Protocol

import openai

from repoqa.clients import OPENAI_ERRORS, openai_client
from repoqa.config import Settings
from repoqa.errors import ProviderError
from repoqa.resilience import RateLimiter, RetryPolicy, call_with_retry, classify_sdk_errors
from repoqa.text import tokenize


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier stored with the index, so queries use the same embedder."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature-hashing embedder: tokens and their character trigrams are hashed into a
    fixed-size signed vector, then L2-normalised. No model, no network, fully deterministic.

    Trigrams give it some fuzziness (`hashing` ~ `hashed`) that exact-match BM25 lacks.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, weight in _features(text):
            digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest())
            sign = 1.0 if digest & 1 else -1.0
            vector[(digest >> 1) % self.dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


def _features(text: str) -> Iterator[tuple[str, float]]:
    for token in tokenize(text):
        yield f"w:{token}", 1.0
        padded = f"<{token}>"
        for i in range(len(padded) - 2):
            yield f"g:{padded[i : i + 3]}", 0.25


MAX_INPUT_CHARS = 24_000


class OpenAIEmbedder:
    """OpenAI embeddings API, called in batches through the retry policy and rate limiter."""

    def __init__(
        self,
        model: str,
        *,
        client: openai.OpenAI,
        policy: RetryPolicy,
        limiter: RateLimiter,
        batch_size: int = 96,
    ) -> None:
        self.model = model
        self._client = client
        self._policy = policy
        self._limiter = limiter
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # The API rejects inputs over 8,191 tokens; ~4 characters per token leaves a margin.
        texts = [text[:MAX_INPUT_CHARS] for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            request = partial(self._embed_batch, batch)
            vectors.extend(call_with_retry(request, self._policy, what="OpenAI embeddings"))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        with self._limiter, classify_sdk_errors("OpenAI", OPENAI_ERRORS):
            response = self._client.embeddings.create(model=self.model, input=batch)
        if len(response.data) != len(batch):
            raise ProviderError(f"OpenAI returned {len(response.data)} vectors for {len(batch)}")
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def build_embedder(settings: Settings, limiter: RateLimiter) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(
            settings.openai_embedding_model,
            client=openai_client(settings.timeout_s),
            policy=RetryPolicy(max_retries=settings.max_retries),
            limiter=limiter,
        )
    return HashingEmbedder()

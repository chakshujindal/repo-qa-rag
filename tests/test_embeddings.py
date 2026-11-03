import json
import math

import httpx2
import pytest

from repoqa.config import Settings
from repoqa.embeddings import HashingEmbedder, OpenAIEmbedder, build_embedder
from repoqa.errors import ConfigError, ProviderError
from repoqa.resilience import RateLimiter, RetryPolicy
from tests.helpers import mock_openai, sequence


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=64)
    first, again = embedder.embed(["hash the password"]), embedder.embed(["hash the password"])
    assert first == again
    assert len(first[0]) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first[0])), 1.0)
    assert embedder.name == "hashing-64"


def test_hashing_embedder_similarity_tracks_shared_words() -> None:
    a, b, c = HashingEmbedder().embed(
        ["hash_password stores a salted digest", "how is the password hashed", "render email"]
    )
    assert cosine(a, b) > cosine(a, c)


def test_empty_text_gives_zero_vector() -> None:
    assert set(HashingEmbedder(dim=8).embed(["the of"])[0]) == {0.0}


def embedding_response(vectors: list[list[float]]) -> httpx2.Response:
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
    data.reverse()  # the API does not promise order; we sort by index
    usage = {"prompt_tokens": 1, "total_tokens": 1}
    return httpx2.Response(200, json={"object": "list", "data": data, "model": "m", "usage": usage})


def test_openai_embedder_batches_and_orders_by_index(
    no_wait_policy: RetryPolicy, limiter: RateLimiter
) -> None:
    handler, seen = sequence(embedding_response([[1.0], [2.0]]), embedding_response([[3.0]]))
    embedder = OpenAIEmbedder(
        "text-embedding-3-small",
        client=mock_openai(handler),
        policy=no_wait_policy,
        limiter=limiter,
        batch_size=2,
    )
    assert embedder.embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]
    assert [json.loads(r.content)["input"] for r in seen] == [["a", "b"], ["c"]]
    assert embedder.name == "openai:text-embedding-3-small"


def test_openai_embedder_retries_429(no_wait_policy: RetryPolicy, limiter: RateLimiter) -> None:
    handler, seen = sequence(
        httpx2.Response(429, json={"error": {"message": "slow down"}}),
        embedding_response([[0.5]]),
    )
    embedder = OpenAIEmbedder(
        "m", client=mock_openai(handler), policy=no_wait_policy, limiter=limiter
    )
    assert embedder.embed(["a"]) == [[0.5]]
    assert len(seen) == 2


def test_openai_embedder_rejects_wrong_vector_count(
    no_wait_policy: RetryPolicy, limiter: RateLimiter
) -> None:
    handler, _ = sequence(embedding_response([[1.0]]))
    embedder = OpenAIEmbedder(
        "m", client=mock_openai(handler), policy=no_wait_policy, limiter=limiter
    )
    with pytest.raises(ProviderError, match="1 vectors for 2"):
        embedder.embed(["a", "b"])


def test_build_embedder(monkeypatch: pytest.MonkeyPatch, limiter: RateLimiter) -> None:
    assert isinstance(build_embedder(Settings(), limiter), HashingEmbedder)
    with pytest.raises(ConfigError, match="OpenAI client"):
        build_embedder(Settings(embedder="openai"), limiter)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(build_embedder(Settings(embedder="openai"), limiter), OpenAIEmbedder)

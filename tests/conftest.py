"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from repoqa.embeddings import HashingEmbedder
from repoqa.indexer import index_repo
from repoqa.resilience import RateLimiter, RetryPolicy
from repoqa.retrieval import Retriever
from repoqa.store import VectorStore
from tests.helpers import SAMPLE_REPO

ENV_PREFIXES = ("REPOQA_", "OPENAI_", "ANTHROPIC_")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests independent of the developer's shell, config and real API keys."""
    for name in list(os.environ):
        if name.startswith(ENV_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setenv("REPOQA_HOME", str(tmp_path / "home"))


@pytest.fixture(scope="session")
def indexed_sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[VectorStore, str]:
    """The bookshop fixture indexed once per test session with the hashing embedder."""
    store = VectorStore(tmp_path_factory.mktemp("chroma"))
    index_repo(SAMPLE_REPO, "bookshop", store, HashingEmbedder(), 200_000)
    return store, "bookshop"


@pytest.fixture(scope="session")
def retriever(indexed_sample: tuple[VectorStore, str]) -> Retriever:
    store, repo = indexed_sample
    return Retriever(store, repo, HashingEmbedder())


@pytest.fixture
def no_wait_policy() -> RetryPolicy:
    return RetryPolicy(max_retries=2, base_delay_s=0.0, max_delay_s=0.0)


@pytest.fixture
def limiter() -> RateLimiter:
    return RateLimiter(2, 60_000, sleep=lambda _: None)

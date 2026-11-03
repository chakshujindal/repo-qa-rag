import pytest
from fastapi.testclient import TestClient

from repoqa.config import Settings
from repoqa.embeddings import HashingEmbedder
from repoqa.errors import ProviderError
from repoqa.indexer import index_repo
from repoqa.llm import ChatMessage, FakeLLM
from repoqa.server import TokenBucket, create_app
from repoqa.store import VectorStore


class FailingLLM:
    name = "failing"

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        raise ProviderError("upstream unavailable after 5 attempts")


def make_client(store: VectorStore, llm: object | None = None, per_minute: int = 600) -> TestClient:
    app = create_app(
        Settings(api_requests_per_minute=per_minute),
        store=store,
        embedder=HashingEmbedder(),
        llm=llm or FakeLLM(),  # type: ignore[arg-type]
    )
    return TestClient(app)


def test_health_lists_repos_and_providers(indexed_sample: tuple[VectorStore, str]) -> None:
    store, repo = indexed_sample
    body = make_client(store).get("/health").json()
    assert body["status"] == "ok"
    assert repo in body["repos"]
    assert (body["embedder"], body["llm"]) == ("hashing-512", "fake")


def test_ask_returns_cited_answer(indexed_sample: tuple[VectorStore, str]) -> None:
    store, repo = indexed_sample
    response = make_client(store).post(
        "/ask", json={"question": "How is a discount code applied?", "repo": repo, "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert any(c.startswith("bookshop/cart.py") for c in body["citations"])
    assert len(body["sources"]) == 3


def test_ask_validates_input(indexed_sample: tuple[VectorStore, str]) -> None:
    store, _ = indexed_sample
    assert make_client(store).post("/ask", json={"question": ""}).status_code == 422


def test_unknown_repo_is_404(indexed_sample: tuple[VectorStore, str]) -> None:
    store, _ = indexed_sample
    response = make_client(store).post("/ask", json={"question": "x", "repo": "nope"})
    assert response.status_code == 404
    assert "No index named 'nope'" in response.json()["detail"]


def test_provider_failure_is_502_and_logged_once(
    indexed_sample: tuple[VectorStore, str], caplog: pytest.LogCaptureFixture
) -> None:
    store, repo = indexed_sample
    client = make_client(store, llm=FailingLLM())
    response = client.post("/ask", json={"question": "How are passwords hashed?", "repo": repo})
    assert response.status_code == 502
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "POST /ask -> 502" in errors[0].getMessage()


def test_rate_limit_returns_429_with_retry_after(indexed_sample: tuple[VectorStore, str]) -> None:
    store, _ = indexed_sample
    client = make_client(store, per_minute=6)  # burst of 1, then one token every 10 s
    assert client.get("/health").status_code == 200
    limited = client.get("/health")
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["Retry-After"]) <= 10


def test_reindex_while_serving_refreshes_the_cached_retriever(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pay.py").write_text("def refund_payment():\n    return 'refund issued'\n")
    index_repo(repo, "live", store, HashingEmbedder(), 10_000)
    client = make_client(store)
    question = {"question": "How is a payment refunded?", "repo": "live"}
    assert client.post("/ask", json=question).json()["citations"] == ["pay.py:1-2"]

    (repo / "pay.py").write_text("\n\ndef refund_payment():\n    return 'refund issued'\n")
    index_repo(repo, "live", store, HashingEmbedder(), 10_000)
    response = client.post("/ask", json=question)
    assert response.status_code == 200
    assert response.json()["citations"] == ["pay.py:3-4"]


def test_index_page_is_served(indexed_sample: tuple[VectorStore, str]) -> None:
    store, _ = indexed_sample
    response = make_client(store).get("/")
    assert response.status_code == 200
    assert "<title>repoqa</title>" in response.text


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_token_bucket_refills_and_is_per_client() -> None:
    clock = Clock()
    bucket = TokenBucket(per_minute=60, burst=2, clock=clock)
    assert bucket.acquire("a") == 0.0
    assert bucket.acquire("a") == 0.0
    assert bucket.acquire("a") == pytest.approx(1.0)  # empty: one token per second
    assert bucket.acquire("b") == 0.0  # other clients are unaffected
    clock.now = 1.0
    assert bucket.acquire("a") == 0.0


def test_token_bucket_evicts_idle_clients() -> None:
    clock = Clock()
    bucket = TokenBucket(per_minute=60, burst=1, clock=clock, max_clients=2)
    bucket.acquire("a")
    bucket.acquire("b")
    clock.now = 5.0
    bucket.acquire("c")
    assert set(bucket._buckets) == {"c"}

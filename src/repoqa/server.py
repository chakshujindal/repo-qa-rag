"""Small HTTP API: POST /ask, GET /health, and a one-page UI at GET /. Rate limited per client."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable
from importlib.resources import files

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from repoqa.answer import answer_question
from repoqa.config import Settings
from repoqa.embeddings import Embedder, build_embedder
from repoqa.errors import (
    IndexMismatchError,
    IndexNotFoundError,
    LLMOutputError,
    ProviderError,
    RepoQAError,
)
from repoqa.llm import LLMClient, build_llm
from repoqa.models import AnswerResult
from repoqa.resilience import RateLimiter
from repoqa.retrieval import Retriever
from repoqa.store import VectorStore, resolve_repo

logger = logging.getLogger(__name__)

ERROR_STATUS: dict[type[RepoQAError], int] = {
    IndexNotFoundError: 404,
    IndexMismatchError: 409,
    ProviderError: 502,
    LLMOutputError: 502,
}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    repo: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class TokenBucket:
    """Per-client token bucket: refills `per_minute` tokens a minute, holds at most `burst`."""

    def __init__(
        self,
        per_minute: int,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        self._rate = per_minute / 60.0
        self._burst = float(burst)
        self._clock = clock
        self._max_clients = max_clients
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last refill time)
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """Take one token. Returns 0 if allowed, else the seconds until a token is available."""
        with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            if len(self._buckets) >= self._max_clients:
                self._evict_idle(now)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1.0 - tokens) / self._rate

    def _evict_idle(self, now: float) -> None:
        """Forget clients whose bucket has refilled completely; they look like new clients."""
        full_after = self._burst / self._rate
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < full_after}


def create_app(
    settings: Settings,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    bucket: TokenBucket | None = None,
) -> FastAPI:
    provider_limiter = RateLimiter(settings.max_concurrency, settings.requests_per_minute)
    store = store or VectorStore(settings.chroma_dir)
    embedder = embedder or build_embedder(settings, provider_limiter)
    llm = llm or build_llm(settings, provider_limiter)
    per_minute = settings.api_requests_per_minute
    bucket = bucket or TokenBucket(per_minute, burst=max(1, per_minute // 6))
    page = files("repoqa").joinpath("static/index.html").read_text(encoding="utf-8")
    retrievers: dict[str, Retriever] = {}
    retrievers_lock = threading.Lock()

    def retriever_for(repo: str) -> Retriever:
        """Cached per repo (it loads all chunks for BM25); rebuilt after a re-index."""
        current = store.collection_id(repo)
        with retrievers_lock:
            cached = retrievers.get(repo)
            if cached is None or cached.collection_id != current:
                retrievers[repo] = Retriever(store, repo, embedder)
            return retrievers[repo]

    app = FastAPI(title="repoqa", version="0.1.0")

    @app.middleware("http")
    async def rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        client = request.client.host if request.client else "unknown"
        wait = bucket.acquire(client)
        if wait > 0:
            return JSONResponse(
                {"detail": "Too many requests"},
                status_code=429,
                headers={"Retry-After": str(math.ceil(wait))},
            )
        return await call_next(request)

    @app.exception_handler(RepoQAError)
    async def handle_repoqa_error(request: Request, exc: RepoQAError) -> JSONResponse:
        status = ERROR_STATUS.get(type(exc), 400)
        logger.error("%s %s -> %d: %s", request.method, request.url.path, status, exc)
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "repos": store.list_repos(),
            "embedder": embedder.name,
            "llm": llm.name,
        }

    @app.post("/ask")
    def ask(body: AskRequest) -> AnswerResult:
        repo = resolve_repo(store, body.repo, settings.default_repo)
        return answer_question(
            body.question,
            retriever_for(repo),
            llm,
            top_k=body.top_k or settings.top_k,
            min_score=settings.min_score,
            flag_invalid_citations=settings.citation_mode == "flag",
        )

    return app

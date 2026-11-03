"""Hybrid retrieval: vector search + BM25, merged with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rank_bm25 import BM25Plus

from repoqa.embeddings import Embedder
from repoqa.errors import IndexMismatchError
from repoqa.models import RetrievedChunk, SearchResult
from repoqa.store import VectorStore
from repoqa.text import tokenize

Mode = Literal["hybrid", "vector", "bm25"]
MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid")
RRF_K = 60  # standard constant from Cormack et al. (2009); damps the weight of top ranks
CANDIDATES_PER_RESULT = 4


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    """score(doc) = sum over rankings of 1 / (k + rank), with rank starting at 1.

    Uses only ranks, so BM25 scores and cosine similarities never need to be on one scale.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Searches one indexed repo. Loads all chunks once to build the in-memory BM25 index."""

    def __init__(self, store: VectorStore, repo: str, embedder: Embedder) -> None:
        indexed_with = store.embedder_name(repo)
        if indexed_with != embedder.name:
            raise IndexMismatchError(
                f"{repo!r} was indexed with {indexed_with}, but the configured embedder is "
                f"{embedder.name}. Re-index, or set REPOQA_EMBEDDER to match."
            )
        self.collection_id = store.collection_id(repo)
        self._store = store
        self._repo = repo
        self._embedder = embedder
        self._chunks = {chunk.citation: chunk for chunk in store.chunks(repo)}
        self._ids = list(self._chunks)
        corpus = [tokenize(self._chunks[cid].search_text) for cid in self._ids]
        self._terms = [set(tokens) for tokens in corpus]
        # BM25+ keeps term weights positive even in tiny corpora, unlike classic Okapi BM25.
        self._bm25 = BM25Plus(corpus) if corpus else None

    def search(self, query: str, top_k: int, mode: Mode = "hybrid") -> SearchResult:
        n_candidates = max(top_k * CANDIDATES_PER_RESULT, 20)
        vector_hits = self._vector_search(query, n_candidates)
        bm25_hits = self._bm25_search(query, n_candidates)
        similarity = dict(vector_hits)

        if mode == "vector":
            ranked = vector_hits
        elif mode == "bm25":
            ranked = bm25_hits
        else:
            fused = reciprocal_rank_fusion([[c for c, _ in vector_hits], [c for c, _ in bm25_hits]])
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)

        hits = [
            RetrievedChunk(self._chunks[cid], score, similarity.get(cid))
            for cid, score in ranked[:top_k]
        ]
        best_similarity = vector_hits[0][1] if vector_hits else 0.0
        return SearchResult(hits, best_similarity, keyword_matches=len(bm25_hits))

    def _vector_search(self, query: str, n: int) -> list[tuple[str, float]]:
        vector = self._embedder.embed([query])[0]
        # Ignore ids we do not know: the index may have been swapped since we loaded it.
        return [hit for hit in self._store.query(self._repo, vector, n) if hit[0] in self._chunks]

    def _bm25_search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Top chunks by BM25, among chunks sharing at least one term with the query."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        matching = [
            (cid, float(score))
            for cid, score, terms in zip(self._ids, scores, self._terms, strict=True)
            if terms.intersection(tokens)
        ]
        return sorted(matching, key=lambda item: item[1], reverse=True)[:n]

from pathlib import Path

import pytest

from repoqa.embeddings import HashingEmbedder
from repoqa.errors import IndexMismatchError, IndexNotFoundError
from repoqa.models import Chunk
from repoqa.retrieval import Retriever, reciprocal_rank_fusion
from repoqa.store import VectorStore, resolve_repo


def test_rrf_rewards_documents_ranked_well_by_both_lists() -> None:
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert max(scores, key=scores.__getitem__) == "b"
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert set(scores) == {"a", "b", "c", "d"}


ALPHA = Chunk("a.py", 1, 5, "python", "alpha", "def alpha(): pass")
BETA = Chunk("b.md", 1, 3, "markdown", None, "# Beta")


def test_store_round_trips_chunks_and_similarities(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    store.replace("demo", "hashing-2", [ALPHA, BETA], [[1.0, 0.0], [0.0, 1.0]])
    assert store.list_repos() == ["demo"]
    assert store.embedder_name("demo") == "hashing-2"
    assert sorted(store.chunks("demo"), key=lambda c: c.path) == [ALPHA, BETA]
    hits = store.query("demo", [1.0, 0.0], n=10)
    assert [cid for cid, _ in hits] == ["a.py:1-5", "b.md:1-3"]
    assert hits[0][1] == pytest.approx(1.0)


def test_replace_swaps_in_a_new_collection(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    store.replace("demo", "hashing-2", [ALPHA, BETA], [[1.0, 0.0], [0.0, 1.0]])
    first_id = store.collection_id("demo")
    store.replace("demo", "hashing-2", [BETA], [[0.0, 1.0]])
    assert store.chunks("demo") == [BETA]  # old chunks are gone
    assert store.collection_id("demo") != first_id
    assert store.list_repos() == ["demo"]  # the temporary "-building" collection is hidden
    store.replace("demo", "hashing-2", [], [])
    assert store.query("demo", [1.0, 0.0], n=5) == []


def test_missing_index_lists_known_repos(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    store.replace("present", "hashing-2", [ALPHA], [[1.0, 0.0]])
    with pytest.raises(IndexNotFoundError, match="indexed repos: present"):
        store.chunks("absent")
    with pytest.raises(IndexNotFoundError):
        store.chunks("x")  # also for names Chroma would reject


def test_resolve_repo(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    with pytest.raises(IndexNotFoundError, match="indexed: none"):
        resolve_repo(store, None, None)
    store.replace("only", "hashing-2", [ALPHA], [[1.0, 0.0]])
    assert resolve_repo(store, None, None) == "only"
    assert resolve_repo(store, "explicit", "default") == "explicit"
    assert resolve_repo(store, None, "default") == "default"
    store.replace("second", "hashing-2", [ALPHA], [[1.0, 0.0]])
    with pytest.raises(IndexNotFoundError, match="indexed: only, second"):
        resolve_repo(store, None, None)


def test_bm25_works_on_a_tiny_corpus(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    embedder = HashingEmbedder()
    chunks = [ALPHA, BETA]
    store.replace("tiny", embedder.name, chunks, embedder.embed([c.search_text for c in chunks]))
    hits = Retriever(store, "tiny", embedder).search("alpha", top_k=2, mode="bm25").hits
    assert [h.chunk for h in hits] == [ALPHA]  # only chunks sharing a term are ranked


def test_retriever_refuses_a_different_embedder(indexed_sample: tuple[VectorStore, str]) -> None:
    store, repo = indexed_sample
    with pytest.raises(IndexMismatchError, match=r"hashing-512.*hashing-64"):
        Retriever(store, repo, HashingEmbedder(dim=64))


def test_retriever_modes_return_scored_chunks(retriever: Retriever) -> None:
    question = "How are user passwords hashed?"
    for mode in ("bm25", "vector", "hybrid"):
        result = retriever.search(question, top_k=3, mode=mode)
        assert len(result.hits) == 3
        assert any(hit.chunk.path == "bookshop/auth.py" for hit in result.hits)
        assert result.best_similarity > 0.2
    hybrid = retriever.search(question, top_k=3).hits
    assert hybrid[0].score >= hybrid[-1].score
    assert all(hit.similarity is not None or hit.bm25 is not None for hit in hybrid)


def test_stopword_only_question_retrieves_nothing_relevant(retriever: Retriever) -> None:
    assert retriever.search("the of a", top_k=2, mode="bm25").hits == []
    assert retriever.search("the of a", top_k=2).best_similarity < 0.05

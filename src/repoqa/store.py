"""ChromaDB (persistent, local) storage: one collection per indexed repository."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from repoqa.errors import IndexNotFoundError
from repoqa.models import Chunk

MetadataValue = str | int | float | bool
BUILDING_SUFFIX = "-building"


class VectorStore:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )

    def list_repos(self) -> list[str]:
        names = (collection.name for collection in self._client.list_collections())
        return sorted(name for name in names if not name.endswith(BUILDING_SUFFIX))

    def replace(
        self,
        repo: str,
        embedder_name: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> None:
        """Build the new index in a side collection, then swap it in under `repo`.

        Readers keep seeing the old index until the swap, and a failed build leaves it intact.
        """
        building = repo + BUILDING_SUFFIX
        self._drop(building)
        collection = self._client.create_collection(
            building,
            metadata={"embedder": embedder_name},
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        step = self._client.get_max_batch_size()
        for start in range(0, len(chunks), step):
            batch = chunks[start : start + step]
            collection.add(
                ids=[c.citation for c in batch],
                embeddings=list(vectors[start : start + step]),
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
            )
        self._drop(repo)
        collection.modify(name=repo)

    def collection_id(self, repo: str) -> str:
        """Changes every time `repo` is re-indexed; used to refresh cached retrievers."""
        return str(self._collection(repo).id)

    def embedder_name(self, repo: str) -> str:
        metadata = self._collection(repo).metadata or {}
        return str(metadata.get("embedder", "unknown"))

    def query(self, repo: str, vector: list[float], n: int) -> list[tuple[str, float]]:
        """Nearest chunks as (citation, cosine similarity), best first."""
        collection = self._collection(repo)
        n = min(n, collection.count())
        if n == 0:
            return []
        result = collection.query(query_embeddings=[vector], n_results=n, include=["distances"])
        distances = (result["distances"] or [[]])[0]
        return [(cid, 1.0 - dist) for cid, dist in zip(result["ids"][0], distances, strict=True)]

    def chunks(self, repo: str) -> list[Chunk]:
        result = self._collection(repo).get(include=["documents", "metadatas"])
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        return [_from_metadata(m, d) for m, d in zip(metadatas, documents, strict=True)]

    def _drop(self, name: str) -> None:
        if name in (collection.name for collection in self._client.list_collections()):
            self._client.delete_collection(name)

    def _collection(self, repo: str) -> chromadb.Collection:
        known = self.list_repos()
        if repo not in known:
            raise IndexNotFoundError(
                f"No index named {repo!r} (indexed repos: {', '.join(known) or 'none'}). "
                "Run `repoqa index` first."
            )
        return self._client.get_collection(repo, embedding_function=None)


def resolve_repo(store: VectorStore, requested: str | None, default: str | None) -> str:
    """Pick the repo to query: explicit > configured default > the only indexed repo."""
    chosen = requested or default
    if chosen:
        return chosen
    repos = store.list_repos()
    if len(repos) == 1:
        return repos[0]
    known = ", ".join(repos) or "none"
    raise IndexNotFoundError(f"Choose a repo with --repo / REPOQA_REPO (indexed: {known})")


def _to_metadata(chunk: Chunk) -> dict[str, MetadataValue]:
    return {
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
        "symbol": chunk.symbol or "",  # Chroma metadata cannot hold None
    }


def _from_metadata(metadata: Mapping[str, Any], text: str) -> Chunk:
    return Chunk(
        path=metadata["path"],
        start_line=int(metadata["start_line"]),
        end_line=int(metadata["end_line"]),
        language=metadata["language"],
        symbol=metadata["symbol"] or None,
        text=text,
    )

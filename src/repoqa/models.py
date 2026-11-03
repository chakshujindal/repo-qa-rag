"""Core data types shared across indexing, retrieval and answering."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of one file, with the metadata needed to cite it."""

    path: str
    start_line: int
    end_line: int
    language: str
    symbol: str | None
    text: str

    @property
    def citation(self) -> str:
        """`path:start-end`. Unique per chunk, so it doubles as the vector store id."""
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def search_text(self) -> str:
        """Text used for both embeddings and BM25: path and symbol help name-based questions."""
        header = f"{self.path} {self.symbol or ''}".strip()
        return f"{header}\n{self.text}"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus how it ranked. `score` is the fused (or single-mode) ranking score;
    `similarity` is the cosine similarity, if the chunk was among the vector candidates."""

    chunk: Chunk
    score: float
    similarity: float | None = None


@dataclass(frozen=True)
class SearchResult:
    hits: list[RetrievedChunk]
    best_similarity: float  # best cosine similarity among all vector candidates
    keyword_matches: int  # chunks sharing at least one keyword with the question (capped)


class SourceRef(BaseModel):
    """One retrieved source as shown to users (CLI table, API response)."""

    id: int
    citation: str
    symbol: str | None
    score: float
    similarity: float | None
    cited: bool = False  # True if a verified citation in the answer points inside it


class AnswerResult(BaseModel):
    question: str
    answer: str
    answered: bool
    citations: list[str]
    rejected_citations: list[str]
    sources: list[SourceRef]
    best_similarity: float

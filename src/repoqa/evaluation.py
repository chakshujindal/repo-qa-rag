"""Retrieval evaluation: hit@k and MRR for each retrieval mode over a questions file."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from repoqa.errors import EvalSpecError
from repoqa.models import Chunk
from repoqa.retrieval import MODES, Mode, Retriever


class Expected(BaseModel):
    """A chunk that counts as a correct retrieval: same path, and same symbol if given."""

    path: str
    symbol: str | None = None

    def matches(self, chunk: Chunk) -> bool:
        if chunk.path != self.path:
            return False
        if self.symbol is None:
            return True
        # A large class is split into `Class.method` chunks; any of them counts for `Class`.
        return chunk.symbol == self.symbol or (chunk.symbol or "").startswith(self.symbol + ".")


class EvalQuestion(BaseModel):
    question: str
    expected: list[Expected] = Field(min_length=1)


class EvalSpec(BaseModel):
    repo: str  # path to the repo to index, relative to the questions file
    k: int = Field(default=5, ge=1)
    questions: list[EvalQuestion] = Field(min_length=1)


@dataclass(frozen=True)
class ModeScore:
    mode: Mode
    hit_at_1: float
    hit_at_3: float
    hit_at_k: float
    mrr: float
    ranks: list[int | None]  # rank of the first relevant chunk per question (None = missed)


def load_spec(path: Path) -> tuple[EvalSpec, Path]:
    """Parse the questions file; returns the spec and the absolute repo path."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        spec = EvalSpec.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise EvalSpecError(f"Invalid eval file {path}: {exc}") from exc
    return spec, (path.parent / spec.repo).resolve()


def first_relevant_rank(retrieved: Sequence[Chunk], expected: Sequence[Expected]) -> int | None:
    for rank, chunk in enumerate(retrieved, start=1):
        if any(e.matches(chunk) for e in expected):
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def evaluate(
    spec: EvalSpec, retriever: Retriever, modes: Sequence[Mode] = MODES
) -> list[ModeScore]:
    """Score each retrieval mode on the same questions; ranks are cut off at k."""
    scores = []
    for mode in modes:
        ranks = []
        for item in spec.questions:
            hits = retriever.search(item.question, spec.k, mode).hits
            ranks.append(first_relevant_rank([h.chunk for h in hits], item.expected))
        scores.append(
            ModeScore(
                mode=mode,
                hit_at_1=hit_rate(ranks, 1),
                hit_at_3=hit_rate(ranks, min(3, spec.k)),
                hit_at_k=hit_rate(ranks, spec.k),
                mrr=mean_reciprocal_rank(ranks),
                ranks=ranks,
            )
        )
    return scores

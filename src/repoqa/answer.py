"""Turn a question into a cited answer: retrieve, gate, prompt, validate, post-check."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ValidationError, model_validator

from repoqa.citations import Citation, check_citations, is_supported
from repoqa.errors import LLMOutputError
from repoqa.llm import ChatMessage, LLMClient
from repoqa.models import AnswerResult, RetrievedChunk, SearchResult, SourceRef
from repoqa.retrieval import Retriever

IDK_ANSWER = "I don't know based on this repo."

SYSTEM_PROMPT = """\
You answer questions about a software repository using ONLY the numbered sources provided.

Rules:
- Every factual sentence must end with a citation in square brackets, using a source's \
cite value, e.g. [src/app.py:12-40]. You may cite a narrower line range inside a source.
- Never cite a file or line range that is not inside one of the sources.
- If the sources do not contain the answer, set "insufficient_context" to true.
- Reply with a single JSON object and nothing else:
  {"answer": "<markdown answer with citations>", "insufficient_context": false}
"""

REPAIR_PROMPT = """\
Your previous reply was not valid. Error: {error}
Reply again with ONLY the JSON object: {{"answer": "...", "insufficient_context": false}}"""


class LLMAnswer(BaseModel):
    """The structured reply we require from the LLM."""

    answer: str = ""
    insufficient_context: bool = False

    @model_validator(mode="after")
    def _answer_required(self) -> LLMAnswer:
        if not self.insufficient_context and not self.answer.strip():
            raise ValueError("'answer' must be non-empty unless insufficient_context is true")
        return self


def build_prompt(question: str, hits: Sequence[RetrievedChunk]) -> str:
    """Numbered sources, each tagged with the exact citation the model should use.

    FakeLLM parses this exact `<source ...>` format (see `llm._SOURCE_BLOCK`).
    """
    blocks = [
        f'<source id="{i}" cite="{hit.chunk.citation}" symbol="{hit.chunk.symbol or ""}">\n'
        f"{hit.chunk.text}\n</source>"
        for i, hit in enumerate(hits, start=1)
    ]
    return "\n\n".join([*blocks, f"Question: {question}"])


def extract_json(raw: str) -> str:
    """The outermost `{...}` in the reply, dropping code fences or chatter around it."""
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end > start else raw


def parse_llm_answer(raw: str) -> LLMAnswer:
    return LLMAnswer.model_validate_json(extract_json(raw))


def ask_llm(llm: LLMClient, prompt: str) -> LLMAnswer:
    """Ask for structured output; on invalid output, send one repair request, then fail."""
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    raw = llm.complete(SYSTEM_PROMPT, messages)
    try:
        return parse_llm_answer(raw)
    except ValidationError as first_error:
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": REPAIR_PROMPT.format(error=_describe(first_error))},
        ]
    repaired = llm.complete(SYSTEM_PROMPT, messages)
    try:
        return parse_llm_answer(repaired)
    except ValidationError as exc:
        raise LLMOutputError(
            f"{llm.name} returned invalid JSON twice; last error: {_describe(exc)}"
        ) from exc


def has_evidence(result: SearchResult, min_score: float) -> bool:
    """Answer only if both retrievers found something: a close enough vector match AND at
    least one chunk sharing a keyword. Dense vectors always return *some* nearest neighbour,
    so similarity alone lets off-topic questions through."""
    return bool(result.hits) and result.best_similarity >= min_score and result.keyword_matches > 0


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    *,
    top_k: int,
    min_score: float,
    flag_invalid_citations: bool = False,
) -> AnswerResult:
    result = retriever.search(question, top_k)
    if not has_evidence(result, min_score):
        return _unanswered(question, result)

    reply = ask_llm(llm, build_prompt(question, result.hits))
    if reply.insufficient_context:
        return _unanswered(question, result)

    chunks = [hit.chunk for hit in result.hits]
    check = check_citations(reply.answer, chunks, flag=flag_invalid_citations)
    if not check.valid:  # nothing in the answer is backed by retrieved code
        return _unanswered(question, result, rejected=check.invalid)
    return AnswerResult(
        question=question,
        answer=check.text,
        answered=True,
        citations=[str(c) for c in check.valid],
        rejected_citations=[str(c) for c in check.invalid],
        sources=source_refs(result.hits, check.valid),
        best_similarity=result.best_similarity,
    )


def source_refs(hits: Sequence[RetrievedChunk], cited: Sequence[Citation] = ()) -> list[SourceRef]:
    return [
        SourceRef(
            id=i,
            citation=hit.chunk.citation,
            symbol=hit.chunk.symbol,
            score=round(hit.score, 4),
            similarity=None if hit.similarity is None else round(hit.similarity, 4),
            cited=any(is_supported(c, [hit.chunk]) for c in cited),
        )
        for i, hit in enumerate(hits, start=1)
    ]


def _unanswered(
    question: str, result: SearchResult, rejected: Sequence[Citation] = ()
) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=IDK_ANSWER,
        answered=False,
        citations=[],
        rejected_citations=[str(c) for c in rejected],
        sources=source_refs(result.hits),
        best_similarity=result.best_similarity,
    )


def _describe(error: ValidationError) -> str:
    first = error.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "reply"
    return f"{location}: {first['msg']}"[:300]

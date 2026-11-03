import json

import pytest

from repoqa.answer import (
    IDK_ANSWER,
    answer_question,
    ask_llm,
    build_prompt,
    extract_json,
    has_evidence,
    parse_llm_answer,
)
from repoqa.errors import LLMOutputError
from repoqa.llm import FakeLLM
from repoqa.models import Chunk, RetrievedChunk, SearchResult
from repoqa.retrieval import Retriever

QUESTION = "How are user passwords hashed before being stored?"


def reply(answer: str, insufficient: bool = False) -> str:
    return json.dumps({"answer": answer, "insufficient_context": insufficient})


def test_prompt_numbers_sources_with_exact_citations() -> None:
    hits = [RetrievedChunk(Chunk("a.py", 3, 9, "python", "run", "def run(): ..."), 0.5)]
    prompt = build_prompt("why?", hits)
    assert prompt.startswith('<source id="1" cite="a.py:3-9" symbol="run">\ndef run(): ...')
    assert prompt.endswith("Question: why?")


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer": "x"}',
        '```json\n{"answer": "x"}\n```',
        'Sure! Here it is: {"answer": "x"} Hope that helps.',
    ],
)
def test_extract_json_tolerates_fences_and_chatter(raw: str) -> None:
    assert parse_llm_answer(raw).answer == "x"


def test_extract_json_without_braces_returns_text() -> None:
    assert extract_json("no json here") == "no json here"


def test_one_repair_retry_then_success() -> None:
    llm = FakeLLM(scripted=["not json at all", reply("Fixed [a.py:1-2].")])
    assert ask_llm(llm, "prompt").answer == "Fixed [a.py:1-2]."
    repair_turn = llm.calls[1]
    assert [m["role"] for m in repair_turn] == ["user", "assistant", "user"]
    assert "Invalid JSON" in repair_turn[2]["content"]


def test_empty_answer_without_insufficient_flag_is_invalid() -> None:
    llm = FakeLLM(scripted=[reply(""), reply("", insufficient=True)])
    assert ask_llm(llm, "prompt").insufficient_context is True
    assert "must be non-empty" in llm.calls[1][2]["content"]


def test_fails_clearly_after_second_invalid_reply() -> None:
    llm = FakeLLM(scripted=["oops", '{"answer": 42}'])
    with pytest.raises(LLMOutputError, match="invalid JSON twice; last error: answer"):
        ask_llm(llm, "prompt")


def test_end_to_end_answer_cites_retrieved_code(retriever: Retriever) -> None:
    result = answer_question(QUESTION, retriever, FakeLLM(), top_k=4, min_score=0.15)
    assert result.answered
    assert result.citations and result.rejected_citations == []
    retrieved = {source.citation for source in result.sources}
    assert set(result.citations) <= retrieved
    assert any(c.startswith("bookshop/auth.py") for c in result.citations)


def test_hallucinated_citation_is_stripped(retriever: Retriever) -> None:
    top = retriever.search(QUESTION, 4).hits[0].chunk.citation
    llm = FakeLLM(scripted=[reply(f"Real [{top}]. Invented [bookshop/crypto.py:1-9].")])
    result = answer_question(QUESTION, retriever, llm, top_k=4, min_score=0.15)
    assert result.answer == f"Real [{top}]. Invented."
    assert result.rejected_citations == ["bookshop/crypto.py:1-9"]


def test_flag_mode_keeps_marked_citation(retriever: Retriever) -> None:
    top = retriever.search(QUESTION, 4).hits[0].chunk.citation
    llm = FakeLLM(scripted=[reply(f"Real [{top}]. Invented [nope.py:1-2].")])
    result = answer_question(
        QUESTION, retriever, llm, top_k=4, min_score=0.15, flag_invalid_citations=True
    )
    assert result.answer == f"Real [{top}]. Invented [nope.py:1-2 (unverified)]."


def test_answer_without_any_verified_citation_becomes_i_dont_know(retriever: Retriever) -> None:
    llm = FakeLLM(scripted=[reply("It deletes the database [ghost.py:1-9]. Trust me.")])
    result = answer_question(QUESTION, retriever, llm, top_k=4, min_score=0.15)
    assert result.answer == IDK_ANSWER and not result.answered
    assert result.rejected_citations == ["ghost.py:1-9"]


def test_sources_are_marked_cited_by_range_containment(retriever: Retriever) -> None:
    top = retriever.search(QUESTION, 4).hits[0].chunk
    narrower = f"{top.path}:{top.start_line}-{top.start_line}"
    llm = FakeLLM(scripted=[reply(f"Narrow [{narrower}].")])
    result = answer_question(QUESTION, retriever, llm, top_k=4, min_score=0.15)
    assert result.sources[0].citation == top.citation
    assert result.sources[0].cited is True
    assert [s.cited for s in result.sources[1:]] == [False] * 3


def test_low_retrieval_score_means_i_dont_know(retriever: Retriever) -> None:
    llm = FakeLLM()
    result = answer_question("What is the capital of France?", retriever, llm, top_k=4,
                             min_score=0.15)  # fmt: skip
    assert result.answer == IDK_ANSWER
    assert not result.answered and result.citations == []
    assert len(llm.calls) == 0  # the LLM is never called


@pytest.mark.parametrize(
    ("similarity", "keyword_matches", "expected"),
    [(0.40, 3, True), (0.10, 3, False), (0.40, 0, False)],
)
def test_evidence_needs_similarity_and_a_shared_keyword(
    similarity: float, keyword_matches: int, expected: bool
) -> None:
    hit = RetrievedChunk(Chunk("a.py", 1, 2, "python", None, "x"), 0.03, similarity)
    result = SearchResult([hit], similarity, keyword_matches)
    assert has_evidence(result, min_score=0.15) is expected
    assert has_evidence(SearchResult([], similarity, keyword_matches), 0.15) is False


def test_llm_can_report_insufficient_context(retriever: Retriever) -> None:
    llm = FakeLLM(scripted=[reply("", insufficient=True)])
    result = answer_question(QUESTION, retriever, llm, top_k=4, min_score=0.15)
    assert result.answer == IDK_ANSWER and not result.answered
    assert result.sources  # sources are still shown for transparency

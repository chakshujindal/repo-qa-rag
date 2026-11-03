from pathlib import Path

import pytest

from repoqa.errors import EvalSpecError
from repoqa.evaluation import (
    EvalQuestion,
    EvalSpec,
    Expected,
    evaluate,
    first_relevant_rank,
    hit_rate,
    load_spec,
    mean_reciprocal_rank,
)
from repoqa.models import Chunk
from repoqa.retrieval import Retriever
from tests.helpers import FIXTURES, SAMPLE_REPO


def chunk(path: str, symbol: str | None) -> Chunk:
    return Chunk(path, 1, 2, "python", symbol, "")


def test_metrics_on_known_ranks() -> None:
    ranks = [1, 2, None, 4]
    assert hit_rate(ranks, 1) == 0.25
    assert hit_rate(ranks, 3) == 0.5
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_expected_matching_rules() -> None:
    retrieved = [chunk("a.py", "other"), chunk("a.py", "Gateway.charge"), chunk("b.md", None)]
    assert first_relevant_rank(retrieved, [Expected(path="a.py", symbol="Gateway")]) == 2
    assert first_relevant_rank(retrieved, [Expected(path="b.md")]) == 3
    assert first_relevant_rank(retrieved, [Expected(path="c.py")]) is None


def test_load_spec_resolves_repo_relative_to_file() -> None:
    spec, repo = load_spec(FIXTURES / "questions.yaml")
    assert repo == SAMPLE_REPO.resolve()
    assert spec.k == 5 and len(spec.questions) >= 10


@pytest.mark.parametrize("content", ["questions: []\nrepo: x\n", "repo: [unclosed\n"])
def test_invalid_spec_is_a_clear_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(content)
    with pytest.raises(EvalSpecError, match="Invalid eval file"):
        load_spec(path)


def test_evaluate_scores_every_mode(retriever: Retriever) -> None:
    spec = EvalSpec(
        repo=".",
        k=3,
        questions=[
            EvalQuestion(
                question="How are user passwords hashed?",
                expected=[Expected(path="bookshop/auth.py", symbol="hash_password")],
            ),
            EvalQuestion(question="zebra giraffe", expected=[Expected(path="nowhere.py")]),
        ],
    )
    scores = evaluate(spec, retriever)
    assert [s.mode for s in scores] == ["bm25", "vector", "hybrid"]
    # Pinned ranks: a change here means retrieval behaviour changed, so look at it on purpose.
    assert [s.ranks for s in scores] == [[1, None], [3, None], [1, None]]
    hybrid = scores[2]
    assert (hybrid.hit_at_1, hybrid.hit_at_3, hybrid.mrr) == (0.5, 0.5, 0.5)


def test_fixture_eval_numbers_match_the_readme(retriever: Retriever) -> None:
    spec, _ = load_spec(FIXTURES / "questions.yaml")
    scores = {s.mode: s for s in evaluate(spec, retriever)}
    assert round(scores["bm25"].mrr, 3) == 0.818
    assert round(scores["vector"].mrr, 3) == 0.619
    assert round(scores["hybrid"].mrr, 3) == 0.752
    assert (scores["hybrid"].hit_at_3, scores["hybrid"].hit_at_k) == (13 / 14, 1.0)

from pathlib import Path

import pytest
from typer.testing import CliRunner

from repoqa import cli
from repoqa.llm import FakeLLM
from tests.helpers import FIXTURES, SAMPLE_REPO

runner = CliRunner()


def test_index_then_ask_with_citations() -> None:
    indexed = runner.invoke(cli.app, ["index", str(SAMPLE_REPO), "--name", "shop"])
    assert indexed.exit_code == 0, indexed.output
    assert "Indexed shop" in indexed.output
    assert "11 files" in indexed.output and "(skipped: none)" in indexed.output

    asked = runner.invoke(cli.app, ["ask", "How is a discount code applied at checkout?"])
    assert asked.exit_code == 0, asked.output
    assert "bookshop/cart.py" in asked.output
    assert "Retrieved sources" in asked.output


def test_ask_off_topic_says_i_dont_know() -> None:
    runner.invoke(cli.app, ["index", str(SAMPLE_REPO), "--name", "shop"])
    result = runner.invoke(cli.app, ["ask", "--repo", "shop", "--no-sources", "Capital of France?"])
    assert result.exit_code == 0
    assert "I don't know based on this repo." in result.output
    assert "Retrieved sources" not in result.output


def test_ask_reports_unbacked_citations_and_escapes_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner.invoke(cli.app, ["index", str(SAMPLE_REPO), "--name", "shop"])
    reply = '{"answer": "Real [bookshop/auth.py:11-15]. Made up [bookshop/nope.py:1-3]."}'
    monkeypatch.setattr(cli, "build_llm", lambda settings, limiter: FakeLLM(scripted=[reply]))
    result = runner.invoke(cli.app, ["ask", "How are user passwords hashed [/salted]?"])
    assert result.exit_code == 0, result.output
    assert "How are user passwords hashed [/salted]?" in result.output
    assert "1 citation(s) not backed by retrieved code: bookshop/nope.py:1-3" in result.output


def test_eval_prints_metrics_table() -> None:
    result = runner.invoke(cli.app, ["eval", str(FIXTURES / "questions.yaml"), "--details"])
    assert result.exit_code == 0, result.output
    for text in ["Retrieval eval", "hit@1", "MRR@5", "bm25", "vector", "hybrid", "Rank of"]:
        assert text in result.output


def test_errors_exit_1_with_one_message(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["--verbose", "index", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert result.output.count("index failed: Not a directory") == 1
    result = runner.invoke(cli.app, ["ask", "--repo", "nope", "anything"])
    assert result.exit_code == 1
    assert "ask failed: No index named 'nope'" in result.output


def test_bad_config_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOQA_LLM", "gpt")
    result = runner.invoke(cli.app, ["ask", "anything"])
    assert result.exit_code == 1
    assert "ask failed: REPOQA_LLM must be one of" in result.output


def test_serve_builds_the_app_and_runs_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: calls.append((host, port)))
    result = runner.invoke(cli.app, ["serve", "--port", "9001"])
    assert result.exit_code == 0, result.output
    assert calls == [("127.0.0.1", 9001)]

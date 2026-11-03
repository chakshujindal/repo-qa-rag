from pathlib import Path

import pytest

from repoqa.config import Settings
from repoqa.errors import ConfigError


def test_defaults_run_offline() -> None:
    settings = Settings.from_env({})
    assert (settings.embedder, settings.llm) == ("hashing", "fake")
    assert settings.anthropic_model == "claude-sonnet-5-5"
    assert settings.chroma_dir == Path(".repoqa/chroma")
    assert settings.clone_dir == Path(".repoqa/repos")


def test_reads_and_converts_env_values() -> None:
    settings = Settings.from_env(
        {"REPOQA_LLM": "anthropic", "REPOQA_TOP_K": "3", "REPOQA_MIN_SCORE": "0.4",
         "REPOQA_REPO": "demo", "REPOQA_HOME": "/tmp/x"}
    )  # fmt: skip
    assert (settings.llm, settings.top_k, settings.min_score) == ("anthropic", 3, 0.4)
    assert settings.default_repo == "demo"
    assert settings.home == Path("/tmp/x")


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"REPOQA_LLM": "gpt"}, "REPOQA_LLM must be one of"),
        ({"REPOQA_EMBEDDER": "bert"}, "REPOQA_EMBEDDER must be one of"),
        ({"REPOQA_CITATION_MODE": "drop"}, "REPOQA_CITATION_MODE"),
        ({"REPOQA_TOP_K": "many"}, "must be an integer"),
        ({"REPOQA_MIN_SCORE": "high"}, "must be a number"),
        ({"REPOQA_TOP_K": "0"}, "REPOQA_TOP_K must be >= 1"),
        ({"REPOQA_TIMEOUT_S": "0"}, "REPOQA_TIMEOUT_S > 0"),
    ],
)
def test_rejects_invalid_values(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)

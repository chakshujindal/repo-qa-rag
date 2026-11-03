"""Settings loaded from environment variables (see `.env.example`)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from repoqa.errors import ConfigError

EMBEDDERS = ("hashing", "openai")
LLMS = ("fake", "openai", "anthropic")
CITATION_MODES = ("strip", "flag")


@dataclass(frozen=True)
class Settings:
    home: Path = Path(".repoqa")
    embedder: str = "hashing"
    llm: str = "fake"
    openai_model: str = "gpt-6.1-sol"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_model: str = "claude-sonnet-5-5"
    top_k: int = 6
    min_score: float = 0.15
    citation_mode: str = "strip"
    max_file_bytes: int = 200_000
    timeout_s: float = 60.0
    max_retries: int = 4
    max_concurrency: int = 4
    requests_per_minute: int = 120
    api_requests_per_minute: int = 30
    default_repo: str | None = None

    @property
    def chroma_dir(self) -> Path:
        return self.home / "chroma"

    @property
    def clone_dir(self) -> Path:
        return self.home / "repos"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from `env` (defaults to `os.environ`), validating every value."""
        env = os.environ if env is None else env
        defaults = cls()
        settings = cls(
            home=Path(env.get("REPOQA_HOME", str(defaults.home))),
            embedder=env.get("REPOQA_EMBEDDER", defaults.embedder),
            llm=env.get("REPOQA_LLM", defaults.llm),
            openai_model=env.get("OPENAI_MODEL", defaults.openai_model),
            openai_embedding_model=env.get(
                "OPENAI_EMBEDDING_MODEL", defaults.openai_embedding_model
            ),
            anthropic_model=env.get("ANTHROPIC_MODEL", defaults.anthropic_model),
            top_k=_int(env, "REPOQA_TOP_K", defaults.top_k),
            min_score=_float(env, "REPOQA_MIN_SCORE", defaults.min_score),
            citation_mode=env.get("REPOQA_CITATION_MODE", defaults.citation_mode),
            max_file_bytes=_int(env, "REPOQA_MAX_FILE_BYTES", defaults.max_file_bytes),
            timeout_s=_float(env, "REPOQA_TIMEOUT_S", defaults.timeout_s),
            max_retries=_int(env, "REPOQA_MAX_RETRIES", defaults.max_retries),
            max_concurrency=_int(env, "REPOQA_MAX_CONCURRENCY", defaults.max_concurrency),
            requests_per_minute=_int(env, "REPOQA_RPM", defaults.requests_per_minute),
            api_requests_per_minute=_int(env, "REPOQA_API_RPM", defaults.api_requests_per_minute),
            default_repo=env.get("REPOQA_REPO") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        _choice("REPOQA_EMBEDDER", self.embedder, EMBEDDERS)
        _choice("REPOQA_LLM", self.llm, LLMS)
        _choice("REPOQA_CITATION_MODE", self.citation_mode, CITATION_MODES)
        for name, value in [
            ("REPOQA_TOP_K", self.top_k),
            ("REPOQA_MAX_FILE_BYTES", self.max_file_bytes),
            ("REPOQA_MAX_CONCURRENCY", self.max_concurrency),
            ("REPOQA_RPM", self.requests_per_minute),
            ("REPOQA_API_RPM", self.api_requests_per_minute),
        ]:
            if value < 1:
                raise ConfigError(f"{name} must be >= 1, got {value}")
        if self.max_retries < 0 or self.timeout_s <= 0:
            raise ConfigError("REPOQA_MAX_RETRIES must be >= 0 and REPOQA_TIMEOUT_S > 0")


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc

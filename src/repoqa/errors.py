"""Exceptions raised by repoqa. The CLI and the HTTP server catch `RepoQAError` at the boundary."""


class RepoQAError(Exception):
    """Base class for every expected, user-facing failure."""


class ConfigError(RepoQAError):
    """An environment variable or option has an invalid value."""


class SourceError(RepoQAError):
    """A repository path or git URL could not be read or cloned."""


class IndexNotFoundError(RepoQAError):
    """The requested repository has not been indexed yet."""


class IndexMismatchError(RepoQAError):
    """The index was built with a different embedder than the one configured now."""


class ProviderError(RepoQAError):
    """An LLM or embedding provider call failed (after retries)."""


class LLMOutputError(RepoQAError):
    """The LLM did not return valid structured output, even after one repair attempt."""


class EvalSpecError(RepoQAError):
    """The evaluation questions file is malformed."""

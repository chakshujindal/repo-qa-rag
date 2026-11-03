"""Code-aware tokenizer shared by BM25 and the hashing embedder."""

from __future__ import annotations

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Question words and Python keywords that would otherwise dominate matches.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "with", "self", "def", "return", "none", "true", "false",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, expanding identifiers into their parts.

    `parseHttpRequest` -> parsehttprequest, parse, http, request
    `max_file_bytes`   -> max_file_bytes, max, file, byte
    """
    tokens: list[str] = []
    for identifier in _IDENTIFIER.findall(text):
        parts = [p.lower() for p in _CAMEL_PART.findall(identifier)]
        whole = identifier.lower()
        candidates = [whole, *parts] if len(parts) > 1 else [whole]
        tokens.extend(_normalize(t) for t in candidates if _keep(t))
    return tokens


def _keep(token: str) -> bool:
    return len(token) > 1 and token not in STOPWORDS


def _normalize(token: str) -> str:
    """Naive plural folding so `chunks` matches `chunk`."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token

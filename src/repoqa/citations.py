"""Post-check LLM citations: every `[path:start-end]` must point inside a retrieved chunk."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from repoqa.models import Chunk

# A bracket and the spaces before it (removed together when a citation is stripped).
_BRACKET = re.compile(r"(?P<space>[ \t]*)\[(?P<body>[^\[\]\n]+)\]")
# One citation: `path:12`, `path:12-40`, also `path:L12-L40` and en dashes.
_CITATION = re.compile(
    r"(?P<path>[^\s\[\]:,;]+):L?(?P<start>\d+)(?:\s*[-\u2013]\s*L?(?P<end>\d+))?"
)


@dataclass(frozen=True)
class Citation:
    path: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


@dataclass(frozen=True)
class CitationCheck:
    text: str  # the answer after removing (or flagging) unsupported citations
    valid: list[Citation]
    invalid: list[Citation]


def parse_bracket(body: str) -> list[Citation] | None:
    """Citations inside one bracket, e.g. `a.py:1-3, b.py:9`. None if it holds anything else."""
    citations = []
    for part in re.split(r"[,;]", body):
        match = _CITATION.fullmatch(part.strip())
        if match is None:
            return None
        start = int(match["start"])
        citations.append(Citation(match["path"], start, int(match["end"] or start)))
    return citations


def is_supported(citation: Citation, chunks: Sequence[Chunk]) -> bool:
    """True if the cited range is non-empty and lies inside one retrieved chunk."""
    return citation.start <= citation.end and any(
        chunk.path == citation.path
        and chunk.start_line <= citation.start
        and citation.end <= chunk.end_line
        for chunk in chunks
    )


def check_citations(answer: str, chunks: Sequence[Chunk], *, flag: bool = False) -> CitationCheck:
    """Validate each citation. Unsupported ones are removed, or marked if `flag` is set.

    Brackets that are not citations (e.g. markdown link text) are left untouched.
    """
    valid: list[Citation] = []
    invalid: list[Citation] = []

    def rewrite(match: re.Match[str]) -> str:
        citations = parse_bracket(match["body"])
        if citations is None:
            return match[0]
        kept = []
        for citation in citations:
            if is_supported(citation, chunks):
                _add_once(valid, citation)
                kept.append(str(citation))
            else:
                _add_once(invalid, citation)
                if flag:
                    kept.append(f"{citation} (unverified)")
        return f"{match['space']}[{', '.join(kept)}]" if kept else ""

    return CitationCheck(_BRACKET.sub(rewrite, answer), valid, invalid)


def _add_once(items: list[Citation], item: Citation) -> None:
    if item not in items:
        items.append(item)

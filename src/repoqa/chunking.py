"""Split files into citable chunks.

Python files are split along the AST: one chunk per top-level function, one per class (or one
per method when a class is large), and "module" chunks for the code in between. Everything else
is split into fixed-size line windows with overlap.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath

from repoqa.models import Chunk

WINDOW_LINES = 60
OVERLAP_LINES = 10
MAX_CHUNK_LINES = 120

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".dart": "dart",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Span:
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    symbol: str | None


def detect_language(path: str) -> str:
    name = PurePosixPath(path)
    if name.name in {"Makefile", "Dockerfile"}:
        return name.name.lower()
    return LANGUAGES.get(name.suffix.lower(), "text")


def split_lines(text: str) -> list[str]:
    """Split on "\n" only, so line numbers match editors and the Python parser.

    (`str.splitlines` also breaks on form feeds and Unicode separators.)
    """
    lines = text.split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def chunk_file(path: str, text: str) -> list[Chunk]:
    language = detect_language(path)
    if language == "python":
        return chunk_python(path, text)
    return chunk_lines(path, split_lines(text), language)


def chunk_python(path: str, text: str) -> list[Chunk]:
    """AST-aware chunks; falls back to line windows if the file does not parse."""
    lines = split_lines(text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):  # invalid, or too deeply nested
        return chunk_lines(path, lines, "python")
    spans = python_spans(tree, len(lines))
    chunks: list[Chunk] = []
    for span in spans:
        if _is_blank(lines, span):
            continue
        if span.end - span.start + 1 > MAX_CHUNK_LINES:
            body = lines[span.start - 1 : span.end]
            chunks.extend(chunk_lines(path, body, "python", span.symbol, first_line=span.start))
        else:
            chunks.append(_make_chunk(path, lines, span, "python"))
    return chunks


def python_spans(tree: ast.Module, line_count: int) -> list[Span]:
    """Definition spans plus the module-level gaps between them, sorted by line."""
    spans: list[Span] = []
    for node in tree.body:
        if isinstance(node, _DEFINITIONS):
            spans.append(Span(_first_line(node), _last_line(node), node.name))
        elif isinstance(node, ast.ClassDef):
            spans.extend(class_spans(node))
    spans.extend(Span(a, b, None) for a, b in gaps(1, line_count, spans))
    return sorted(spans, key=lambda s: s.start)


def class_spans(node: ast.ClassDef) -> list[Span]:
    """A small class is one chunk; a large one is split into methods plus the class body."""
    start, end = _first_line(node), _last_line(node)
    methods = [n for n in node.body if isinstance(n, _DEFINITIONS)]
    if end - start + 1 <= MAX_CHUNK_LINES or not methods:
        return [Span(start, end, node.name)]
    spans = [Span(_first_line(m), _last_line(m), f"{node.name}.{m.name}") for m in methods]
    spans.extend(Span(a, b, node.name) for a, b in gaps(start, end, spans))
    return spans


def gaps(start: int, end: int, spans: list[Span]) -> list[tuple[int, int]]:
    """Line ranges inside [start, end] not covered by any span."""
    covered = sorted((s.start, s.end) for s in spans)
    result: list[tuple[int, int]] = []
    cursor = start
    for span_start, span_end in covered:
        if span_start > cursor:
            result.append((cursor, span_start - 1))
        cursor = max(cursor, span_end + 1)
    if cursor <= end:
        result.append((cursor, end))
    return result


def chunk_lines(
    path: str,
    lines: list[str],
    language: str,
    symbol: str | None = None,
    *,
    first_line: int = 1,
    window: int = WINDOW_LINES,
    overlap: int = OVERLAP_LINES,
) -> list[Chunk]:
    """Fixed-size windows of `window` lines; consecutive windows share `overlap` lines."""
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    chunks: list[Chunk] = []
    offset = 0
    while offset < len(lines):
        piece = lines[offset : offset + window]
        span = Span(first_line + offset, first_line + offset + len(piece) - 1, symbol)
        if "".join(piece).strip():
            chunks.append(Chunk(path, span.start, span.end, language, symbol, "\n".join(piece)))
        if offset + window >= len(lines):
            break
        offset += window - overlap
    return chunks


def _make_chunk(path: str, lines: list[str], span: Span, language: str) -> Chunk:
    text = "\n".join(lines[span.start - 1 : span.end])
    return Chunk(path, span.start, span.end, language, span.symbol, text)


def _first_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Include decorators in the chunk."""
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _last_line(node: ast.stmt) -> int:
    return node.end_lineno or node.lineno


def _is_blank(lines: list[str], span: Span) -> bool:
    return not "".join(lines[span.start - 1 : span.end]).strip()

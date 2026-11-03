import textwrap

from repoqa.chunking import (
    MAX_CHUNK_LINES,
    Span,
    chunk_file,
    chunk_lines,
    detect_language,
    gaps,
)

PYTHON_SOURCE = textwrap.dedent(
    '''\
    """Module docstring."""
    import os

    LIMIT = 3


    @decorator
    def first(a):
        return a


    class Small:
        x = 1

        def method(self):
            return self.x


    async def second():
        return await thing()
    '''
)


def test_python_is_chunked_by_definition_with_symbols_and_line_ranges() -> None:
    chunks = chunk_file("pkg/mod.py", PYTHON_SOURCE)
    # Module-level code becomes its own chunk; blank gaps between definitions are dropped.
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        (None, 1, 6),
        ("first", 7, 9),
        ("Small", 12, 16),
        ("second", 19, 20),
    ]
    first = chunks[1]
    assert first.text.startswith("@decorator")  # decorators belong to the function
    assert first.language == "python"
    assert first.citation == "pkg/mod.py:7-9"


def test_large_class_is_split_into_methods_plus_class_body() -> None:
    body = "\n".join(f"        v{i} = {i}" for i in range(MAX_CHUNK_LINES))
    source = (
        f"class Big:\n    attr = 1\n\n    def run(self):\n{body}\n\n"
        "    def stop(self):\n        pass\n"
    )
    symbols = [(c.symbol, c.start_line) for c in chunk_file("big.py", source)]
    assert ("Big", 1) in symbols
    assert ("Big.run", 4) in symbols
    assert any(symbol == "Big.stop" for symbol, _ in symbols)


def test_oversized_function_is_windowed_but_keeps_its_symbol() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(MAX_CHUNK_LINES + 30))
    chunks = chunk_file("long.py", f"def long():\n{body}\n")
    assert len(chunks) > 1
    assert {c.symbol for c in chunks} == {"long"}
    assert chunks[0].start_line == 1


def test_unparseable_python_falls_back_to_line_windows() -> None:
    chunks = chunk_file("broken.py", "def broken(:\n    pass\n")
    assert len(chunks) == 1 and chunks[0].symbol is None
    deep = "x = " + "+".join(["a"] * 200_000) + "\n"  # too deep for the parser
    assert [c.symbol for c in chunk_file("deep.py", deep)] == [None]


def test_line_numbers_ignore_form_feeds_and_unicode_separators() -> None:
    source = "x = 1  # page\x0c break\n\ndef f():\n    return '\u2028'\n"
    chunks = chunk_file("ff.py", source)
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [(None, 1, 2), ("f", 3, 4)]


def test_line_windows_overlap_and_skip_blank_windows() -> None:
    lines = [f"line {i}" for i in range(1, 26)]
    chunks = chunk_lines("notes.md", lines, "markdown", window=10, overlap=3)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (8, 17), (15, 24), (22, 25)]
    assert chunk_lines("blank.txt", ["", "  "], "text") == []


def test_line_window_rejects_overlap_not_smaller_than_window() -> None:
    try:
        chunk_lines("a.txt", ["a"], "text", window=5, overlap=5)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_detect_language() -> None:
    assert detect_language("src/app.TS") == "typescript"
    assert detect_language("Makefile") == "makefile"
    assert detect_language("LICENSE") == "text"


def test_gaps_reports_uncovered_ranges() -> None:
    spans = [Span(3, 4, "a"), Span(7, 8, "b")]
    assert gaps(1, 10, spans) == [(1, 2), (5, 6), (9, 10)]
    assert gaps(3, 4, [Span(3, 4, "a")]) == []

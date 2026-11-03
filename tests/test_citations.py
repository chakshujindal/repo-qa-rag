from repoqa.citations import check_citations
from repoqa.models import Chunk

CHUNKS = [
    Chunk("app/auth.py", 10, 30, "python", "hash_password", "..."),
    Chunk("README.md", 1, 12, "markdown", None, "..."),
    Chunk("app/(shop)/page.tsx", 1, 40, "typescript", None, "..."),
]


def strings(citations: list) -> list[str]:
    return [str(c) for c in citations]


def test_valid_citations_are_kept_including_narrower_ranges() -> None:
    check = check_citations("Hashing [app/auth.py:10-30] uses PBKDF2 [app/auth.py:12-14].", CHUNKS)
    assert strings(check.valid) == ["app/auth.py:10-30", "app/auth.py:12-14"]
    assert check.invalid == []


def test_single_line_l_prefix_en_dash_and_parentheses_in_paths() -> None:
    en_dash = "\u2013"
    answer = f"[README.md:3] [README.md:L2-L4] [README.md:5{en_dash}6] [app/(shop)/page.tsx:9]"
    check = check_citations(answer, CHUNKS)
    assert strings(check.valid) == [
        "README.md:3-3", "README.md:2-4", "README.md:5-6", "app/(shop)/page.tsx:9-9",
    ]  # fmt: skip


def test_unsupported_citations_are_stripped_with_their_leading_space() -> None:
    answer = "Uses PBKDF2 [app/auth.py:10-30] and a pepper [app/auth.py:25-40]. Also [x.py:1-2]."
    check = check_citations(answer, CHUNKS)
    assert check.text == "Uses PBKDF2 [app/auth.py:10-30] and a pepper. Also."
    assert strings(check.valid) == ["app/auth.py:10-30"]
    assert strings(check.invalid) == ["app/auth.py:25-40", "x.py:1-2"]


def test_several_citations_in_one_bracket_are_checked_one_by_one() -> None:
    check = check_citations("Both [app/auth.py:10-12, ghost.py:5-9; README.md:1].", CHUNKS)
    assert check.text == "Both [app/auth.py:10-12, README.md:1-1]."
    assert strings(check.invalid) == ["ghost.py:5-9"]


def test_stripping_leaves_code_indentation_and_other_brackets_alone() -> None:
    answer = (
        "```python\n    if x:\n        return  [1]\n```\nSee [the docs](https://x.io) [ghost.py:1]"
    )
    check = check_citations(answer, CHUNKS)
    assert (
        check.text == "```python\n    if x:\n        return  [1]\n```\nSee [the docs](https://x.io)"
    )


def test_flag_mode_marks_instead_of_removing() -> None:
    check = check_citations("Pepper [app/auth.py:25-40].", CHUNKS, flag=True)
    assert check.text == "Pepper [app/auth.py:25-40 (unverified)]."


def test_reversed_range_is_invalid_and_duplicates_reported_once() -> None:
    check = check_citations(
        "[README.md:5-2] [README.md:5-2] [README.md:1-2] [README.md:1-2]", CHUNKS
    )
    assert strings(check.invalid) == ["README.md:5-2"]
    assert strings(check.valid) == ["README.md:1-2"]

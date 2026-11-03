from pathlib import Path

import pytest

from repoqa.embeddings import HashingEmbedder
from repoqa.errors import ProviderError, SourceError
from repoqa.indexer import MAX_LINE_CHARS, collect_chunks, index_repo
from repoqa.store import VectorStore
from tests.helpers import SAMPLE_REPO


class BrokenEmbedder(HashingEmbedder):
    def embed(self, texts: object) -> list[list[float]]:
        raise ProviderError("embedding API down")


def make_repo(root: Path, files: dict[str, str | bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = root / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    return root


def test_collect_chunks_covers_python_markdown_and_js() -> None:
    chunks, files, skipped = collect_chunks(SAMPLE_REPO, 200_000)
    assert {"bookshop/auth.py", "docs/deployment.md", "web/app.js"} <= {c.path for c in chunks}
    assert ("bookshop/auth.py", "hash_password") in {(c.path, c.symbol) for c in chunks}
    assert files == 11 and not skipped


def test_index_repo_skips_unreadable_and_minified_files(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path / "repo",
        {
            "ok.py": "def f():\n    return 1\n",
            "latin1.txt": "caf\xe9".encode("latin-1"),
            "bundle.js": "var a=1;" * (MAX_LINE_CHARS // 4),
        },
    )
    store = VectorStore(tmp_path / "chroma")

    stats = index_repo(repo, "tiny", store, HashingEmbedder(), 100_000)

    assert (stats.files, stats.chunks) == (1, 1)
    assert stats.skipped == {"unreadable": 1, "minified": 1}
    assert store.embedder_name("tiny") == "hashing-512"


def test_byte_order_mark_does_not_break_ast_chunking(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", {"bom.py": "\ufeffdef f():\n    return 1\n"})
    chunks, _, _ = collect_chunks(repo, 10_000)
    assert [(c.symbol, c.start_line) for c in chunks] == [("f", 1)]


def test_failed_reindex_keeps_the_previous_index(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    repo = make_repo(tmp_path / "repo", {"a.py": "def a():\n    pass\n"})
    index_repo(repo, "keep", store, HashingEmbedder(), 10_000)
    before = store.chunks("keep")

    (repo / "b.py").write_text("def b():\n    pass\n")
    with pytest.raises(ProviderError):
        index_repo(repo, "keep", store, BrokenEmbedder(), 10_000)
    assert store.chunks("keep") == before

    empty = make_repo(tmp_path / "empty", {})
    with pytest.raises(SourceError, match="No indexable"):
        index_repo(empty, "keep", store, HashingEmbedder(), 10_000)
    assert store.chunks("keep") == before

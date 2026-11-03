"""`repoqa index`: walk -> chunk -> embed -> store."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from repoqa.chunking import chunk_file
from repoqa.embeddings import Embedder
from repoqa.errors import SourceError
from repoqa.models import Chunk
from repoqa.store import VectorStore
from repoqa.walker import WalkStats, iter_text_files

logger = logging.getLogger(__name__)

MAX_LINE_CHARS = 2_000  # longer lines usually mean minified or generated files


@dataclass(frozen=True)
class IndexStats:
    repo: str
    root: Path
    files: int
    chunks: int
    skipped: Counter[str]  # reason -> number of files


def is_minified(text: str) -> bool:
    return any(len(line) > MAX_LINE_CHARS for line in text.split("\n"))


def collect_chunks(root: Path, max_file_bytes: int) -> tuple[list[Chunk], int, Counter[str]]:
    """Chunk every indexable file under `root`. Returns (chunks, files used, skip reasons)."""
    walk = WalkStats()
    chunks: list[Chunk] = []
    files = 0
    skipped: Counter[str] = Counter()
    for rel_path in iter_text_files(root, max_file_bytes, walk):
        try:
            text = (root / rel_path).read_text(encoding="utf-8-sig")  # -sig drops a BOM
        except (OSError, UnicodeDecodeError):
            logger.debug("skipping unreadable file %s", rel_path)
            skipped["unreadable"] += 1
            continue
        if is_minified(text):
            skipped["minified"] += 1
            continue
        files += 1
        chunks.extend(chunk_file(rel_path, text))
    return chunks, files, skipped + walk.skipped


def index_repo(
    root: Path, repo: str, store: VectorStore, embedder: Embedder, max_file_bytes: int
) -> IndexStats:
    """Rebuild the index for `repo` from the files under `root`.

    Embeds everything before touching the store, so a failure keeps the previous index.
    """
    chunks, files, skipped = collect_chunks(root, max_file_bytes)
    if not chunks:
        raise SourceError(f"No indexable text files found under {root}")
    vectors = embedder.embed([chunk.search_text for chunk in chunks])
    store.replace(repo, embedder.name, chunks, vectors)
    return IndexStats(repo=repo, root=root, files=files, chunks=len(chunks), skipped=skipped)

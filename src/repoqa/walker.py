"""Walk a repository, honouring .gitignore files and skipping binary or oversized files."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

# Ignored even without a .gitignore: VCS metadata, environments, caches, our own index.
DEFAULT_IGNORES = [
    ".git/",
    ".hg/",
    ".svn/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".repoqa/",
    # Never send likely secrets to an embedding API, even in folders without a .gitignore.
    ".env",
    ".env.*",
    "!.env.example",
    ".envrc",
    "*.pem",
    "*.key",
    "*.p12",
    "id_rsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    ".pypirc",
]
BINARY_SNIFF_BYTES = 8192


@dataclass
class WalkStats:
    files: int = 0
    skipped: Counter[str] = field(default_factory=Counter)  # reason -> count


@dataclass(frozen=True)
class _IgnoreRule:
    base: str  # directory (relative to the repo root, "" for root) that owns this .gitignore
    spec: pathspec.GitIgnoreSpec


def iter_text_files(root: Path, max_bytes: int, stats: WalkStats | None = None) -> Iterator[str]:
    """Yield repo-relative POSIX paths of text files worth indexing, in a stable order."""
    stats = stats if stats is not None else WalkStats()
    rules = [_IgnoreRule("", pathspec.GitIgnoreSpec.from_lines(DEFAULT_IGNORES))]
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        rules.extend(load_gitignore(rel_dir, Path(dirpath)))
        active = [r for r in rules if _owns(r.base, rel_dir)]
        dirnames[:] = sorted(d for d in dirnames if not is_ignored(active, _join(rel_dir, d) + "/"))
        for name in sorted(filenames):
            rel_path = _join(rel_dir, name)
            if is_ignored(active, rel_path):
                continue
            reason = skip_reason(Path(dirpath) / name, max_bytes)
            if reason:
                stats.skipped[reason] += 1
            else:
                stats.files += 1
                yield rel_path


def load_gitignore(rel_dir: str, abs_dir: Path) -> list[_IgnoreRule]:
    """Load the .gitignore in `abs_dir`, if any."""
    gitignore = abs_dir / ".gitignore"
    if not gitignore.is_file():
        return []
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return [_IgnoreRule(rel_dir, pathspec.GitIgnoreSpec.from_lines(lines))]


def is_ignored(rules: list[_IgnoreRule], rel_path: str) -> bool:
    """Apply rules from the root down; a deeper .gitignore overrides a shallower one."""
    ignored = False
    for rule in rules:
        local = rel_path[len(rule.base) + 1 :] if rule.base else rel_path
        result = rule.spec.check_file(local)
        if result.include is not None:
            ignored = result.include
    return ignored


def is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" in handle.read(BINARY_SNIFF_BYTES)


def skip_reason(path: Path, max_bytes: int) -> str | None:
    """Why `path` should not be indexed, or None if it should."""
    if path.is_symlink():  # could point outside the repo, e.g. at ~/.ssh
        return "symlink"
    if not path.is_file():  # sockets, fifos, ...
        return "special"
    try:
        if path.stat().st_size > max_bytes:
            return "large"
        return "binary" if is_binary(path) else None
    except OSError:  # e.g. permission denied
        return "unreadable"


def _owns(base: str, rel_dir: str) -> bool:
    return base == "" or rel_dir == base or rel_dir.startswith(base + "/")


def _join(rel_dir: str, name: str) -> str:
    return f"{rel_dir}/{name}" if rel_dir else name

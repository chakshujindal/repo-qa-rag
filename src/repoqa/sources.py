"""Resolve `repoqa index <source>`: a local directory, or a git URL that we shallow-clone."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from repoqa.errors import SourceError

GIT_URL_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://", "file://")
CLONE_TIMEOUT_S = 300


def is_git_url(source: str) -> bool:
    return source.startswith(GIT_URL_PREFIXES) or source.endswith(".git")


def repo_name(source: str) -> str:
    """`https://github.com/org/my-repo.git` -> `my-repo`; `./src/app` -> `app`."""
    tail = source.rstrip("/").replace(":", "/").split("/")[-1]
    return tail.removesuffix(".git") or "repo"


def collection_name(name: str) -> str:
    """Chroma collection names allow [a-zA-Z0-9._-], 3-512 chars, alphanumeric at both ends.

    We cap at 200 characters to leave room for the temporary "-building" suffix.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name)[:200].strip("-._")
    return (cleaned or "repo").ljust(3, "0")


def resolve_source(source: str, clone_dir: Path) -> Path:
    """Return a local directory for `source`, cloning (or refreshing) it if it is a URL."""
    if is_git_url(source):
        return shallow_clone(source, clone_dir)
    path = Path(source).expanduser().resolve()
    if not path.is_dir():
        raise SourceError(f"Not a directory: {source}")
    return path


def shallow_clone(url: str, clone_dir: Path) -> Path:
    """Clone `url` with depth 1 into the cache, or fetch the latest commit if already cached."""
    digest = hashlib.sha1(url.encode()).hexdigest()[:10]
    dest = clone_dir / f"{collection_name(repo_name(url))}-{digest}"
    if (dest / ".git").is_dir():
        _git(["fetch", "--depth", "1", "origin"], cwd=dest)
        _git(["reset", "--hard", "FETCH_HEAD"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # "--" stops a source string that starts with "-" from being read as a git option.
        _git(["clone", "--depth", "1", "--quiet", "--", url, str(dest)], cwd=dest.parent)
    return dest


def _git(args: list[str], cwd: Path) -> None:
    try:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise SourceError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SourceError(f"git {args[0]} failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError(f"git {args[0]} timed out after {CLONE_TIMEOUT_S}s") from exc

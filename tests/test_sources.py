import subprocess
from pathlib import Path

import pytest

from repoqa.errors import SourceError
from repoqa.sources import collection_name, is_git_url, repo_name, resolve_source


def make_git_repo(path: Path, content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text(content)
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=path, check=True)
    subprocess.run([*git, "add", "."], cwd=path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=path, check=True)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://github.com/org/my-repo.git", True),
        ("git@github.com:org/my-repo.git", True),
        ("file:///tmp/repo", True),
        ("./src/app", False),
        ("/abs/path", False),
    ],
)
def test_is_git_url(source: str, expected: bool) -> None:
    assert is_git_url(source) is expected


def test_repo_and_collection_names() -> None:
    assert repo_name("https://github.com/org/my-repo.git") == "my-repo"
    assert repo_name("git@github.com:org/tool.git") == "tool"
    assert repo_name("/work/project/") == "project"
    assert collection_name("my repo!") == "my-repo"
    assert collection_name("a") == "a00"
    assert collection_name("@@") == "repo"
    long_name = collection_name("a" * 199 + "-tail")
    assert len(long_name) == 199 and long_name.endswith("a")  # never ends with "-"


def test_local_directory_resolves_to_absolute_path(tmp_path: Path) -> None:
    assert resolve_source(str(tmp_path), tmp_path / "cache") == tmp_path.resolve()


def test_missing_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="Not a directory"):
        resolve_source(str(tmp_path / "nope"), tmp_path / "cache")


def test_shallow_clone_then_refresh(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    make_git_repo(upstream, "VERSION = 1\n")
    url = upstream.as_uri()

    clone = resolve_source(url, tmp_path / "cache")
    assert (clone / "app.py").read_text() == "VERSION = 1\n"
    depth = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=clone, capture_output=True, text=True
    )
    assert depth.stdout.strip() == "1"

    (upstream / "app.py").write_text("VERSION = 2\n")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e.com", "commit", "-qam", "v2"],
        cwd=upstream,
        check=True,
    )
    assert resolve_source(url, tmp_path / "cache") == clone
    assert (clone / "app.py").read_text() == "VERSION = 2\n"


def test_clone_failure_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="git clone failed"):
        resolve_source((tmp_path / "missing.git").as_uri(), tmp_path / "cache")


def test_missing_git_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SourceError, match="git is not installed"):
        resolve_source("https://example.com/repo.git", tmp_path / "cache")


def test_git_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(SourceError, match="timed out"):
        resolve_source("https://example.com/repo.git", tmp_path / "cache")

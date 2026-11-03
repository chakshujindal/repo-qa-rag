import os
from pathlib import Path

from repoqa.walker import WalkStats, iter_text_files


def write(root: Path, rel: str, content: str | bytes = "x = 1\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def test_respects_root_and_nested_gitignore(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "*.log\nbuild/\n")
    write(tmp_path, "app.py")
    write(tmp_path, "debug.log", "noise")
    write(tmp_path, "build/out.py")
    write(tmp_path, "pkg/.gitignore", "generated.py\n!keep.log\n")
    write(tmp_path, "pkg/generated.py")
    write(tmp_path, "pkg/keep.log", "kept by a deeper negation")
    write(tmp_path, "pkg/mod.py")

    files = list(iter_text_files(tmp_path, max_bytes=1_000))

    assert files == [".gitignore", "app.py", "pkg/.gitignore", "pkg/keep.log", "pkg/mod.py"]


def test_skips_default_dirs_binary_and_large_files(tmp_path: Path) -> None:
    write(tmp_path, ".git/config", "[core]")
    write(tmp_path, "node_modules/lib/index.js")
    write(tmp_path, ".venv/lib/site.py")
    write(tmp_path, "logo.png", b"\x89PNG\x00\x00binary")
    write(tmp_path, "big.txt", "y" * 2_000)
    write(tmp_path, "small.txt", "hello")
    write(tmp_path, ".env", "OPENAI_API_KEY=secret")
    write(tmp_path, ".env.local", "TOKEN=secret")
    write(tmp_path, ".env.example", "OPENAI_API_KEY=your-key")
    write(tmp_path, "certs/server.pem", "-----BEGIN PRIVATE KEY-----")
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    os.mkfifo(tmp_path / "pipe")
    stats = WalkStats()

    files = list(iter_text_files(tmp_path, max_bytes=1_000, stats=stats))

    assert files == [".env.example", "small.txt"]
    assert stats.files == 2
    assert stats.skipped == {"binary": 1, "large": 1, "symlink": 1, "special": 1}


def test_symlinks_are_never_followed_out_of_the_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write(outside, "id_key.txt", "PRIVATE KEY")
    repo = tmp_path / "repo"
    write(repo, "app.py")
    (repo / "notes.txt").symlink_to(outside / "id_key.txt")
    (repo / "linked_dir").symlink_to(outside, target_is_directory=True)

    assert list(iter_text_files(repo, max_bytes=1_000)) == ["app.py"]


def test_unreadable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    write(tmp_path, "ok.py")
    write(tmp_path, "locked.py")
    (tmp_path / "locked.py").chmod(0)
    stats = WalkStats()
    try:
        files = list(iter_text_files(tmp_path, max_bytes=1_000, stats=stats))
    finally:
        (tmp_path / "locked.py").chmod(0o644)
    assert files == ["ok.py"]
    assert stats.skipped == {"unreadable": 1}

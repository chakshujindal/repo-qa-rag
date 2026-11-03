"""Command-line interface: `repoqa index | ask | eval | serve`."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from repoqa.answer import answer_question
from repoqa.config import Settings
from repoqa.embeddings import build_embedder
from repoqa.errors import RepoQAError
from repoqa.evaluation import ModeScore, evaluate, load_spec
from repoqa.indexer import IndexStats, index_repo
from repoqa.llm import build_llm
from repoqa.models import AnswerResult
from repoqa.resilience import RateLimiter
from repoqa.retrieval import Retriever
from repoqa.sources import collection_name, is_git_url, repo_name, resolve_source
from repoqa.store import VectorStore, resolve_repo

app = typer.Typer(
    help="Ask questions about any codebase; answers cite the exact file and lines.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = logging.getLogger("repoqa")


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_path=False)],
        force=True,
    )


def provider_limiter(settings: Settings) -> RateLimiter:
    """One limiter shared by the embedder and the LLM of a command."""
    return RateLimiter(settings.max_concurrency, settings.requests_per_minute)


@contextmanager
def boundary(command: str) -> Iterator[None]:
    """The CLI's error boundary: log expected failures once, exit with status 1."""
    try:
        yield
    except RepoQAError as exc:
        logger.error("%s failed: %s", command, exc)
        raise typer.Exit(code=1) from exc


@app.command()
def index(
    source: Annotated[str, typer.Argument(help="Local path or git URL.")],
    name: Annotated[str | None, typer.Option(help="Index name (default: repo name).")] = None,
) -> None:
    """Index a repository: walk, chunk, embed and store it."""
    with boundary("index"):
        settings = Settings.from_env()
        limiter = provider_limiter(settings)
        root = resolve_source(source, settings.clone_dir)
        repo = collection_name(name or repo_name(source if is_git_url(source) else str(root)))
        stats = index_repo(
            root,
            repo,
            VectorStore(settings.chroma_dir),
            build_embedder(settings, limiter),
            settings.max_file_bytes,
        )
    print_index_stats(stats)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question about the code.")],
    repo: Annotated[str | None, typer.Option(help="Index name to query.")] = None,
    top_k: Annotated[int | None, typer.Option(min=1, max=50, help="Chunks to retrieve.")] = None,
    sources: Annotated[bool, typer.Option(help="Show the retrieved sources table.")] = True,
) -> None:
    """Answer a question with citations to the retrieved code."""
    with boundary("ask"):
        settings = Settings.from_env()
        limiter = provider_limiter(settings)
        store = VectorStore(settings.chroma_dir)
        retriever = Retriever(
            store,
            resolve_repo(store, repo, settings.default_repo),
            build_embedder(settings, limiter),
        )
        result = answer_question(
            question,
            retriever,
            build_llm(settings, limiter),
            top_k=top_k or settings.top_k,
            min_score=settings.min_score,
            flag_invalid_citations=settings.citation_mode == "flag",
        )
    print_answer(result, show_sources=sources)


@app.command("eval")
def eval_command(
    questions: Annotated[Path, typer.Argument(help="YAML file of questions + expected chunks.")],
    details: Annotated[bool, typer.Option(help="Show the rank for every question.")] = False,
) -> None:
    """Measure retrieval quality (hit@k, MRR) for BM25, vector and hybrid search."""
    with boundary("eval"):
        settings = Settings.from_env()
        spec, repo_root = load_spec(questions)
        limiter = provider_limiter(settings)
        embedder = build_embedder(settings, limiter)
        store = VectorStore(settings.chroma_dir)
        repo = collection_name(f"eval-{repo_root.name}")
        index_repo(repo_root, repo, store, embedder, settings.max_file_bytes)
        scores = evaluate(spec, Retriever(store, repo, embedder))
    title = f"Retrieval eval: {len(spec.questions)} questions, k={spec.k}"
    console.print(eval_table(scores, spec.k, title, caption=f"embedder: {embedder.name}"))
    if details:
        console.print(details_table(scores, [q.question for q in spec.questions]))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Run the HTTP API and the single-page UI."""
    import uvicorn  # imported lazily: only this command needs it

    from repoqa.server import create_app

    with boundary("serve"):
        api = create_app(Settings.from_env())
    uvicorn.run(api, host=host, port=port)


def print_index_stats(stats: IndexStats) -> None:
    reasons = ", ".join(f"{count} {reason}" for reason, count in sorted(stats.skipped.items()))
    console.print(
        f"[green]Indexed[/green] [bold]{stats.repo}[/bold] from {escape(str(stats.root))}: "
        f"{stats.files} files -> {stats.chunks} chunks "
        f"(skipped: {reasons or 'none'})"
    )


def print_answer(result: AnswerResult, *, show_sources: bool) -> None:
    style = "green" if result.answered else "yellow"
    title = escape(result.question)  # questions may contain [brackets]
    console.print(Panel(Markdown(result.answer), title=title, border_style=style))
    if result.rejected_citations:
        console.print(
            f"[red]{len(result.rejected_citations)} citation(s) not backed by retrieved "
            f"code:[/red] {escape(', '.join(result.rejected_citations))}"
        )
    if show_sources and result.sources:
        table = Table(
            title=f"Retrieved sources (best similarity {result.best_similarity:.2f}; * = cited)",
            box=box.SIMPLE_HEAD,
        )
        table.add_column("#", min_width=2, no_wrap=True)
        widest = max(len(src.citation) for src in result.sources)
        table.add_column("citation", min_width=widest, no_wrap=True)
        table.add_column("symbol", overflow="ellipsis", no_wrap=True, max_width=18)
        table.add_column("rrf", justify="right", min_width=6, no_wrap=True)
        table.add_column("cosine", justify="right", min_width=6, no_wrap=True)
        for src in result.sources:
            similarity = "-" if src.similarity is None else f"{src.similarity:.2f}"
            table.add_row(
                f"{src.id}{'*' if src.cited else ''}",
                escape(src.citation),
                escape(src.symbol or ""),
                f"{src.score:.4f}",
                similarity,
            )
        console.print(table)


def eval_table(scores: list[ModeScore], k: int, title: str, caption: str) -> Table:
    table = Table(title=title, caption=caption)
    for column in ("retriever", "hit@1", "hit@3", f"hit@{k}", f"MRR@{k}"):
        table.add_column(column, justify="left" if column == "retriever" else "right")
    best_mrr = max(s.mrr for s in scores)
    for s in scores:
        style = "bold green" if s.mrr == best_mrr else None
        table.add_row(
            s.mode,
            f"{s.hit_at_1:.2f}",
            f"{s.hit_at_3:.2f}",
            f"{s.hit_at_k:.2f}",
            f"{s.mrr:.3f}",
            style=style,
        )
    return table


def details_table(scores: list[ModeScore], questions: list[str]) -> Table:
    table = Table(title="Rank of first relevant chunk (- = not in top k)")
    table.add_column("question")
    for s in scores:
        table.add_column(s.mode, justify="right")
    for i, question in enumerate(questions):
        ranks = ("-" if s.ranks[i] is None else str(s.ranks[i]) for s in scores)
        table.add_row(escape(question), *ranks)
    return table

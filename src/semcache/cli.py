"""Typer CLI: ``demo``, ``stats``, and ``version``.

The demo seeds a few prompts (counting as misses), then queries paraphrases and
unrelated prompts to show HITs vs MISS, and finally prints the stats table. It
runs fully offline and deterministically on the :class:`HashingEmbedder`.
"""

from __future__ import annotations

import itertools

import typer
from rich.console import Console

from semcache import __version__
from semcache.cache import SemanticCache
from semcache.dashboard import render

app = typer.Typer(
    add_completion=False,
    help="semcache — zero-infra semantic cache for LLM calls.",
    no_args_is_help=True,
)

# (prompt, fake LLM answer, tokens, model) seeded into the cache.
_SEED: list[tuple[str, str, int, str]] = [
    (
        "How do I reset my password?",
        "Open Settings > Security and click 'Reset password'.",
        180,
        "gpt-4o",
    ),
    (
        "What is your refund policy?",
        "Refunds are available within 30 days of purchase.",
        150,
        "gpt-4o",
    ),
    (
        "How do I cancel my subscription?",
        "Go to Billing and choose 'Cancel subscription'.",
        165,
        "gpt-4o",
    ),
]

# Paraphrases (expected HIT) and unrelated prompts (expected MISS). The
# paraphrases deliberately share vocabulary so the offline HashingEmbedder
# scores them above threshold without any real embedding model.
_QUERIES: list[str] = [
    "how can I reset my password?",  # paraphrase of seed 0 -> HIT
    "what is your refund policy please?",  # paraphrase of seed 1 -> HIT
    "how do I cancel my subscription?",  # exact match of seed 2 -> HIT
    "How do I reset my password?",  # exact match of seed 0 -> HIT
    "Tell me a joke about penguins.",  # unrelated -> MISS
]


def _build_demo_cache(*, threshold: float = 0.6) -> tuple[SemanticCache, list[tuple[str, bool]]]:
    """Seed a cache, run the query workload, and return it plus per-query outcomes."""
    clock = itertools.count(start=1.0, step=1.0)
    cache = SemanticCache(threshold=threshold, time_fn=lambda: next(clock))
    for prompt, answer, tokens, model in _SEED:
        cache.set(prompt, answer, tokens=tokens, model=model)
    outcomes: list[tuple[str, bool]] = []
    for query in _QUERIES:
        hit = cache.get(query) is not None
        outcomes.append((query, hit))
    return cache, outcomes


def _print_workload(console: Console, outcomes: list[tuple[str, bool]]) -> None:
    console.print("[bold]Querying paraphrases against the seeded cache:[/bold]\n")
    for query, hit in outcomes:
        if hit:
            console.print(f"  [green]HIT [/green]  {query}")
        else:
            console.print(f"  [yellow]MISS[/yellow]  {query}")
    console.print()


@app.command()
def demo(
    threshold: float = typer.Option(0.6, help="Cosine similarity threshold for a hit."),
) -> None:
    """Seed prompts, query paraphrases (HIT/MISS), and print the stats table."""
    console = Console()
    cache, outcomes = _build_demo_cache(threshold=threshold)
    console.print(
        f"[bold cyan]semcache demo[/bold cyan]  (threshold={threshold}, "
        f"seeded {len(_SEED)} prompts, offline HashingEmbedder)\n"
    )
    _print_workload(console, outcomes)
    render(cache, console=console)


@app.command()
def stats(
    threshold: float = typer.Option(0.6, help="Cosine similarity threshold for a hit."),
) -> None:
    """Run the demo workload and show savings (tokens / $)."""
    console = Console()
    cache, _ = _build_demo_cache(threshold=threshold)
    s = cache.stats
    console.print(
        f"[bold]Saved [green]{s.tokens_saved:,}[/green] tokens "
        f"(~[green]${s.cost_saved:,.4f}[/green]) across "
        f"{s.lookups} lookups at a {s.hit_rate * 100:.1f}% hit rate.[/bold]\n"
    )
    render(cache, console=console)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()

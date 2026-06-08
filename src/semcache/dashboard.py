"""Rich rendering of cache statistics and recent entries."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from semcache.cache import SemanticCache
from semcache.models import CacheEntry, Stats


def stats_table(stats: Stats) -> Table:
    """Build a two-column summary table of ``stats``."""
    table = Table(title="semcache :: statistics", title_style="bold cyan", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Lookups", str(stats.lookups))
    table.add_row("Hits", f"[green]{stats.hits}[/green]")
    table.add_row("Misses", f"[yellow]{stats.misses}[/yellow]")
    table.add_row("Hit rate", f"[bold green]{stats.hit_rate * 100:.1f}%[/bold green]")
    table.add_row("Entries", str(stats.entries))
    table.add_row("Evictions", str(stats.evictions))
    table.add_row("Tokens saved", f"{stats.tokens_saved:,}")
    table.add_row("Cost saved", f"[bold green]${stats.cost_saved:,.4f}[/bold green]")
    return table


def entries_table(entries: list[CacheEntry], *, limit: int = 10) -> Table:
    """Build a table of the most recently used ``entries`` (up to ``limit``)."""
    table = Table(title="recent entries", title_style="bold cyan", expand=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Prompt", overflow="fold", max_width=44)
    table.add_column("Hits", justify="right")
    table.add_column("Tokens", justify="right")
    recent = list(reversed(entries))[:limit]
    for i, entry in enumerate(recent, start=1):
        table.add_row(str(i), entry.key, str(entry.hits), str(entry.tokens))
    return table


def render(cache: SemanticCache, *, console: Console | None = None) -> None:
    """Print the stats and recent-entries tables for ``cache``."""
    out = console or Console()
    out.print(Panel.fit("semcache — semantic cache report", style="bold magenta"))
    out.print(stats_table(cache.stats))
    out.print(entries_table(cache.entries))

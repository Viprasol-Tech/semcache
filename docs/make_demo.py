"""Render the ``semcache demo`` workload to an SVG for the README hero.

Run with::

    PYTHONPATH=src python docs/make_demo.py

This reuses the *real* library: the same seed prompts, paraphrase queries, and
the injected clock used by ``semcache demo`` (so the output is deterministic),
then renders the HIT/MISS list and the stats / recent-entries tables into a
``rich`` recording console and exports ``docs/assets/demo.svg``.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from semcache.cli import _SEED, _build_demo_cache, _print_workload
from semcache.dashboard import render

OUT = Path(__file__).resolve().parent / "assets" / "demo.svg"


def main() -> None:
    """Reproduce the demo workload and save it as ``docs/assets/demo.svg``."""
    console = Console(record=True, width=100)
    threshold = 0.6
    cache, outcomes = _build_demo_cache(threshold=threshold)
    console.print(
        f"[bold cyan]semcache demo[/bold cyan]  (threshold={threshold}, "
        f"seeded {len(_SEED)} prompts, offline HashingEmbedder)\n"
    )
    _print_workload(console, outcomes)
    render(cache, console=console)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(OUT), title="semcache")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

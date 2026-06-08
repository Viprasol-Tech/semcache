"""CLI smoke tests via the Typer CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner

from semcache import __version__
from semcache.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_demo_runs_without_error():
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0


def test_demo_shows_hit_and_miss():
    result = runner.invoke(app, ["demo"])
    assert "HIT" in result.stdout
    assert "MISS" in result.stdout


def test_demo_prints_stats_table():
    result = runner.invoke(app, ["demo"])
    assert "Hit rate" in result.stdout
    assert "Tokens saved" in result.stdout


def test_stats_runs_without_error():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "tokens" in result.stdout.lower()


def test_demo_threshold_option():
    result = runner.invoke(app, ["demo", "--threshold", "0.5"])
    assert result.exit_code == 0


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # no_args_is_help exits with code 0 and prints usage.
    assert "Usage" in result.stdout or "Commands" in result.stdout


def test_stats_reports_savings():
    result = runner.invoke(app, ["stats"])
    assert "$" in result.stdout

"""Tests for the centralized console module."""

from __future__ import annotations


def test_console_starts_non_quiet():
    """Default console should not be quiet."""
    from toyforge._console import console

    # Fresh-module behavior: console isn't quiet by default
    assert not console.quiet


def test_set_quiet_swaps_console():
    """set_quiet(True) updates the proxy's inner Console without rebinding."""
    from toyforge._console import console, set_quiet

    original_id = id(console)
    set_quiet(True)
    try:
        assert console.quiet
        assert id(console) == original_id  # same proxy, swapped inner
    finally:
        set_quiet(False)
        assert not console.quiet


def test_set_quiet_propagates_across_modules():
    """Verify that swapping inner Console affects modules that imported via
    `from toyforge._console import console as _console`.

    Uses scenario_gen.expand (no torch dependency) as the consumer module.
    """
    import toyforge._console as _cm
    import toyforge.scenario_gen.expand as _expand

    set_quiet = _cm.set_quiet
    set_quiet(True)
    try:
        # Both the canonical proxy and the consumer's alias must report quiet.
        assert _cm.console.quiet
        assert _expand._console.quiet
    finally:
        set_quiet(False)
    assert not _cm.console.quiet
    assert not _expand._console.quiet


def test_verbose_flag_toggles():
    """set_verbose flips the module-level flag without affecting console.quiet."""
    from toyforge._console import is_verbose, set_verbose

    assert not is_verbose()
    set_verbose(True)
    try:
        assert is_verbose()
    finally:
        set_verbose(False)
    assert not is_verbose()


def test_cli_verbose_and_quiet_mutually_exclusive(tmp_path):
    """Passing both --verbose and --quiet should error."""
    from typer.testing import CliRunner

    from toyforge.cli import app

    runner = CliRunner()
    # Use the verify-seeds subcommand because it requires no setup
    result = runner.invoke(app, ["--verbose", "--quiet", "verify-seeds"])
    assert result.exit_code != 0
    combined = (result.output or "") + str(result.exception or "")
    assert "mutually exclusive" in combined.lower() or "verbose" in combined.lower()


# ---------------------------------------------------------------------------
# warn_overwrite helper tests
# ---------------------------------------------------------------------------


def test_warn_overwrite_prints_yellow_message_when_files_exist(tmp_path, capsys):
    from toyforge._console import set_quiet, warn_overwrite

    set_quiet(False)  # ensure not suppressed
    p1 = tmp_path / "a.txt"
    p1.write_text("x")
    p2 = tmp_path / "b.txt"
    p2.write_text("y")
    p3 = tmp_path / "c.txt"  # does not exist
    warn_overwrite([p1, p2, p3])
    out = capsys.readouterr().out
    assert "Overwriting" in out
    assert "a.txt" in out
    assert "b.txt" in out
    assert "c.txt" not in out


def test_warn_overwrite_silent_when_no_files_exist(tmp_path, capsys):
    from toyforge._console import warn_overwrite

    warn_overwrite([tmp_path / "missing.txt"])
    out = capsys.readouterr().out
    assert "Overwriting" not in out

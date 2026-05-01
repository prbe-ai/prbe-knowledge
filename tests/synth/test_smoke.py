"""Smoke test: the synth CLI is importable and prints help."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "synth" in result.stdout.lower()


def test_extract_stub_exits_nonzero() -> None:
    """The extract subcommand stub must propagate its non-zero exit code
    via __main__.py's SystemExit wrapper."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "extract", "--profile", "/dev/null"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not yet implemented" in result.stderr

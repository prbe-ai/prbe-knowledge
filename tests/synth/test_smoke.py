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

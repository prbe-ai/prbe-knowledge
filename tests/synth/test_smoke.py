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


def test_extract_bad_profile_exits_nonzero() -> None:
    """Passing an empty/invalid profile must exit non-zero and surface an error.

    Updated in Task 21: the stub 'not yet implemented' message is gone now that
    extract is fully wired. We check the real error path instead.
    """
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "extract", "--profile", "/dev/null"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    # /dev/null yields an empty YAML → ProfileError about the mapping type
    assert "ProfileError" in result.stderr or "profile" in result.stderr.lower()

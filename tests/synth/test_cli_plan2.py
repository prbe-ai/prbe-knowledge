"""Subprocess-driven smoke tests for Plan 2 CLI subcommands.

These tests don't run against a real DB — they cover argument parsing,
help-text shape, and error paths. The integrate-mode end-to-end test
lives in test_e2e_run.py (Task 16) and skips without PRBE_TEST_DB_URL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "scripts.synth", *args]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_run_help_lists_plan2_flags() -> None:
    result = _run(["run", "--help"])
    assert result.returncode == 0
    assert "--integrate" in result.stdout
    assert "--time-window" in result.stdout
    assert "--archetypes" in result.stdout
    assert "--limit-scenarios" in result.stdout
    assert "--reset" in result.stdout


def test_init_help_lists_profile_flag() -> None:
    result = _run(["init", "--help"])
    assert result.returncode == 0
    assert "--profile" in result.stdout


def test_clean_help_lists_customer_flag() -> None:
    result = _run(["clean", "--help"])
    assert result.returncode == 0
    assert "--customer" in result.stdout


def test_clean_refuses_non_synth_prefix(tmp_path: Path) -> None:
    result = _run(["clean", "--customer", "prod-tenant-01"])
    assert result.returncode != 0
    assert "refuse to clean non-synthetic" in result.stderr


def test_run_local_writes_world_model_and_manifest(tmp_repo_profile_dir: Path) -> None:
    """Smoke: --integrate NOT set -> local files only -> manifest.json + raw/ exist."""
    out_dir = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    result = _run([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out_dir),
        "--time-window", "14d",
        "--limit-scenarios", "2",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "raw").is_dir()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["mode"] == "local"
    assert manifest["customer_id"].startswith("cust-eval-")


def test_run_archetype_filter_restricts_output(tmp_repo_profile_dir: Path) -> None:
    out_dir = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    result = _run([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out_dir),
        "--time-window", "14d",
        "--archetypes", "STANDUP_UPDATE",
    ])
    assert result.returncode == 0
    # Notion is only emitted by ON_CALL_HANDOFF; --archetypes filter excludes it.
    assert not (out_dir / "raw" / "notion").exists() or not list((out_dir / "raw" / "notion").glob("*.json"))

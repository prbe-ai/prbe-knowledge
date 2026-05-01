"""End-to-end + determinism + integrate-smoke tests for `synth run`.

These tests exercise the full Plan 2 pipeline against the existing
`tmp_repo` fixture from Plan 1's conftest, and pin the deterministic
output contract: same (profile, seed, time_window) -> byte-identical
emitted JSON files.

The integrate-mode smoke is gated on PRBE_TEST_DB_URL and is skipped
in standard CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "scripts.synth", *args]
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = ""  # force the company-context stub fallback
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)


def test_e2e_run_local_writes_full_artifact_set(tmp_repo_profile_dir: Path) -> None:
    """Full pipeline: profile -> WorldModel -> scenarios -> wrappers ->
    local files + eval artifacts. Asserts the run-artifact directory has
    everything Plan 2 promises in section 13 of the spec."""
    out = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"

    result = _run_cli([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out),
        "--time-window", "30d",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    # Manifest + index + frozen profile + world_model snapshot all present.
    assert (out / "manifest.json").exists()
    assert (out / "docs_index.jsonl").exists()
    assert (out / "profile.yaml").exists()
    assert (out / "world_model.json").exists()
    assert (out / "company_context.json").exists()
    assert (out / "warnings.log").exists()

    # raw/ contains both source dirs (STANDUP_UPDATE -> slack, ON_CALL_HANDOFF
    # -> slack + notion). At least one slack and one notion doc.
    slack_docs = list((out / "raw" / "slack").glob("*.json"))
    notion_docs = list((out / "raw" / "notion").glob("*.json"))
    assert len(slack_docs) > 0
    assert len(notion_docs) > 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mode"] == "local"
    assert manifest["seed"] == 7
    assert "STANDUP_UPDATE" in manifest["archetypes_executed"]
    assert "ON_CALL_HANDOFF" in manifest["archetypes_executed"]
    expected_doc_count = len(slack_docs) + len(notion_docs)
    assert manifest["totals"]["documents"] == expected_doc_count

    # docs_index.jsonl row count matches doc count and is sorted by occurred_at.
    rows = [json.loads(line) for line in (out / "docs_index.jsonl").read_text().strip().split("\n")]
    assert len(rows) == expected_doc_count
    occurred_ats = [r["occurred_at"] for r in rows]
    assert occurred_ats == sorted(occurred_ats)


def test_e2e_run_local_is_deterministic(tmp_repo_profile_dir: Path, tmp_path: Path) -> None:
    """Two runs with the same (profile, seed, time_window) must produce
    byte-identical raw/ output. (manifest.json's started_at/finished_at and
    the auto-generated run_id are excluded — those are wall-clock fields.)
    """
    profile = tmp_repo_profile_dir / "profile.yaml"
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    for out in (out_a, out_b):
        result = _run_cli([
            "run",
            "--profile", str(profile),
            "--output-dir", str(out),
            "--time-window", "14d",
        ])
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

    # Compare every raw/<source>/<file>.json byte-for-byte.
    a_files = sorted((out_a / "raw").rglob("*.json"))
    b_files = sorted((out_b / "raw").rglob("*.json"))
    assert [f.relative_to(out_a) for f in a_files] == [f.relative_to(out_b) for f in b_files]
    for fa, fb in zip(a_files, b_files, strict=True):
        assert fa.read_bytes() == fb.read_bytes(), f"diverged: {fa.relative_to(out_a)}"

    # docs_index.jsonl must also be byte-identical.
    assert (out_a / "docs_index.jsonl").read_bytes() == (out_b / "docs_index.jsonl").read_bytes()


def test_e2e_run_archetype_filter_excludes_other(tmp_repo_profile_dir: Path) -> None:
    """--archetypes ON_CALL_HANDOFF -> no slack standup messages emitted.
    (ON_CALL_HANDOFF emits BOTH slack AND notion, but its slack docs go to
    #oncall channel; STANDUP_UPDATE-only docs would be in #standup.)
    The slack/ dir will still have the oncall-thread docs, but no
    STANDUP_UPDATE-archetype docs in docs_index.jsonl.
    """
    out = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    result = _run_cli([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out),
        "--time-window", "30d",
        "--archetypes", "ON_CALL_HANDOFF",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    rows = [json.loads(line) for line in (out / "docs_index.jsonl").read_text().strip().split("\n") if line]
    archetypes = {r["archetype"] for r in rows}
    assert archetypes == {"ON_CALL_HANDOFF"}


@pytest.mark.skipif(
    not os.environ.get("PRBE_TEST_DB_URL"),
    reason="PRBE_TEST_DB_URL env not set; skipping integrate smoke",
)
def test_e2e_run_integrate_smoke(tmp_repo_profile_dir: Path) -> None:
    """Live integrate smoke: requires a disposable Postgres + R2 endpoint.

    Sequence: synth init -> synth run --integrate -> assert ingestion_queue
    row count matches local-file count -> synth clean.
    """
    out = tmp_repo_profile_dir / "out"
    profile = tmp_repo_profile_dir / "profile.yaml"
    customer_id = "cust-eval-fake-01"

    init = _run_cli(["init", "--profile", str(profile)])
    assert init.returncode == 0, f"init stderr:\n{init.stderr}"

    run = _run_cli([
        "run",
        "--profile", str(profile),
        "--output-dir", str(out),
        "--time-window", "14d",
        "--integrate",
    ])
    assert run.returncode == 0, f"run stderr:\n{run.stderr}"

    local_doc_count = len(list((out / "raw").rglob("*.json")))

    # Cross-check ingestion_queue row count via psql.
    psql = subprocess.run(
        [
            "psql", os.environ["PRBE_TEST_DB_URL"], "-tAc",
            f"SELECT COUNT(*) FROM ingestion_queue WHERE customer_id = '{customer_id}'",
        ],
        check=True, capture_output=True, text=True,
    )
    queue_count = int(psql.stdout.strip())
    assert queue_count == local_doc_count, (
        f"ingestion_queue rows ({queue_count}) != local files ({local_doc_count})"
    )

    clean = _run_cli(["clean", "--customer", customer_id])
    assert clean.returncode == 0, f"clean stderr:\n{clean.stderr}"

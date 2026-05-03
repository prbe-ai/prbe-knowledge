"""Recording smoke test: --record-llm writes fixture files to tests/fixtures/synth_llm/.

Skipped unless ANTHROPIC_API_KEY and GOOGLE_API_KEY are set. Run this after prompt changes
to refresh committed fixtures.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.synth.cli import _build_args, _run_async
from scripts.synth.profile import load_profile

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "synth_llm"


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("GOOGLE_API_KEY")),
    reason="Recording smoke requires ANTHROPIC_API_KEY + GOOGLE_API_KEY",
)
async def test_recording_smoke_writes_fixtures(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
) -> None:
    """Run synth run --record-llm; assert fixture files written under tests/fixtures/synth_llm/.

    Slow + expensive — only run when refreshing fixtures.
    """
    output_dir = tmp_path / "recorded-output"
    args = _build_args([
        "run", "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
        "--record-llm", "--output-dir", str(output_dir),
        "--archetypes", "STANDUP_UPDATE,ON_CALL_HANDOFF",
    ])
    profile = load_profile(Path(args.profile))
    rc = await _run_async(profile, output_dir, args)
    assert rc == 0

    fixture_files = list(_FIXTURES_DIR.rglob("*.json")) if _FIXTURES_DIR.exists() else []
    assert len(fixture_files) > 0, (
        f"expected fixture files under {_FIXTURES_DIR} after --record-llm run, "
        "but none were written. check that MockLlmClient record mode is wired correctly."
    )

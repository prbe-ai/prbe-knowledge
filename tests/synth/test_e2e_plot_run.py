"""End-to-end mock-LLM test: full pipeline from profile -> eval artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.synth.cli import _build_args, _run_async
from scripts.synth.llm.base import LlmResponse
from scripts.synth.llm.mock_client import MockLlmClient
from scripts.synth.profile import load_profile

_PLANNER_STUB_OUTPUT = {
    # Empty cast/services/timeline so _validate_against_world always passes
    # (no per-world coupling needed).
    "title": "Stub Scenario",
    "summary": "A stub scenario for E2E testing.",
    "cast": [],
    "affected_services": [],
    "affected_repos": [],
    "root_cause": "stub root cause",
    "decision": None,
    "outcome": None,
    "timeline": [],
    "source_emissions": {},
    "eval_questions": [],
}

_PASS2_STUB_OUTPUT = {"passed": True, "violations": []}


@pytest.fixture
def stub_mock_llm(monkeypatch):
    """Monkeypatch MockLlmClient.generate and generate_structured to return canned stubs.

    Stubs are dispatched by Pydantic schema name:
      PlannerOutputSchema -> _PLANNER_STUB_OUTPUT (empty cast/services so world-validation passes)
      Pass2OutputSchema   -> _PASS2_STUB_OUTPUT (passed=True so no scenario drops)

    `generate` always returns "stub writer output". This lets the templated archetypes
    (STANDUP_UPDATE, ON_CALL_HANDOFF) produce real docs from world entities, while the
    plot archetypes (INCIDENT, LAUNCH, BIG_REFACTOR) emit empty/dropped scenarios that
    don't pollute the artifact set. The test verifies the run completes successfully
    and the artifact directories are created — not specific plot content.
    """
    async def _stub_generate(self, req):
        return LlmResponse(text="stub writer output")

    async def _stub_generate_structured(self, req, schema):
        if schema.__name__ == "PlannerOutputSchema":
            return _PLANNER_STUB_OUTPUT
        if schema.__name__ == "Pass2OutputSchema":
            return _PASS2_STUB_OUTPUT
        raise ValueError(f"Unexpected stub schema: {schema.__name__}")

    monkeypatch.setattr(MockLlmClient, "generate", _stub_generate)
    monkeypatch.setattr(MockLlmClient, "generate_structured", _stub_generate_structured)
    # Ensure the company-context stub fallback fires (no Anthropic key needed).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")


async def test_e2e_run_creates_run_artifacts_with_mock_llm(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
    stub_mock_llm,
) -> None:
    """Full pipeline with --mock-llm: manifest, docs_index, world_model, raw/ all exist.

    Restricts to templated archetypes (STANDUP_UPDATE, ON_CALL_HANDOFF) so the
    run only emits slack/notion docs supported by the Plan 2 IngestionWriter.
    Plot archetypes (INCIDENT, LAUNCH, BIG_REFACTOR) emit linear/github/sentry
    docs that would require the Plan 3 writer extension — not in scope here.
    """
    output_dir = tmp_path / "output"
    args = _build_args([
        "run",
        "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
        "--mock-llm",
        "--output-dir", str(output_dir),
        "--archetypes", "STANDUP_UPDATE,ON_CALL_HANDOFF",
    ])
    profile = load_profile(Path(args.profile))

    rc = await _run_async(profile, output_dir, args)
    assert rc == 0

    # Plan 2 / Plan 3 artifacts that Task 17 + earlier ship:
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "docs_index.jsonl").exists()
    assert (output_dir / "profile.yaml").exists()
    assert (output_dir / "world_model.json").exists()
    assert (output_dir / "company_context.json").exists()
    # Templated archetypes always emit slack docs in tiny_test:
    raw_dir = output_dir / "raw"
    assert raw_dir.is_dir()
    assert any(raw_dir.glob("slack/*.json"))


async def test_e2e_run_creates_scenarios_dir_with_mock_llm(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
    stub_mock_llm,
) -> None:
    """Task 17's write_scenarios produces scenarios/<id>.json files."""
    output_dir = tmp_path / "output"
    args = _build_args([
        "run", "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
        "--mock-llm", "--output-dir", str(output_dir),
        "--archetypes", "STANDUP_UPDATE,ON_CALL_HANDOFF",
    ])
    profile = load_profile(Path(args.profile))
    await _run_async(profile, output_dir, args)

    scenarios_dir = output_dir / "scenarios"
    assert scenarios_dir.is_dir()
    json_files = sorted(scenarios_dir.glob("*.json"))
    assert len(json_files) > 0, "expected at least one templated scenario JSON"

    # Each scenario JSON parses and has the expected top-level keys.
    for jf in json_files:
        data = json.loads(jf.read_text())
        assert "id" in data
        assert "archetype_name" in data


async def test_e2e_questions_jsonl_exists(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
    stub_mock_llm,
) -> None:
    """Task 17's write_questions_jsonl always creates the file (may be empty if no eval Qs)."""
    output_dir = tmp_path / "output"
    args = _build_args([
        "run", "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
        "--mock-llm", "--output-dir", str(output_dir),
        "--archetypes", "STANDUP_UPDATE,ON_CALL_HANDOFF",
    ])
    profile = load_profile(Path(args.profile))
    await _run_async(profile, output_dir, args)
    assert (output_dir / "questions.jsonl").exists()


async def test_e2e_run_is_deterministic_with_mock_llm(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
    stub_mock_llm,
) -> None:
    """Two runs with the same profile + --mock-llm produce byte-identical raw/ output."""
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    for out in (out_a, out_b):
        args = _build_args([
            "run", "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
            "--mock-llm", "--output-dir", str(out),
            "--archetypes", "STANDUP_UPDATE,ON_CALL_HANDOFF",
        ])
        profile = load_profile(Path(args.profile))
        await _run_async(profile, out, args)

    a_files = sorted((out_a / "raw").rglob("*.json"))
    b_files = sorted((out_b / "raw").rglob("*.json"))
    assert [f.relative_to(out_a) for f in a_files] == [f.relative_to(out_b) for f in b_files]
    for fa, fb in zip(a_files, b_files, strict=True):
        assert fa.read_bytes() == fb.read_bytes(), f"diverged: {fa.relative_to(out_a)}"

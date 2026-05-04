"""End-to-end mock-LLM tests: full pipeline from profile -> eval artifacts.

Test 1: test_e2e_plot_pipeline_emits_plot_docs
    Smart-stub that keys off the real WorldModel. Asserts INCIDENT, LAUNCH,
    BIG_REFACTOR plot docs actually land on disk (github/linear/sentry/notion/slack).

Test 2: test_e2e_manifest_and_templated_artifacts
    Regression test: manifest + world_model.json + raw/slack always present
    after a full run (including plot archetypes with the smart stub).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.synth.cli import _build_args, _run_async
from scripts.synth.llm.base import LlmResponse
from scripts.synth.llm.mock_client import MockLlmClient
from scripts.synth.profile import load_profile

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SAFE_WRITER_TEXT = "Plain English summary about the situation in this scenario."
_PASS2_STUB_OUTPUT = {"passed": True, "violations": []}


def _make_smart_planner_stub(world_holder: dict):
    """Return a generate_structured stub that picks real entities from the world.

    world_holder["world"] is populated by the monkeypatched merge_world_model
    wrapper before the first planner call, so by the time generate_structured
    fires the world is already available.
    """

    async def _stub_generate_structured(self, req, schema):
        if schema.__name__ == "Pass2OutputSchema":
            return _PASS2_STUB_OUTPUT

        if schema.__name__ == "PlannerOutputSchema":
            world = world_holder.get("world")
            if world is None or not world.people or not world.services:
                # Fallback: empty cast causes validator drop — should never happen
                # when using tmp_repo_profile_dir (which always yields people+services).
                return {
                    "title": "Stub Scenario",
                    "summary": "A stub scenario.",
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

            # Pull real entities so planner validation passes
            people = list(world.people)
            cast_ids = [people[0].canonical_id]
            if len(people) >= 2:
                cast_ids.append(people[1].canonical_id)

            svc = world.services[0]
            channel = world.channels[0].name if world.channels else None

            timeline = []
            if channel:
                from datetime import UTC, datetime
                timeline = [
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "source": "slack",
                        "kind": "message",
                        "channel": channel,
                    }
                ]

            return {
                "title": "Stub Scenario",
                "summary": "A stub scenario for real plot pipeline testing.",
                "cast": [{"canonical_id": cid, "role_in_scenario": "participant"} for cid in cast_ids],
                "affected_services": [svc.qualified],
                "affected_repos": [svc.repo_url],
                "root_cause": "stub root cause",
                "decision": None,
                "outcome": None,
                "timeline": timeline,
                "source_emissions": {
                    "slack": 1,
                    "linear": 1,
                    "notion": 1,
                    "github": 1,
                },
                "eval_questions": [
                    {
                        "input": "What happened?",
                        "answer_substring": "Stub Scenario",
                        "difficulty": "easy",
                    }
                ],
            }

        raise ValueError(f"Unexpected stub schema: {schema.__name__}")

    return _stub_generate_structured


# ---------------------------------------------------------------------------
# Test 1 — smart stub drives plot docs to disk
# ---------------------------------------------------------------------------


async def test_e2e_plot_pipeline_emits_plot_docs(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Real plot pipeline E2E: smart stub + tmp_repo profile produces plot docs on disk.

    Asserts that LAUNCH and BIG_REFACTOR plot scenarios survive validation and emit
    docs to raw/github/, raw/linear/, raw/notion/, raw/slack/. INCIDENT also emits
    docs (including a templated sentry alert) after the auto-generated kebab fix.
    """
    # Capture the WorldModel once it's built so the planner stub can use real entities.
    world_holder: dict = {}

    # Monkeypatch both import sites for merge_world_model.
    import scripts.synth.cli as cli_mod
    import scripts.synth.world_model as wm_mod

    _real_merge = wm_mod.merge_world_model

    def _capturing_merge(*args, **kwargs):
        result = _real_merge(*args, **kwargs)
        world_holder["world"] = result
        return result

    monkeypatch.setattr(cli_mod, "merge_world_model", _capturing_merge)
    monkeypatch.setattr(wm_mod, "merge_world_model", _capturing_merge)

    # Stub the LLM clients: writer returns safe prose; planner returns real entities.
    async def _stub_generate(self, req):
        return LlmResponse(text=_SAFE_WRITER_TEXT)

    smart_structured = _make_smart_planner_stub(world_holder)

    monkeypatch.setattr(MockLlmClient, "generate", _stub_generate)
    monkeypatch.setattr(MockLlmClient, "generate_structured", smart_structured)

    # No real LLM calls for company inference.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    output_dir = tmp_path / "output"
    args = _build_args([
        "run",
        "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
        "--mock-llm",
        "--output-dir", str(output_dir),
    ])
    profile = load_profile(Path(args.profile))

    rc = await _run_async(profile, output_dir, args)
    assert rc == 0

    raw = output_dir / "raw"

    # Plot archetypes emit to github, linear, notion
    assert any(raw.glob("github/*.json")), "expected GitHub docs from LAUNCH/BIG_REFACTOR"
    assert any(raw.glob("linear/*.json")), "expected Linear docs from plot archetypes"
    assert any(raw.glob("notion/*.json")), "expected Notion docs from plot archetypes"

    # INCIDENT emits a templated sentry alert (plus LLM docs for the other sources)
    assert any(raw.glob("sentry/*.json")), "expected Sentry docs from INCIDENT"

    # Templated archetypes produce slack docs
    assert any(raw.glob("slack/*.json"))

    # scenarios/ should contain at least one scenario JSON
    scenarios = list((output_dir / "scenarios").glob("*.json"))
    assert len(scenarios) > 0

    # Plot archetypes now yield (spec, doc) pairs — their specs reach write_scenarios.
    plot_scenarios = [
        s for s in scenarios
        if any(p in s.read_text() for p in ("INCIDENT", "LAUNCH", "BIG_REFACTOR"))
    ]
    assert len(plot_scenarios) > 0, "expected at least one plot scenario JSON in scenarios/"

    # questions.jsonl should be non-empty: plot specs carry eval_questions now.
    questions_bytes = (output_dir / "questions.jsonl").read_bytes()
    assert questions_bytes, "expected plot eval questions in questions.jsonl"
    question_lines = questions_bytes.strip().splitlines()
    assert len(question_lines) > 0, "expected at least one plot eval question row"

    # At least one github doc exists
    github_docs = list(raw.glob("github/*.json"))
    assert len(github_docs) >= 1

    # Each github doc is valid JSON (content format varies by archetype — webhook payload or SynthDoc)
    for gd in github_docs:
        data = json.loads(gd.read_text())
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Test 2 — manifest + templated artifacts always present (regression)
# ---------------------------------------------------------------------------


async def test_e2e_manifest_and_templated_artifacts(
    tmp_repo_profile_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Full run produces manifest.json, docs_index.jsonl, world_model.json, and slack docs.

    Uses the same smart stub so plot archetypes don't drop and cause false negatives.
    """
    world_holder: dict = {}

    import scripts.synth.cli as cli_mod
    import scripts.synth.world_model as wm_mod

    _real_merge = wm_mod.merge_world_model

    def _capturing_merge(*args, **kwargs):
        result = _real_merge(*args, **kwargs)
        world_holder["world"] = result
        return result

    monkeypatch.setattr(cli_mod, "merge_world_model", _capturing_merge)
    monkeypatch.setattr(wm_mod, "merge_world_model", _capturing_merge)

    async def _stub_generate(self, req):
        return LlmResponse(text=_SAFE_WRITER_TEXT)

    smart_structured = _make_smart_planner_stub(world_holder)

    monkeypatch.setattr(MockLlmClient, "generate", _stub_generate)
    monkeypatch.setattr(MockLlmClient, "generate_structured", smart_structured)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    output_dir = tmp_path / "output"
    args = _build_args([
        "run",
        "--profile", str(tmp_repo_profile_dir / "profile.yaml"),
        "--mock-llm",
        "--output-dir", str(output_dir),
    ])
    profile = load_profile(Path(args.profile))

    rc = await _run_async(profile, output_dir, args)
    assert rc == 0

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "docs_index.jsonl").exists()
    assert (output_dir / "world_model.json").exists()
    assert (output_dir / "questions.jsonl").exists()
    assert (output_dir / "raw").is_dir()
    assert any((output_dir / "raw").glob("slack/*.json")), "templated archetypes should emit slack docs"
    assert (output_dir / "profile.yaml").exists()
    assert (output_dir / "company_context.json").exists()

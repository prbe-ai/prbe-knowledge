"""Tests for write_scenarios — scenarios/<id>.json artifact writer."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson

from scripts.synth.output.eval_artifacts import write_scenarios
from scripts.synth.scenarios import EvalQuestion, ScenarioSpec

_TS = datetime(2026, 5, 1, tzinfo=UTC)


def _make_scenario(sid: str, **extra) -> ScenarioSpec:
    defaults = dict(
        id=sid,
        archetype_name="INCIDENT",
        instance_ts=_TS,
        title=f"title-{sid}",
        summary="test summary",
        cast=(),
        affected_services=("payments-svc",),
        root_cause="feature flag default flipped",
        decision=None,
        outcome=None,
        eval_questions=(),
    )
    defaults.update(extra)
    return ScenarioSpec(**defaults)


def test_empty_scenarios_creates_directory_but_no_files(tmp_path: Path) -> None:
    """Empty scenario list → scenarios/ directory is created, but no JSON files written."""
    write_scenarios(tmp_path, [])
    scenarios_dir = tmp_path / "scenarios"
    assert scenarios_dir.is_dir()
    assert list(scenarios_dir.iterdir()) == []


def test_one_scenario_one_file(tmp_path: Path) -> None:
    """Single scenario → exactly one file at scenarios/<id>.json."""
    scenario = _make_scenario("scn-payments-2026-04-12")
    write_scenarios(tmp_path, [scenario])
    out_file = tmp_path / "scenarios" / "scn-payments-2026-04-12.json"
    assert out_file.exists()
    data = orjson.loads(out_file.read_bytes())
    assert data["id"] == "scn-payments-2026-04-12"


def test_multiple_scenarios_multiple_files(tmp_path: Path) -> None:
    """Multiple scenarios → one file per scenario; all ids present."""
    scenarios = [_make_scenario(f"scn-{i}") for i in range(3)]
    write_scenarios(tmp_path, scenarios)
    files = list((tmp_path / "scenarios").iterdir())
    assert len(files) == 3
    ids_on_disk = {orjson.loads(f.read_bytes())["id"] for f in files}
    assert ids_on_disk == {"scn-0", "scn-1", "scn-2"}


def test_full_spec_fields_serialized(tmp_path: Path) -> None:
    """ScenarioSpec extra fields (title, summary, root_cause, eval_questions) all serialized."""
    q = EvalQuestion(
        question="What caused the outage?",
        answer_substring="flag flipped",
        tags=("INCIDENT",),
        difficulty="easy",
        question_index=0,
    )
    scenario = _make_scenario(
        "scn-full",
        title="Full Spec Test",
        summary="Payments-svc 500s after feature flag rollout",
        root_cause="feature flag default flipped without backend ready",
        eval_questions=(q,),
    )
    write_scenarios(tmp_path, [scenario])
    data = orjson.loads((tmp_path / "scenarios" / "scn-full.json").read_bytes())
    assert data["title"] == "Full Spec Test"
    assert data["root_cause"] == "feature flag default flipped without backend ready"
    assert len(data["eval_questions"]) == 1
    assert data["eval_questions"][0]["question"] == "What caused the outage?"
    assert data["affected_services"] == ["payments-svc"]

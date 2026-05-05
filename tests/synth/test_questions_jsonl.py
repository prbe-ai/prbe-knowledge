"""Tests for write_questions_jsonl — questions.jsonl eval artifact writer."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.eval_artifacts import write_questions_jsonl
from scripts.synth.scenarios import EvalQuestion, ScenarioSpec

_TS = datetime(2026, 5, 1, tzinfo=UTC)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_scenario(sid: str, questions: list[EvalQuestion]) -> ScenarioSpec:
    return ScenarioSpec(
        id=sid,
        archetype_name="INCIDENT",
        instance_ts=_TS,
        title=f"title-{sid}",
        summary="summary",
        cast=(),
        affected_services=(),
        root_cause=None,
        decision=None,
        outcome=None,
        eval_questions=tuple(questions),
    )


def _make_doc(doc_id: str, scenario_id: str, source: Source) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=source,
        source_event_id=doc_id,
        text="some text",
        occurred_at=None,
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id=scenario_id,
        archetype="INCIDENT",
        personas=(),
        services_mentioned=(),
        priority=0,
    )


def _make_question(
    qtext: str,
    answer: str,
    tags: list[str],
    difficulty: str,
    idx: int = 0,
) -> EvalQuestion:
    return EvalQuestion(
        question=qtext,
        answer_substring=answer,
        tags=tuple(tags),
        difficulty=difficulty,
        question_index=idx,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_empty_scenarios_produces_empty_file(tmp_path: Path) -> None:
    """Empty scenario list → file created but empty (zero bytes or zero lines)."""
    write_questions_jsonl(tmp_path, [], [])
    out = tmp_path / "questions.jsonl"
    assert out.exists()
    assert out.read_bytes().strip() == b""


def test_one_scenario_two_questions_two_rows(tmp_path: Path) -> None:
    """Single scenario with 2 EvalQuestions → 2 rows in questions.jsonl."""
    q1 = _make_question("What caused the outage?", "flag flipped", ["INCIDENT"], "easy", idx=0)
    q2 = _make_question("Who fixed it?", "alice", ["INCIDENT", "cross-source"], "medium", idx=1)
    scenario = _make_scenario("scn-abc", [q1, q2])
    docs = [
        _make_doc("scn-abc-slack-0", "scn-abc", Source.SLACK),
        _make_doc("scn-abc-notion-0", "scn-abc", Source.NOTION),
    ]
    write_questions_jsonl(tmp_path, [scenario], docs)
    lines = (tmp_path / "questions.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    row0 = orjson.loads(lines[0])
    assert row0["input"] == "What caused the outage?"
    assert row0["expected"]["answer_substring"] == "flag flipped"
    assert row0["difficulty"] == "easy"
    assert row0["scenario_id"] == "scn-abc"
    row1 = orjson.loads(lines[1])
    assert row1["difficulty"] == "medium"


def test_multiple_scenarios_sorted_by_scenario_id_then_difficulty(tmp_path: Path) -> None:
    """Multiple scenarios → rows sorted by (scenario_id, difficulty) for determinism."""
    # scenario "scn-zzz" should sort after "scn-aaa"
    q_z = _make_question("Q-Z easy?", "ans", ["LAUNCH"], "easy", idx=0)
    q_a1 = _make_question("Q-A easy?", "ans", ["INCIDENT"], "easy", idx=0)
    q_a2 = _make_question("Q-A medium?", "ans", ["INCIDENT"], "medium", idx=1)
    scenario_z = _make_scenario("scn-zzz", [q_z])
    scenario_a = _make_scenario("scn-aaa", [q_a1, q_a2])
    write_questions_jsonl(tmp_path, [scenario_z, scenario_a], [])
    lines = (tmp_path / "questions.jsonl").read_bytes().splitlines()
    scenario_ids = [orjson.loads(line)["scenario_id"] for line in lines]
    assert scenario_ids == ["scn-aaa", "scn-aaa", "scn-zzz"]


def test_evidence_doc_keys_derived_from_emitted_docs(tmp_path: Path) -> None:
    """evidence_doc_keys lists paths derived from emitted_docs whose scenario_id matches."""
    q = _make_question("Who reported?", "alice", ["INCIDENT"], "easy", idx=0)
    scenario = _make_scenario("scn-xyz", [q])
    docs = [
        _make_doc("scn-xyz-slack-0", "scn-xyz", Source.SLACK),
        _make_doc("scn-xyz-notion-0", "scn-xyz", Source.NOTION),
        _make_doc("scn-other-slack-0", "scn-other", Source.SLACK),  # different scenario — excluded
    ]
    write_questions_jsonl(tmp_path, [scenario], docs)
    row = orjson.loads((tmp_path / "questions.jsonl").read_bytes().splitlines()[0])
    keys = row["expected"]["evidence_doc_keys"]
    # Only docs matching scenario_id "scn-xyz" should appear
    assert all("scn-xyz" in k for k in keys)
    assert not any("scn-other" in k for k in keys)
    assert len(keys) == 2


def test_round_trip_orjson_loads(tmp_path: Path) -> None:
    """Write then orjson.loads each line → equal to expected dicts (round-trip)."""
    q = _make_question("When did feature ship?", "2026-04-15", ["LAUNCH"], "easy", idx=0)
    scenario = _make_scenario("scn-rt", [q])
    docs = [_make_doc("scn-rt-slack-0", "scn-rt", Source.SLACK)]
    write_questions_jsonl(tmp_path, [scenario], docs)
    lines = (tmp_path / "questions.jsonl").read_bytes().splitlines()
    assert len(lines) == 1
    row = orjson.loads(lines[0])
    assert set(row.keys()) == {"input", "expected", "tags", "scenario_id", "difficulty"}
    assert isinstance(row["expected"]["evidence_doc_keys"], list)
    assert isinstance(row["tags"], list)

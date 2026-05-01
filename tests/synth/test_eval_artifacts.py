"""Tests for eval artifact writers: manifest.json, docs_index.jsonl, profile.yaml, warnings.log."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import yaml

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.eval_artifacts import (
    write_docs_index,
    write_manifest,
    write_profile,
    write_warnings,
)
from scripts.synth.profile import Profile
from scripts.synth.validator import Violation
from scripts.synth.world_model import WorldModel


def _profile() -> Profile:
    raw = {
        "customer_id": "cust-eval-test-01",
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny-test",
        "seed": 42,
    }
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _world() -> WorldModel:
    return WorldModel(
        repos=(),
        people=(),
        services=(),
        topic_pool=(),
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="acme",
        seed=42,
        extracted_at=datetime(2026, 5, 1, tzinfo=UTC),
        sha_set={"github.com/x/y": "abc123"},
    )


def _doc(source: Source, source_event_id: str, occurred_at: datetime) -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=source,
        source_event_id=source_event_id,
        text="x",
        occurred_at=occurred_at,
        channel="#standup" if source == Source.SLACK else None,
        page_id=source_event_id if source == Source.NOTION else None,
        thread_parent_id=None,
        scenario_id="scn-1",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )


def test_write_manifest_produces_expected_keys(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        run_id="run-x",
        profile=_profile(),
        world=_world(),
        totals={
            "archetypes_executed": {"STANDUP_UPDATE": {"requested": 5, "generated": 5, "dropped": 0}},
            "totals": {"scenarios": 5, "documents": 5, "questions": 0},
            "warnings_count": 0,
        },
        mode="local",
        started_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 1, 10, 5, tzinfo=UTC),
    )
    payload = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert payload["run_id"] == "run-x"
    assert payload["seed"] == 42
    assert payload["customer_id"] == "cust-eval-test-01"
    assert payload["mode"] == "local"
    assert payload["world_model"]["people_count"] == 0
    assert payload["archetypes_executed"]["STANDUP_UPDATE"]["generated"] == 5
    assert payload["totals"]["scenarios"] == 5


def test_write_docs_index_orders_by_occurred_at_then_id(tmp_path: Path) -> None:
    docs = [
        _doc(Source.SLACK, "doc-2", datetime(2026, 5, 1, 9, 0, tzinfo=UTC)),
        _doc(Source.SLACK, "doc-1", datetime(2026, 5, 1, 8, 0, tzinfo=UTC)),
        _doc(Source.NOTION, "doc-3", datetime(2026, 5, 1, 9, 0, tzinfo=UTC)),
    ]
    write_docs_index(tmp_path, docs)
    raw = (tmp_path / "docs_index.jsonl").read_text().strip().split("\n")
    rows = [orjson.loads(line) for line in raw]
    assert [r["doc_id"] for r in rows] == ["doc-1", "doc-2", "doc-3"]
    assert rows[0]["raw_key"] == "raw/slack/doc-1.json"
    assert rows[2]["raw_key"] == "raw/notion/doc-3.json"


def test_write_profile_dumps_raw_yaml(tmp_path: Path) -> None:
    write_profile(tmp_path, _profile())
    parsed = yaml.safe_load((tmp_path / "profile.yaml").read_text())
    assert parsed["customer_id"] == "cust-eval-test-01"
    assert parsed["seed"] == 42


def test_write_warnings_formats_violations_and_notes(tmp_path: Path) -> None:
    violations = (
        Violation(doc_id="doc-1", out_of_world=("foo",)),
        Violation(doc_id="doc-2", out_of_world=("bar", "baz")),
    )
    notes = ["dropped 1 scenario without recent topics"]
    write_warnings(tmp_path, violations, notes)
    text = (tmp_path / "warnings.log").read_text()
    assert "VIOLATION: doc=doc-1" in text
    assert "VIOLATION: doc=doc-2" in text
    assert "NOTE: dropped 1 scenario" in text


def test_write_docs_index_empty_input_writes_empty_file(tmp_path: Path) -> None:
    write_docs_index(tmp_path, [])
    path = tmp_path / "docs_index.jsonl"
    assert path.exists()
    assert path.read_bytes() == b""


def test_write_warnings_empty_input_writes_empty_file(tmp_path: Path) -> None:
    write_warnings(tmp_path, (), [])
    path = tmp_path / "warnings.log"
    assert path.exists()
    assert path.read_text() == ""

"""Eval artifact writers — manifest.json, docs_index.jsonl, profile.yaml, warnings.log,
questions.jsonl, and scenarios/*.json.

Each writer is deterministic given its inputs (modulo wall-clock fields like
`started_at` / `finished_at` in the manifest, which the caller passes in).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
import yaml

if TYPE_CHECKING:
    from scripts.synth.archetypes.base import ScenarioSpec
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.profile import Profile
    from scripts.synth.validator import Violation
    from scripts.synth.world_model import WorldModel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default(obj: object) -> object:
    """orjson-style default encoder: handles dataclasses, datetimes, Paths, tuples."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: getattr(obj, k)
            for k in (f.name for f in dataclasses.fields(obj))  # type: ignore[arg-type]
        }
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _dumps(obj: object) -> bytes:
    """Recursively encode *obj* to pretty JSON bytes.

    Handles dataclasses (via field walk), datetimes, Paths, tuples, and dicts.
    Returns ``bytes`` (UTF-8) compatible with ``Path.write_bytes``.
    """

    def _encode(v: object) -> object:
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {
                k: _encode(getattr(v, k))
                for k in (f.name for f in dataclasses.fields(v))  # type: ignore[arg-type]
            }
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple | list):
            return [_encode(x) for x in v]
        if isinstance(v, dict):
            return {k: _encode(val) for k, val in v.items()}
        return v

    return json.dumps(_encode(obj), indent=2, sort_keys=False).encode()


def write_manifest(
    out_dir: Path,
    *,
    run_id: str,
    profile: Profile,
    world: WorldModel,
    totals: dict,
    mode: str,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Write the run-level summary to `<out_dir>/manifest.json`."""
    manifest = {
        "run_id": run_id,
        "profile_name": profile.preset,
        "seed": profile.seed,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "customer_id": profile.customer_id,
        "mode": mode,
        "repos": [
            {
                "url": repo.url,
                "sha": world.sha_set.get(repo.url, "unknown"),
                "mode": "local",
            }
            for repo in profile.repos
        ],
        "world_model": {
            "people_count": len(world.people),
            "services_count": len(world.services),
            "channels_count": len(world.channels),
        },
        "archetypes_executed": totals.get("archetypes_executed", {}),
        "totals": totals.get("totals", {}),
        "warnings_count": totals.get("warnings_count", 0),
    }
    (out_dir / "manifest.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
    )


def write_docs_index(out_dir: Path, docs: list[SynthDoc]) -> None:
    """Write one row per SynthDoc, sorted (occurred_at, id) for determinism."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_docs = sorted(docs, key=lambda d: (d.occurred_at, d.id))
    lines: list[bytes] = []
    for doc in sorted_docs:
        row = {
            "doc_id": doc.id,
            "scenario_id": doc.scenario_id,
            "archetype": doc.archetype,
            "source": doc.source.value,
            "occurred_at": doc.occurred_at.isoformat(),
            "raw_key": f"raw/{doc.source.value}/{doc.source_event_id}.json",
            "personas": list(doc.personas),
            "services_mentioned": list(doc.services_mentioned),
            "is_evidence_for_question_ids": [],
        }
        lines.append(orjson.dumps(row))
    payload = b"\n".join(lines) + (b"\n" if lines else b"")
    (out_dir / "docs_index.jsonl").write_bytes(payload)


def write_profile(out_dir: Path, profile: Profile) -> None:
    """Freeze the resolved profile as YAML next to the run output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.yaml").write_text(
        yaml.safe_dump(profile.raw, sort_keys=False)
    )


def write_warnings(
    out_dir: Path,
    violations: tuple[Violation, ...],
    notes: list[str],
) -> None:
    """Plain-text log of validator violations + freeform notes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for v in violations:
        lines.append(f"VIOLATION: doc={v.doc_id} out_of_world={list(v.out_of_world)}")
    lines.extend(f"NOTE: {n}" for n in notes)
    payload = "\n".join(lines) + ("\n" if lines else "")
    (out_dir / "warnings.log").write_text(payload)


def write_questions_jsonl(
    out_dir: Path,
    scenarios: list[ScenarioSpec],
    emitted_docs: list[SynthDoc],
) -> None:
    """Write one question per line to <out_dir>/questions.jsonl.

    Rows are sorted by (scenario_id, difficulty, question_index) for determinism.
    Each row matches spec §12.1:
        {"input": ..., "expected": {"answer_substring": ..., "evidence_doc_keys": [...]},
         "tags": [...], "scenario_id": ..., "difficulty": ...}

    evidence_doc_keys is derived from emitted_docs whose scenario_id matches the
    question's scenario.  Each key is ``raw/<source>/<source_event_id>.json``, sorted
    for determinism.  This uses post-materialization SynthDocs rather than the
    pre-materialization DocSpecs so the paths match what is actually on disk.
    """
    out_path = out_dir / "questions.jsonl"

    # Build lookup: scenario_id -> sorted list of "raw/<source>/<source_event_id>.json" keys
    docs_by_scenario: dict[str, list[str]] = {}
    for doc in emitted_docs:
        key = f"raw/{doc.source.value}/{doc.source_event_id}.json"
        docs_by_scenario.setdefault(doc.scenario_id, []).append(key)
    # Sort the keys within each scenario bucket for determinism
    for bucket in docs_by_scenario.values():
        bucket.sort()

    # (scenario_id, difficulty, question_index, row_dict)
    rows: list[tuple[str, str, int, dict]] = []

    for scenario in scenarios:
        doc_keys = docs_by_scenario.get(scenario.id, [])
        for q in scenario.eval_questions:
            row = {
                "input": q.question,
                "expected": {
                    "answer_substring": q.answer_substring,
                    "evidence_doc_keys": doc_keys,
                },
                "tags": list(q.tags),
                "scenario_id": scenario.id,
                "difficulty": q.difficulty,
            }
            rows.append((scenario.id, q.difficulty, q.question_index, row))

    rows.sort(key=lambda t: (t[0], t[1], t[2]))

    lines = b"\n".join(orjson.dumps(r) for _, _, _, r in rows)
    out_path.write_bytes(lines)


def write_scenarios(out_dir: Path, scenarios: list[ScenarioSpec]) -> None:
    """Write <out_dir>/scenarios/<scenario_id>.json per scenario.

    Each file is the full ScenarioSpec serialized via _dumps (handles
    dataclasses, datetimes, Paths, and tuples).  The scenarios/ directory
    is created if it does not exist.
    """
    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        out_file = scenarios_dir / f"{scenario.id}.json"
        out_file.write_bytes(_dumps(scenario))

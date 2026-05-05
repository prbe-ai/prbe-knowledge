"""Eval artifact writers — manifest.json, docs_index.jsonl, profile.yaml, warnings.log.

Each writer is deterministic given its inputs (modulo wall-clock fields like
`started_at` / `finished_at` in the manifest, which the caller passes in).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
import yaml

if TYPE_CHECKING:
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.profile import Profile
    from scripts.synth.validator import Violation
    from scripts.synth.world_model import WorldModel


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

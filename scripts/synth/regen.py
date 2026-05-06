"""Validator regen loop — orchestrates per-doc regeneration when a plot
scenario fails strict validation.

Public surface:
- format_failure_context: build the failure block for a regen prompt
- splice_regenerated: merge regenerated text back into a docs tuple
- regen_loop: per-scenario async orchestrator (max N rounds)
- RegenResult, RoundReport: result shapes for callers + observability

Decisions (locked from 2026-05-05 handoff):
- Per-doc regen, not per-scenario.
- Pass 1 + Pass 2 violations feed a single unified prompt.
- Default round budget 3 (configurable via Profile.regen_max_rounds).
- On exhaustion: drop scenario, terminal log includes survival info.
"""

from __future__ import annotations

from dataclasses import replace

from scripts.synth.llm.validator_pass2 import Pass2Result
from scripts.synth.output.base import SynthDoc
from scripts.synth.validator import Violation


def format_failure_context(
    *,
    pass1_violations: tuple[Violation, ...],
    pass2_result: Pass2Result | None,
    target_doc_id: str,
) -> str:
    """Render Pass 1 + Pass 2 violations for `target_doc_id` as a prompt block.

    Returns "" when neither pass flagged the target. The string is meant to
    be interpolated into writer_regen.txt as the `{failure_context}` field.
    """
    lines: list[str] = []

    pass1_for_doc = [v for v in pass1_violations if v.doc_id == target_doc_id]
    if pass1_for_doc:
        tokens = sorted({t for v in pass1_for_doc for t in v.out_of_world})
        lines.append(
            "Pass 1 (out-of-world tokens): "
            + ", ".join(tokens)
            + " — these names are NOT in the WorldModel allowlist. "
            "Replace them or rephrase to avoid them."
        )

    if pass2_result is not None:
        pass2_for_doc = [v for v in pass2_result.violations if v.doc_id == target_doc_id]
        for v in pass2_for_doc:
            lines.append(f"Pass 2 (consistency issue): {v.issue}")

    return "\n".join(lines)


def splice_regenerated(
    original_docs: tuple[SynthDoc, ...],
    *,
    regenerated_text_by_doc_id: dict[str, str],
) -> tuple[SynthDoc, ...]:
    """Return a new tuple where named docs have new `text`, everything else is identical.

    Preserves doc order, count, and every SynthDoc field (id, source,
    source_event_id, occurred_at, channel, page_id, thread_parent_id,
    scenario_id, archetype, personas, services_mentioned, priority) — only
    `text` changes for entries listed in `regenerated_text_by_doc_id`.

    Raises:
        ValueError: if `regenerated_text_by_doc_id` references a doc id
            that is not present in `original_docs`.
    """
    if not regenerated_text_by_doc_id:
        return original_docs

    known_ids = {d.id for d in original_docs}
    unknown = set(regenerated_text_by_doc_id.keys()) - known_ids
    if unknown:
        raise ValueError(
            f"splice_regenerated: doc id(s) not in original scenario: {sorted(unknown)}"
        )

    out: list[SynthDoc] = []
    for d in original_docs:
        new_text = regenerated_text_by_doc_id.get(d.id)
        if new_text is None:
            out.append(d)
        else:
            out.append(replace(d, text=new_text))
    return tuple(out)

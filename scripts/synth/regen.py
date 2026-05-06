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

from dataclasses import dataclass

from scripts.synth.llm.validator_pass2 import Pass2Result
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

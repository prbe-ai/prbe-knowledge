"""Haiku tiebreaker for ambiguous classification cases (spec §6 step 2).

Third leg of the hybrid match pipeline (rules → embedding → LLM). When
the rules + embedding combination produces genuinely ambiguous top
candidates, ``resolve_ambiguity`` calls Anthropic with the incident
plus a short list of candidate class IDs and lets the LLM pick one
(or ``"none"`` if none fit). The classifier orchestrator (future task)
catches a ``choice=None`` result and decides degraded-mode policy;
this module's only job is the LLM round-trip plus fail-safe handling.

Failure modes that all collapse to ``choice=None``:

- The Anthropic API raises (timeout, 4xx, 5xx) — caught, logged.
- The model returns invalid JSON — caught, logged.
- The model returns a ``choice`` that isn't in the candidates list (it
  hallucinated a class_id) — rejected without a log fire, since this
  is a guarded-against contract violation rather than a transport /
  parse failure.

The first two emit a structured ``kg.classifier.tiebreaker_failed``
warning (event name is part of the operational contract — operators
grep for it in production). The hallucination path is silent because
it's expected behavior the contract handles, not an outage signal.

Tiebreaker frequency is intentionally not budgeted as a fixed rate;
spec §6 step 2 explicitly states it's high during Phase 0-1 bootstrap
and drops as the corpus matures, measured continuously rather than
gated against a target.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from shared.logging import get_logger

log = get_logger(__name__)

PROMPT = (
    "You are a debugging classifier. Given an incident and a list of "
    "candidate bug-class IDs, pick the single best match or 'none' if "
    "none fit. Reply ONLY with valid JSON (no markdown, no extra prose) "
    'in the form: {{"choice": <class_id|"none">, "rationale": <one sentence>}}.\n\n'
    "Incident:\n{incident}\n\nCandidates:\n{candidates}"
)


@dataclass(frozen=True)
class TiebreakerResult:
    """Outcome of a tiebreaker call.

    ``choice`` is the chosen class_id, or ``None`` when the LLM said
    ``"none"`` / errored / hallucinated. ``rationale`` carries a one-
    sentence justification on the success path or the error message
    on the failure path.

    Frozen so equality compares by value (tests assert
    ``out == TiebreakerResult(...)``) and the result is safely hashable
    if a future caller wants to memoize.
    """

    choice: str | None
    rationale: str


def resolve_ambiguity(
    *,
    anthropic: Any,
    incident: dict[str, object],
    candidates: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> TiebreakerResult:
    """Ask Haiku to pick one ``class_id`` from ``candidates`` for ``incident``.

    Args:
        anthropic: A sync Anthropic client (duck-typed — the tests pass
            ``MagicMock``). Must expose ``messages.create(...)`` returning
            an object whose ``.content[0].text`` is the model's response.
        incident: Free-form incident payload. Serialized via
            ``json.dumps(..., default=str)`` so datetime fields the spec
            allows in candidate payloads round-trip cleanly.
        candidates: Class IDs in contention. The function rejects any
            ``choice`` not in this list (defense-in-depth against
            hallucinated IDs).
        model: Anthropic model name. Defaults to the pinned Haiku 4.5
            snapshot used by spec §6 step 2 ("optional Haiku tiebreaker
            <500ms when it fires"). Overridable for eval / regression
            testing on alternate snapshots.

    Returns:
        ``TiebreakerResult``. ``choice`` is the picked class_id on the
        happy path, or ``None`` on any of: model said ``"none"``, API
        raised, JSON parse failed, or model hallucinated a class_id
        not in ``candidates``. ``rationale`` carries the model's
        justification on the happy path, or the error message / a note
        about the unknown ID on the failure paths.
    """
    prompt = PROMPT.format(
        incident=json.dumps(incident, default=str),
        candidates="\n".join(f"- {c}" for c in candidates),
    )
    try:
        msg = anthropic.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        parsed = json.loads(text)
        choice = parsed.get("choice")
        rationale = str(parsed.get("rationale", ""))
        if choice == "none" or choice is None:
            return TiebreakerResult(choice=None, rationale=rationale)
        if choice not in candidates:
            return TiebreakerResult(
                choice=None,
                rationale=f"LLM picked unknown class_id: {choice!r}",
            )
        return TiebreakerResult(choice=str(choice), rationale=rationale)
    except Exception as e:
        log.warning("kg.classifier.tiebreaker_failed", error=str(e))
        return TiebreakerResult(
            choice=None, rationale=f"tiebreaker error: {e}"
        )

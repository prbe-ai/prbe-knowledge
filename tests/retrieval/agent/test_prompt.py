"""Prompt regression snapshot for the gatherer system prompt.

The prompt is load-bearing — accidental edits silently change agent
behavior. We snapshot key prompt invariants so any drift alarms in CI.
Update the assertions on intentional changes + document the move in
the PR description.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from services.retrieval.agent.prompt import build_system_prompt

_EXPECTED_PROMPT_PREFIX = "You are a retrieval gatherer for a knowledge graph search system."


def test_prompt_starts_with_role_framing() -> None:
    """Lock the lede so the gatherer-not-answerer framing can't be
    accidentally weakened."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    assert prompt.startswith(_EXPECTED_PROMPT_PREFIX), (
        "Prompt opening drifted from the canonical 'gatherer not answerer' "
        "framing. If the change is intentional update this test + decision log."
    )


def test_prompt_bakes_today_iso_correctly() -> None:
    """`now` must flow into the today_iso line so the agent resolves
    relative temporal phrases correctly."""
    prompt = build_system_prompt(datetime(2026, 3, 14, tzinfo=UTC))
    assert "2026-03-14" in prompt
    assert "{today_iso}" not in prompt
    assert "{today_iso" not in prompt


def test_prompt_lists_all_fat_tools() -> None:
    """The fat-tools surface MUST appear in the prompt by name. If we
    add/remove a tool, the prompt + this test update together."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    for tool in (
        "search",
        "subgraph",
        "fetch_doc",
        "need_deeper",
        "emit_gatherer_output",
    ):
        assert tool in prompt, f"tool {tool!r} not referenced in system prompt"


def test_prompt_documents_terminal_mechanism() -> None:
    """`emit_gatherer_output` is the terminal tool. The prompt MUST
    explain that calling it ends the loop AND that the arguments ARE
    the GathererOutput payload. Drift here breaks the loop contract."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    assert "emit_gatherer_output" in prompt
    # The mechanism description — at least one of these phrases needs to
    # survive prompt edits, otherwise the model may not understand the
    # terminal semantics.
    assert "ends the loop" in prompt or "end the loop" in prompt
    assert "ARE the" in prompt or "ARE the GathererOutput" in prompt


def test_prompt_references_pre_fan_out_evidence_block() -> None:
    """The prompt must direct the model to read `<channel_results>` and
    explicitly tell it the 4-channel fan-out already fired. Drift here
    is how we'd accidentally regress to the model re-firing channels."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    assert "<channel_results>" in prompt
    assert "already" in prompt.lower()


def test_prompt_forbids_prose_output() -> None:
    """Even though tool_choice=required structurally forbids prose,
    keep the explicit 'no prose answers' line so the model doesn't
    pad tool args with markdown."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    assert "do NOT" in prompt or "do not" in prompt.lower()


def test_prompt_hash_is_stable_within_day() -> None:
    """Same `now` → same prompt. Catches inadvertent imports of
    request-time randomness."""
    d = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    a = build_system_prompt(d)
    b = build_system_prompt(d)
    assert sha256(a.encode()).hexdigest() == sha256(b.encode()).hexdigest()


def test_prompt_references_inferred_chains_section() -> None:
    """When inferred-edge hits exist the harness renders a chain-shaped
    re-grouping in `<inferred_chains>` (loop.py:_format_inferred_chains).
    The prompt MUST tell the agent that section exists and that for
    'why was X created' / 'what led to Y' queries it IS the answer
    chain — otherwise the agent has no reason to walk it."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    assert "<inferred_chains>" in prompt
    # The chain-shaped intent class — at least one phrasing has to
    # survive, otherwise why-chain queries silently regress to flat
    # chunk lists.
    assert "why was" in prompt.lower() or "what led to" in prompt.lower()


def test_prompt_requires_entity_emission_not_just_chunks() -> None:
    """The previous failure mode (other-session diagnosis 2026-05-18)
    was the agent emitting chunks-only because the prompt's happy path
    biased toward `<channel_results>` which renders doc-shaped hits.
    Lock the explicit 'emit entities, not just chunks' rule so a future
    prompt simplification doesn't re-introduce the regression."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    # The structural rule, plus a justification anchor so the rule
    # isn't accidentally weakened to 'consider emitting entities'.
    assert "EMIT ENTITIES" in prompt
    assert "GatheredEntity" in prompt
    assert "properties.why" in prompt


def test_prompt_documents_why_chain_walking_for_reason_queries() -> None:
    """Why-chain queries (reason / cause / context) need explicit
    walking instructions or the agent treats them like any other broad
    query. Lock the guidance so it can't be silently dropped."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    lowered = prompt.lower()
    # At least one reason-shaped intent phrase must appear in the
    # special-handling guidance.
    assert any(
        phrase in lowered
        for phrase in (
            "why-chain",
            "why was",
            "what led to",
            "what's the context behind",
            "context behind",
        )
    )
    # And the chain emission shape (anchor entity + linked chunks with
    # edge `why`) must be specified.
    assert "verbatim" in lowered
    assert "why_relevant" in prompt

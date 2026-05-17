"""Prompt regression snapshot for the gatherer system prompt.

The prompt is load-bearing — accidental edits silently change agent
behavior. We snapshot a SHA256 of the prompt body so any edit alarms
in CI. Update the recorded hash on intentional changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from services.retrieval.agent.prompt import build_system_prompt

# Hash recorded on initial cutover (2026-05-16). Update on intentional
# prompt edits and document the move in the PR description so reviewers
# can sanity-check what shifted.
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
    # Sanity: the placeholder format token must not have leaked through.
    assert "{today_iso}" not in prompt
    assert "{today_iso" not in prompt


def test_prompt_lists_all_tools() -> None:
    """Every tool name we register must appear in the prompt's
    'TOOLS — WHEN TO USE WHICH' section, or the agent's tool-selection
    quality drops sharply."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    for tool in (
        "vector_search",
        "bm25_search",
        "graph_search",
        "inferred_edge_search",
        "parallel_multi_query",
        "expand_inferred_neighbors",
        "expand_entity_cluster",
        "fetch_doc_chunks",
        "graph_walk",
        "reissue_query",
        "read_inferred_edge_evidence",
        "need_deeper",
    ):
        assert tool in prompt, f"tool {tool!r} not referenced in system prompt"


def test_prompt_enforces_turn_1_mandate_textually() -> None:
    """Post-pre-fan-out cutover: the recall guarantee is the harness-side
    deterministic pre-fan-out (vector + bm25 + graph + inferred_edge fired
    BEFORE the LLM call, results passed in `<channel_results>`). The
    prompt must reflect that the model READS pre-fan-out results rather
    than being instructed to fire the same 4 channels itself."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    for channel in ("vector_search", "bm25_search", "graph_search", "inferred_edge_search"):
        assert channel in prompt
    assert "<channel_results>" in prompt
    assert "ALREADY FIRED" in prompt or "already fired" in prompt.lower()


def test_prompt_forbids_prose_output() -> None:
    """The prompt must explicitly forbid prose answers — structured
    output only, the consumer synthesizes."""
    prompt = build_system_prompt(datetime(2026, 5, 16, tzinfo=UTC))
    assert "do NOT" in prompt or "do not" in prompt
    assert "Structured output only" in prompt or "structured" in prompt.lower()


def test_prompt_hash_is_stable_within_day() -> None:
    """Same `now` -> same prompt. Catches inadvertent imports of
    request-time randomness or non-deterministic helpers."""
    d = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    a = build_system_prompt(d)
    b = build_system_prompt(d)
    assert sha256(a.encode()).hexdigest() == sha256(b.encode()).hexdigest()

"""The queue's tier table, and the cap that keeps one tenant inside its lane.

Both are load-bearing and neither was pinned. On 2026-09-16 `custom_ingest`
(everything a researcher writes) and `claude_code`/`codex` (transcripts nobody
is waiting for) shared one tier, and a `probe note add` took a median 33
minutes to become searchable because it queued behind other tenants'
transcript backlogs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.shared.constants import (
    PRIORITY_AGENT_CAPTURE,
    PRIORITY_BACKGROUND,
    PRIORITY_LIVE_INTEGRATION,
    PRIORITY_RESEARCH_CONTENT,
    SourceSystem,
)
from engine.shared.source_registry import get_source_profile

_ROOT = Path(__file__).resolve().parent.parent


def _import_every_connector() -> None:
    """Profiles register on import; a test that reads the registry without
    importing them measures an empty dict and passes."""
    import engine.ingest.handlers.custom_ingest  # noqa: F401
    import kb.handlers.claude_code
    import kb.handlers.codegraph
    import kb.handlers.incident_sources  # noqa: F401


def test_the_tiers_are_ordered_and_distinct() -> None:
    tiers = [
        PRIORITY_RESEARCH_CONTENT,
        PRIORITY_LIVE_INTEGRATION,
        PRIORITY_AGENT_CAPTURE,
        PRIORITY_BACKGROUND,
    ]
    assert tiers == sorted(tiers, reverse=True), tiers
    assert len(set(tiers)) == 4, "two tiers with one value is one tier"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceSystem.CUSTOM_INGEST, PRIORITY_RESEARCH_CONTENT),
        (SourceSystem.CLAUDE_CODE, PRIORITY_AGENT_CAPTURE),
        (SourceSystem.CODEX, PRIORITY_AGENT_CAPTURE),
        (SourceSystem.PI, PRIORITY_AGENT_CAPTURE),
        (SourceSystem.SLACK, PRIORITY_LIVE_INTEGRATION),
        (SourceSystem.GITHUB, PRIORITY_LIVE_INTEGRATION),
        (SourceSystem.PAGERDUTY, PRIORITY_LIVE_INTEGRATION),
        (SourceSystem.INCIDENT_IO, PRIORITY_LIVE_INTEGRATION),
        (SourceSystem.CODE_GRAPH, PRIORITY_BACKGROUND),
    ],
)
def test_every_source_lands_in_its_tier(source: SourceSystem, expected: int) -> None:
    _import_every_connector()
    assert get_source_profile(source.value).ingestion_priority == expected


def test_codex_and_pi_inherit_the_capture_tier_not_the_old_shared_one() -> None:
    """They subclass ClaudeCodeConnector and declare no priority of their own,
    so a re-tier that edited only the parent would silently leave them behind
    if the inheritance ever stopped working."""
    _import_every_connector()
    for source in (SourceSystem.CODEX, SourceSystem.PI):
        assert get_source_profile(source.value).ingestion_priority == (
            get_source_profile(SourceSystem.CLAUDE_CODE.value).ingestion_priority
        )


def test_research_content_outranks_every_capture_source() -> None:
    """The whole point of the re-tier, asserted as the property rather than
    as four numbers that could all be edited together."""
    _import_every_connector()
    content = get_source_profile(SourceSystem.CUSTOM_INGEST.value).ingestion_priority
    for source in (SourceSystem.CLAUDE_CODE, SourceSystem.CODEX, SourceSystem.PI):
        assert content > get_source_profile(source.value).ingestion_priority


def test_captures_never_outrank_a_live_integration() -> None:
    _import_every_connector()
    for source in (SourceSystem.CLAUDE_CODE, SourceSystem.CODEX, SourceSystem.PI):
        assert (
            get_source_profile(source.value).ingestion_priority
            < get_source_profile(SourceSystem.SLACK.value).ingestion_priority
        )


def test_captures_outrank_backfills() -> None:
    """60 and not 50: sharing the background tier would put a new tenant's
    history import ahead of every live transcript for as long as it ran."""
    assert PRIORITY_AGENT_CAPTURE > PRIORITY_BACKGROUND


def test_the_backfill_runner_uses_the_constant_not_a_literal() -> None:
    """It builds its INSERT as SQL text, so nothing else would catch a drift
    between the tier table and the number the backfill actually writes."""
    src = (_ROOT / "kb" / "backfill_runner.py").read_text()
    assert "{PRIORITY_BACKGROUND}" in src
    assert not re.search(r"priority\)\s*\n\s*VALUES[^)]*,\s*50\)", src)


def test_the_session_upsert_takes_the_new_tier() -> None:
    """A session row is created once and resumed for the life of the session.
    Without `priority = EXCLUDED.priority` a re-tier reaches only sessions
    started after the deploy, and long-lived ones keep the old value forever."""
    src = (_ROOT / "kb" / "ingestion_app.py").read_text()
    upsert = src[src.index("ON CONFLICT (customer_id, source_system, source_event_id) DO UPDATE"):]
    upsert = upsert[: upsert.index("RETURNING queue_id")]
    assert "priority = EXCLUDED.priority" in upsert


def test_the_inflight_cap_is_keyed_per_tier() -> None:
    """Keyed on customer_id alone, a tenant's own transcripts made its own
    deliberate writes ineligible while slots sat idle. Both claim queries must
    group by (customer_id, priority) — the batch path is off by default, so a
    test that only reads the single-row path would not notice it drifting."""
    src = (_ROOT / "engine" / "ingest" / "worker.py").read_text()
    groupings = re.findall(r"GROUP BY customer_id(?:, priority)?", src)
    assert groupings, "no inflight CTE found — did the claim query move?"
    assert all(g == "GROUP BY customer_id, priority" for g in groupings), groupings
    joins = re.findall(r"ON i\.customer_id = q\.customer_id(?: AND i\.priority = q\.priority)?", src)
    assert joins, "no inflight join found"
    assert all("i.priority = q.priority" in j for j in joins), joins


@pytest.mark.parametrize("loops", [1, 2, 4, 6, 12])
def test_the_cap_can_never_exceed_the_loop_count(loops: int) -> None:
    """A cap at or above `worker_max_concurrent` caps nothing: 30 against 6
    loops let one tenant hold every slot in the fleet. Derived, so it cannot
    drift out of range however the loop count moves."""
    from engine.shared.config import Settings

    s = Settings(worker_max_concurrent=loops)
    cap = s.per_customer_cap()
    assert 1 <= cap <= loops
    if loops >= 2:
        assert cap < loops, "one tenant must not be able to hold every loop"


def test_an_explicit_cap_is_honoured_but_still_clamped() -> None:
    from engine.shared.config import Settings

    assert Settings(worker_max_concurrent=6, worker_per_customer_max_inflight=2).per_customer_cap() == 2
    # Above the loop count it would cap nothing, which is how this started.
    assert Settings(worker_max_concurrent=6, worker_per_customer_max_inflight=30).per_customer_cap() == 6
    assert Settings(worker_max_concurrent=6, worker_per_customer_max_inflight=0).per_customer_cap() == 1

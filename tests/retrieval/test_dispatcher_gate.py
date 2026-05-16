"""Dispatcher gate-verify tests.

The dispatcher trusts Haiku's `mode` field but re-checks the gate locally:
even if Haiku says `mode=list`, we route to search if the inputs don't
actually satisfy the gate. This catches bad Haiku snapshots and protects
search-correct queries from being silently dispatched to SQL.
"""

from __future__ import annotations

from datetime import UTC, datetime

from services.retrieval.pipeline import _gate_verify_list
from services.retrieval.router import Intent, RouterEntity
from shared.models import TemporalMode, TemporalSpec


def _spec_latest() -> TemporalSpec:
    return TemporalSpec()  # default: LATEST, no since/until


def _spec_changed_between() -> TemporalSpec:
    return TemporalSpec(
        mode=TemporalMode.CHANGED_BETWEEN,
        since=datetime(2026, 4, 20, tzinfo=UTC),
        until=datetime(2026, 4, 28, tzinfo=UTC),
    )


def test_gate_passes_with_sort_only() -> None:
    intent = Intent(
        query_text="show recent commits",
        mode="list",
        confidence=0.9,
        sort={"field": "updated_at", "direction": "desc"},
        entities=[],
    )
    assert _gate_verify_list(intent, _spec_latest()) is True


def test_gate_passes_with_temporal_only() -> None:
    intent = Intent(
        query_text="commits last week",
        mode="list",
        confidence=0.9,
        sort=None,
        entities=[],
    )
    assert _gate_verify_list(intent, _spec_changed_between()) is True


def test_gate_fails_with_no_sort_and_no_temporal() -> None:
    intent = Intent(
        query_text="what is auth",
        mode="list",
        confidence=0.9,
        sort=None,
        entities=[],
    )
    assert _gate_verify_list(intent, _spec_latest()) is False


def test_gate_fails_with_topic_entity_even_with_sort() -> None:
    """Hybrid query: 'most recent commits about auth' — sort is set, but
    `auth` is a TOPIC (feature) entity. Must route to search."""
    intent = Intent(
        query_text="most recent commits about auth",
        mode="list",
        confidence=0.9,
        sort={"field": "updated_at", "direction": "desc"},
        entities=[
            RouterEntity(
                entity_type="feature",
                canonical_id="auth",
                display_name="auth",
                confidence=0.7,
            )
        ],
    )
    assert _gate_verify_list(intent, _spec_latest()) is False


def test_gate_passes_with_narrowing_entity_and_sort() -> None:
    """'show me the latest commits to auth.py' — file_path is NARROWING,
    not topic. Should pass the gate to list mode."""
    intent = Intent(
        query_text="show me the latest commits to auth.py",
        mode="list",
        confidence=0.9,
        sort={"field": "updated_at", "direction": "desc"},
        entities=[
            RouterEntity(
                entity_type="file_path",
                canonical_id="auth.py",
                display_name="auth.py",
                confidence=0.95,
            )
        ],
    )
    assert _gate_verify_list(intent, _spec_latest()) is True


def test_gate_fails_with_decision_topic_entity() -> None:
    intent = Intent(
        query_text="recent commits about the database decision",
        mode="list",
        confidence=0.9,
        sort={"field": "updated_at", "direction": "desc"},
        entities=[
            RouterEntity(
                entity_type="decision",
                canonical_id="d1",
                display_name="d1",
                confidence=0.9,
            )
        ],
    )
    assert _gate_verify_list(intent, _spec_latest()) is False


def test_gate_fails_with_error_group_topic_entity() -> None:
    intent = Intent(
        query_text="commits related to error eg1",
        mode="list",
        confidence=0.9,
        sort=None,
        entities=[
            RouterEntity(
                entity_type="error_group",
                canonical_id="eg1",
                display_name="eg1",
                confidence=0.9,
            )
        ],
    )
    assert _gate_verify_list(intent, _spec_changed_between()) is False

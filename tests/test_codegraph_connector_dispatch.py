"""Unit tests for the CodeGraphConnector's dispatch logic.

Tests parse_webhook_event and verify_signature without touching the DB.
Heavier integration tests (clone + walk + extract → real DB rows) belong
in tests/integration.
"""

from __future__ import annotations

import httpx
import pytest

from engine.ingest.handlers.base import ConnectorContext
from engine.shared.config import Settings
from engine.shared.exceptions import InvalidWebhookPayload
from kb.handlers.codegraph import (
    KIND_DISCONNECT,
    KIND_INCREMENTAL,
    KIND_INITIAL_BACKFILL,
    CodeGraphConnector,
    _recompute_event_id,
)


def _connector() -> CodeGraphConnector:
    settings = Settings()
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return CodeGraphConnector(ctx)


def test_verify_signature_always_false() -> None:
    """code_graph has no public webhook surface — verify_signature returns
    False so standalone /webhooks/code_graph hard-401s. Events are enqueued
    in-process by code_graph/bridge.py, never over HTTP."""
    c = _connector()
    assert c.verify_signature({}, b"") is False
    assert c.verify_signature({"x-github-event": "push"}, b"junk") is False


def test_parse_recognizes_initial_backfill_kind() -> None:
    c = _connector()
    payload = {
        "kind": KIND_INITIAL_BACKFILL,
        "repo": "org/repo",
        "sha": "abc123",
    }
    result = c.parse_webhook_event("cust", {}, payload)
    assert result is not None
    assert result.source_event_id == "code_graph:backfill:org/repo:abc123"


def test_parse_recognizes_incremental_kind() -> None:
    c = _connector()
    payload = {
        "kind": KIND_INCREMENTAL,
        "repo": "org/repo",
        "sha": "deadbeef",
    }
    result = c.parse_webhook_event("cust", {}, payload)
    assert result is not None
    assert result.source_event_id == "code_graph:incremental:org/repo:deadbeef"


def test_parse_recognizes_disconnect_kind() -> None:
    c = _connector()
    payload = {
        "kind": KIND_DISCONNECT,
        "repos": ["org/a", "org/b"],
        "enqueued_at": "2026-05-03T12:00:00+00:00",
    }
    result = c.parse_webhook_event("cust", {}, payload)
    assert result is not None
    # Disconnect event_id includes sorted repo set + timestamp.
    assert "code_graph:disconnect:" in result.source_event_id
    assert "org/a+org/b" in result.source_event_id
    assert "2026-05-03T12:00:00+00:00" in result.source_event_id


def test_parse_unknown_kind_raises() -> None:
    c = _connector()
    with pytest.raises(InvalidWebhookPayload, match="unknown code_graph"):
        c.parse_webhook_event("cust", {}, {"kind": "fanout"})


def test_recompute_event_id_disconnect_sorts_repos() -> None:
    """Different orderings of the same repo set produce the same event_id."""
    p1 = {
        "kind": KIND_DISCONNECT,
        "repos": ["b/repo", "a/repo"],
        "enqueued_at": "2026-05-03",
    }
    p2 = {
        "kind": KIND_DISCONNECT,
        "repos": ["a/repo", "b/repo"],
        "enqueued_at": "2026-05-03",
    }
    assert _recompute_event_id(p1) == _recompute_event_id(p2)

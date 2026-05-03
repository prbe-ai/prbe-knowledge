"""Unit tests for the Wiki connector + wiki-link parser.

Covers:
- parse_wiki_links across plain / typed / malformed / Unicode targets
- slugify round-trip
- WikiConnector.parse_webhook_event for upsert + delete + bad payloads
- build_normalization_result emits Document, graph_nodes/edges, ACL correctly
- doc_id is deterministic by (customer, source_system, source_id)
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.registry import build_connector
from services.ingestion.handlers.wiki import (  # noqa: F401 — registers the connector
    WIKI_PAYLOAD_KEY,
    WikiConnector,
    build_normalization_result,
)
from services.ingestion.wiki_links import parse_wiki_links, slugify
from shared.config import Settings
from shared.constants import (
    CompileTrigger,
    DocClass,
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.models import WebhookEvent


def _make_ctx() -> ConnectorContext:
    return ConnectorContext(
        settings=Settings(environment="local"),
        http=httpx.AsyncClient(),
    )


def _make_event(payload: dict, *, customer_id: str = "cust-1") -> WebhookEvent:
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.WIKI,
        source_event_id="evt-x",
        received_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
        payload_s3_key="",
        payload_s3_keys=[],
        raw_payload={WIKI_PAYLOAD_KEY: payload},
        headers={},
    )


# ---------------------------------------------------------------------------
# wiki_links parser
# ---------------------------------------------------------------------------


def test_parse_wiki_links_plain() -> None:
    body = "See [[onboarding guide]] for the steps."
    [link] = parse_wiki_links(body)
    assert link.kind == "plain"
    assert link.target == "onboarding guide"
    assert link.raw == "[[onboarding guide]]"
    assert body[link.span[0] : link.span[1]] == link.raw


def test_parse_wiki_links_typed_kinds() -> None:
    body = (
        "Ping [[Person: mahit]] and [[Person: rich]] when [[Service: prbe-knowledge]] "
        "is degraded. See [[Decision: pgvector]] and [[Repo: prbe-backend]] "
        "and [[Ticket: PRB-9]] and [[Feature: auth]]."
    )
    links = parse_wiki_links(body)
    by_kind = {(link.kind, link.target) for link in links}
    assert ("person", "mahit") in by_kind
    assert ("person", "rich") in by_kind
    assert ("service", "prbe-knowledge") in by_kind
    assert ("decision", "pgvector") in by_kind
    assert ("repo", "prbe-backend") in by_kind
    assert ("ticket", "PRB-9") in by_kind
    assert ("feature", "auth") in by_kind


def test_parse_wiki_links_unknown_kind_is_plain() -> None:
    body = "[[Sentinel: alpha]] but [[plain target]]."
    links = parse_wiki_links(body)
    assert links[0].kind == "plain"
    # Unknown prefix preserves the full inner text so the dangling-link surfacer
    # can still flag it.
    assert links[0].target == "Sentinel: alpha"
    assert links[1].kind == "plain"


def test_parse_wiki_links_empty_and_unicode() -> None:
    assert parse_wiki_links("") == []
    assert parse_wiki_links("no links here") == []

    body = "Run the [[Service: café-export]] job — coordinate with [[Person: 田中]]."
    [svc, person] = parse_wiki_links(body)
    assert svc.target == "café-export"
    assert person.target == "田中"


def test_parse_wiki_links_ignores_empty_brackets() -> None:
    assert parse_wiki_links("[[]]") == []
    # Colon with no target collapses to plain so the raw form is preserved.
    [link] = parse_wiki_links("[[Person:]]")
    assert link.kind == "plain"


def test_slugify_basic() -> None:
    assert slugify("Slack Backfill Stuck") == "slack-backfill-stuck"
    assert slugify("  Hello, World!  ") == "hello-world"
    # ASCII fold for accented input.
    assert slugify("Café Export") == "cafe-export"
    # Empty / pure-symbol input collapses to "".
    assert slugify("!!!") == ""


# ---------------------------------------------------------------------------
# Connector — parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_webhook_event_upsert() -> None:
    connector = build_connector(SourceSystem.WIKI, _make_ctx())
    parsed = connector.parse_webhook_event(
        "cust-1",
        {},
        {
            WIKI_PAYLOAD_KEY: {
                "wiki_type": "runbook",
                "slug": "slack-backfill-stuck",
                "title": "Slack backfill stuck",
                "body": "When stuck...",
                "updated_at": "2026-05-01T12:00:00Z",
            }
        },
    )
    assert parsed is not None
    assert parsed.event_kind == IngestionEventType.MANUAL
    assert parsed.source_event_id == ("runbook:slack-backfill-stuck:edit:2026-05-01T12:00:00Z")
    assert parsed.parse_hint["wiki_type"] == "runbook"
    assert parsed.parse_hint["is_delete"] is False


def test_parse_webhook_event_delete() -> None:
    connector = build_connector(SourceSystem.WIKI, _make_ctx())
    parsed = connector.parse_webhook_event(
        "cust-1",
        {},
        {
            WIKI_PAYLOAD_KEY: {
                "wiki_type": "decision",
                "slug": "pgvector",
                "is_delete": True,
                "updated_at": "2026-05-01T12:00:00Z",
            }
        },
    )
    assert parsed is not None
    assert parsed.source_event_id.startswith("decision:pgvector:delete:")
    assert parsed.parse_hint["is_delete"] is True


def test_parse_webhook_event_unknown_type_raises() -> None:
    connector = build_connector(SourceSystem.WIKI, _make_ctx())
    with pytest.raises(InvalidWebhookPayload):
        connector.parse_webhook_event(
            "cust-1",
            {},
            {WIKI_PAYLOAD_KEY: {"wiki_type": "bogus", "slug": "x"}},
        )


def test_parse_webhook_event_missing_payload_raises() -> None:
    connector = build_connector(SourceSystem.WIKI, _make_ctx())
    with pytest.raises(InvalidWebhookPayload):
        connector.parse_webhook_event("cust-1", {}, {"unrelated": True})


def test_verify_signature_returns_false() -> None:
    # No external webhook surface — /webhooks/wiki must always 401.
    connector = build_connector(SourceSystem.WIKI, _make_ctx())
    assert connector.verify_signature({}, b"payload") is False


# ---------------------------------------------------------------------------
# build_normalization_result
# ---------------------------------------------------------------------------


def test_build_normalization_result_runbook_with_typed_links() -> None:
    body = (
        "When the Slack backfill stalls, ping [[Person: mahit]] and check "
        "[[Service: prbe-knowledge]]. See [[Decision: serialize-cc-claims]] "
        "and the [[plain runbook]] for the full sequence."
    )
    event = _make_event(
        {
            "wiki_type": "runbook",
            "slug": "slack-backfill-stuck",
            "title": "Slack backfill stuck",
            "body": body,
            "frontmatter": {"owner": "mahit", "severity": "high"},
            "updated_at": "2026-05-01T12:00:00Z",
        }
    )
    result = build_normalization_result(event)

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_id == "wiki:runbook:slack-backfill-stuck"
    assert doc.source_system == SourceSystem.WIKI
    assert doc.source_id == "runbook:slack-backfill-stuck"
    assert doc.source_url == "/wiki/runbook/slack-backfill-stuck"
    assert doc.doc_type == DocType.WIKI_RUNBOOK
    assert doc.doc_class == DocClass.MANUAL_ENTRY
    assert doc.title == "Slack backfill stuck"
    assert doc.metadata["body"] == body
    assert doc.metadata["frontmatter"] == {"owner": "mahit", "severity": "high"}
    assert doc.metadata["dangling_links"] == ["[[plain runbook]]"]
    assert doc.compiled_from_doc_ids is None
    assert doc.compile_trigger is None
    assert doc.deleted_at is None
    assert doc.body_token_count > 0

    # ACL: workspace-read.
    [principal] = doc.acl.principals
    assert principal.principal_type == PrincipalType.WORKSPACE
    assert principal.principal_id == "cust-1"
    assert principal.permission == Permission.READ

    # Graph: DOCUMENT node + one node per typed link, deduped on (label, id).
    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels
    assert (NodeLabel.PERSON, "mahit") in labels
    assert (NodeLabel.SERVICE, "prbe-knowledge") in labels
    assert (NodeLabel.DECISION, "serialize-cc-claims") in labels

    edge_keys = {
        (e.edge_type, e.from_canonical_id, e.to_label, e.to_canonical_id)
        for e in result.graph_edges
    }
    assert (
        EdgeType.MENTIONS,
        doc.doc_id,
        NodeLabel.PERSON,
        "mahit",
    ) in edge_keys
    assert (
        EdgeType.DESCRIBES,
        doc.doc_id,
        NodeLabel.SERVICE,
        "prbe-knowledge",
    ) in edge_keys
    assert (
        EdgeType.DESCRIBES,
        doc.doc_id,
        NodeLabel.DECISION,
        "serialize-cc-claims",
    ) in edge_keys

    # ACL snapshot row.
    [acl_row] = result.acl_snapshots
    assert acl_row.source_system == SourceSystem.WIKI
    assert acl_row.principal_type == PrincipalType.WORKSPACE
    assert acl_row.resource_id == doc.doc_id


def test_doc_id_is_deterministic() -> None:
    base_payload = {
        "wiki_type": "decision",
        "slug": "adopt-pgvector",
        "title": "Adopt pgvector",
        "body": "We adopt pgvector.",
        "updated_at": "2026-05-01T12:00:00Z",
    }
    a = build_normalization_result(_make_event(base_payload, customer_id="cust-1"))
    b = build_normalization_result(_make_event(base_payload, customer_id="cust-1"))
    assert a.documents[0].doc_id == b.documents[0].doc_id
    # Different customer = same doc_id (the customer_id scoping happens at
    # the persistence layer, not in doc_id construction — matches notion).
    c = build_normalization_result(_make_event(base_payload, customer_id="cust-2"))
    assert a.documents[0].doc_id == c.documents[0].doc_id
    assert c.documents[0].customer_id == "cust-2"


def test_build_delete_emits_tombstone() -> None:
    event = _make_event(
        {
            "wiki_type": "runbook",
            "slug": "stale",
            "title": "",
            "body": "",
            "is_delete": True,
            "updated_at": "2026-05-01T12:00:00Z",
        }
    )
    result = build_normalization_result(event)
    doc = result.documents[0]
    assert doc.deleted_at is not None
    assert doc.body_size_bytes == 0
    assert doc.metadata["dangling_links"] == []
    # Only the DOCUMENT node — no link-derived nodes on a delete.
    [node] = result.graph_nodes
    assert node.label == NodeLabel.DOCUMENT


def test_build_compiled_wiki_sets_compile_trigger() -> None:
    event = _make_event(
        {
            "wiki_type": "service_card",
            "slug": "prbe-knowledge",
            "title": "prbe-knowledge service card",
            "body": "Owns retrieval + ingestion for [[Repo: prbe-knowledge]].",
            "doc_class": DocClass.COMPILED_WIKI.value,
            "compiled_from_doc_ids": ["github:commit:abc"],
            "compile_trigger": CompileTrigger.SOURCE_UPDATE.value,
            "updated_at": "2026-05-01T12:00:00Z",
        }
    )
    result = build_normalization_result(event)
    doc = result.documents[0]
    assert doc.doc_class == DocClass.COMPILED_WIKI
    assert doc.compiled_from_doc_ids == ["github:commit:abc"]
    assert doc.compile_trigger == CompileTrigger.SOURCE_UPDATE
    assert doc.compiled_at == event.received_at


def test_build_invalid_doc_class_raises() -> None:
    event = _make_event(
        {
            "wiki_type": "runbook",
            "slug": "x",
            "title": "x",
            "body": "x",
            "doc_class": "bogus",
            "updated_at": "2026-05-01T12:00:00Z",
        }
    )
    with pytest.raises(InvalidWebhookPayload):
        build_normalization_result(event)


def test_build_dedupes_repeated_typed_links() -> None:
    event = _make_event(
        {
            "wiki_type": "feature",
            "slug": "auth",
            "title": "Auth",
            "body": "[[Person: mahit]] [[Person: mahit]] [[Person: mahit]]",
            "updated_at": "2026-05-01T12:00:00Z",
        }
    )
    result = build_normalization_result(event)
    # One DOCUMENT + one Person node (deduped), three MENTIONS edges (one per link).
    person_nodes = [n for n in result.graph_nodes if n.label == NodeLabel.PERSON]
    assert len(person_nodes) == 1
    person_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.MENTIONS]
    assert len(person_edges) == 3

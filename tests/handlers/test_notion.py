"""Unit tests for the Notion connector.

Covers:
- parse_webhook_event on real Notion 2025-09-03 events: page.properties_updated,
  page.content_updated, page.deleted, database.schema_updated,
  data_source.content_updated, data_source.deleted
- parse_webhook_event returns None for deferred (comment.*) and unknown event types
- blocks_to_markdown on a mix of block types
- normalize without hydration (no token) → workspace ACL only, empty body
- normalize with hydrated content → body populated, mentions → PERSON nodes,
  permissions → granular ACL rows
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from engine.ingest.handlers.base import ConnectorContext
from engine.ingest.handlers.registry import build_connector
from engine.shared.config import Settings
from engine.shared.constants import (
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.models import WebhookEvent
from kb.handlers.notion import (  # noqa: F401 — registers
    NotionConnector,
    blocks_to_markdown,
)

FIXTURES = Path(__file__).resolve().parents[1].parent / "fixtures" / "notion"


def _load(name: str) -> dict:
    with (FIXTURES / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_ctx(env: str = "local") -> ConnectorContext:
    settings = Settings(environment=env)
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


# ---------------------------------------------------------------------------
# parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_webhook_event_page_properties_updated() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    payload = _load("page_properties_updated.json")
    result = notion.parse_webhook_event("cust-1", {}, payload)

    assert result is not None
    # Notion's documented `data` shape for page.properties_updated does NOT
    # include `last_edited_time` — the parser falls back to the top-level
    # `timestamp`. Same value here, asserted to lock that fallback in.
    assert result.source_event_id.startswith(
        "page:page_abc123:edit:2026-04-22T12:00:00.000Z:"
    )
    # Trailing :<16-hex> is a stable payload fingerprint disambiguating rapid
    # same-second edits.
    assert len(result.source_event_id.rsplit(":", 1)[-1]) == 16
    assert result.event_kind == IngestionEventType.WEBHOOK
    assert result.parse_hint["resource_type"] == "page"
    assert result.parse_hint["resource_id"] == "page_abc123"
    assert result.parse_hint["workspace_id"] == "ws_TEST"
    assert result.parse_hint["event_type"] == "page.properties_updated"
    assert result.parse_hint["is_delete"] is False
    assert result.received_at.tzinfo is not None


def test_parse_webhook_event_page_content_updated() -> None:
    """page.content_updated is the high-frequency edit event — every block
    edit on a page funnels through here. Pre-fix, this fell into the unknown
    bucket and was silently dropped."""
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "page.content_updated",
            "timestamp": "2026-04-22T12:00:00.000Z",
            "workspace_id": "ws_TEST",
            "entity": {"type": "page", "id": "page_abc123"},
            "data": {
                "parent": {"id": "page_parent_999", "type": "page"},
                "updated_blocks": [
                    {"id": "blk_1", "type": "block"},
                    {"id": "blk_2", "type": "block"},
                ],
            },
        },
    )
    assert result is not None
    assert result.parse_hint["event_type"] == "page.content_updated"
    assert result.parse_hint["resource_type"] == "page"
    assert result.parse_hint["is_delete"] is False


def test_parse_webhook_event_database_schema_updated() -> None:
    """database.schema_updated is deprecated in 2025-09-03 but still emitted
    to subscriptions on older API versions, so we keep it accepted."""
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "database.schema_updated",
            "timestamp": "2026-04-22T12:00:00.000Z",
            "workspace_id": "ws_TEST",
            "entity": {"type": "database", "id": "db_abc"},
            "data": {
                "parent": {"id": "page_p", "type": "page"},
                "updated_properties": [
                    {"id": "prop_1", "name": "Status", "action": "created"}
                ],
            },
        },
    )
    assert result is not None
    assert result.parse_hint["resource_type"] == "database"
    assert result.parse_hint["is_delete"] is False


def test_parse_webhook_event_data_source_content_updated() -> None:
    """data_source.* events are the 2025-09-03 replacement for the
    deprecated database.content_updated / database.schema_updated. Must be
    accepted (not dropped) even though hydration support is a follow-up."""
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "data_source.content_updated",
            "timestamp": "2026-04-22T12:00:00.000Z",
            "workspace_id": "ws_TEST",
            "entity": {"type": "data_source", "id": "ds_abc"},
            "data": {
                "parent": {"id": "page_p", "type": "page"},
                "updated_blocks": [{"id": "blk_1", "type": "block"}],
            },
        },
    )
    assert result is not None
    assert result.parse_hint["resource_type"] == "data_source"
    assert result.parse_hint["is_delete"] is False


def test_parse_webhook_event_page_deleted_produces_tombstone() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "page.deleted",
            "entity": {"type": "page", "id": "x"},
            "timestamp": "2026-04-22T12:00:00.000Z",
        },
    )
    assert result is not None
    assert result.source_event_id.startswith("page:x:delete:")
    assert result.parse_hint["is_delete"] is True


def test_parse_webhook_event_data_source_deleted_produces_tombstone() -> None:
    """data_source.deleted is the new-API analogue of database.deleted +
    must produce a tombstone too, otherwise deletes leak."""
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "data_source.deleted",
            "entity": {"type": "data_source", "id": "ds_x"},
            "timestamp": "2026-04-22T12:00:00.000Z",
        },
    )
    assert result is not None
    assert result.source_event_id.startswith("data_source:ds_x:delete:")
    assert result.parse_hint["is_delete"] is True


def test_parse_webhook_event_comment_events_deferred() -> None:
    """Comment events are recognized but not yet ingested — parse must
    return None (skipped), not raise. Locks in the deferred-events
    behavior so a future spec change shows up as a failing test rather
    than as silent drops in prod."""
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    for event_type in ("comment.created", "comment.updated", "comment.deleted"):
        result = notion.parse_webhook_event(
            "cust-1",
            {},
            {
                "type": event_type,
                "timestamp": "2026-04-22T12:00:00.000Z",
                "workspace_id": "ws_TEST",
                "entity": {"type": "comment", "id": "cmt_x"},
                "data": {
                    "page_id": "page_abc123",
                    "parent": {"id": "page_abc123", "type": "page"},
                },
            },
        )
        assert result is None, f"{event_type} must be deferred (returned None)"


def test_parse_webhook_event_ignores_unknown_type() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    assert (
        notion.parse_webhook_event(
            "cust-1",
            {},
            {"type": "user.created", "entity": {"type": "user", "id": "x"}},
        )
        is None
    )


def test_parse_synthetic_poll_shape() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "customer_id": "cust-1",
            "resource_type": "page",
            "resource_id": "page_synth",
            "polled_at": "2026-04-22T11:00:00Z",
            "last_edited_time": "2026-04-22T10:59:00Z",
        },
    )

    assert result is not None
    assert result.source_event_id == "page:page_synth:2026-04-22T10:59:00Z"
    assert result.event_kind == IngestionEventType.SYNC


def test_parse_rejects_unknown_shape() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    from engine.shared.exceptions import InvalidWebhookPayload

    with pytest.raises(InvalidWebhookPayload):
        notion.parse_webhook_event("cust-1", {}, {"foo": "bar"})


# ---------------------------------------------------------------------------
# blocks_to_markdown
# ---------------------------------------------------------------------------


def test_blocks_to_markdown_mixed_types() -> None:
    fx = _load("blocks.json")
    md = blocks_to_markdown(fx["results"])

    # Ordered assertions — the mapping should be stable.
    assert "## Incident runbook" in md
    assert "When payments 500s, page @Bob immediately." in md
    assert "- Check Stripe dashboard" in md
    assert "- Roll back last deploy" in md
    assert "```bash\nfly deploy --image prev\n```" in md
    # Unknown block type → placeholder
    assert "[block:image]" in md


def test_blocks_to_markdown_handles_empty() -> None:
    assert blocks_to_markdown([]) == ""


def test_blocks_to_markdown_numbered_list_restarts() -> None:
    blocks = [
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "first"}]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "second"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "break"}]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "one again"}]}},
    ]
    md = blocks_to_markdown(blocks)
    lines = md.splitlines()
    assert lines[0] == "1. first"
    assert lines[1] == "2. second"
    assert lines[2] == "break"
    assert lines[3] == "1. one again"


def test_blocks_to_markdown_recurses_into_children() -> None:
    """Toggle / column / list-children content must surface, not vanish.

    Previously: nested children were never fetched and rendered as
    `[block:toggle]` placeholders, dropping the entire collapsed body.
    """
    blocks = [
        {
            "type": "toggle",
            "toggle": {"rich_text": [{"plain_text": "How to deploy"}]},
            "_children": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "Run fly deploy"}]},
                },
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"plain_text": "verify health"}]
                    },
                },
            ],
        },
        {
            "type": "column_list",
            "column_list": {},
            "_children": [
                {
                    "type": "column",
                    "column": {},
                    "_children": [
                        {
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"plain_text": "left column"}]
                            },
                        }
                    ],
                },
                {
                    "type": "column",
                    "column": {},
                    "_children": [
                        {
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"plain_text": "right column"}]
                            },
                        }
                    ],
                },
            ],
        },
    ]
    md = blocks_to_markdown(blocks)
    assert "How to deploy" in md
    assert "Run fly deploy" in md
    assert "verify health" in md
    # Container blocks render children at the same depth (no indent).
    assert "left column" in md
    assert "right column" in md
    # Toggle children are indented one level under the toggle line.
    toggle_line = next(line for line in md.splitlines() if "How to deploy" in line)
    body_line = next(line for line in md.splitlines() if "Run fly deploy" in line)
    assert toggle_line.lstrip(" >").startswith("How to deploy")
    assert body_line.startswith("  ")


def test_blocks_to_markdown_child_page_placeholder_no_recursion() -> None:
    """child_page is a separate ingestion root — render as a placeholder
    referencing the title, but do NOT inline its content (would duplicate)."""
    blocks = [
        {
            "type": "child_page",
            "child_page": {"title": "Subpage runbook"},
        }
    ]
    md = blocks_to_markdown(blocks)
    assert "[child_page: Subpage runbook]" in md


def test_extract_mentioned_user_ids_descends_into_children() -> None:
    """Mentions inside toggle/column children must still surface as PERSON
    nodes — otherwise nested @mentions silently vanish from the graph."""
    from kb.handlers.notion import _extract_mentioned_user_ids

    blocks = [
        {
            "type": "toggle",
            "toggle": {"rich_text": [{"plain_text": "Toggle title"}]},
            "_children": [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "mention",
                                "mention": {
                                    "type": "user",
                                    "user": {"id": "user_carol"},
                                },
                                "plain_text": "@Carol",
                            }
                        ]
                    },
                }
            ],
        }
    ]
    assert _extract_mentioned_user_ids(blocks) == ["user_carol"]


# ---------------------------------------------------------------------------
# _fetch_all_blocks recursion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_all_blocks_recurses_into_has_children() -> None:
    """Nested blocks (has_children=true) must be fetched and attached as
    `_children`. The original implementation only fetched direct children,
    which is the bulk of the "missing docs" the user reported."""
    from engine.shared.models import IntegrationToken

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # /v1/blocks/{id}/children
        path = request.url.path
        calls.append(path)
        if path == "/v1/blocks/page_root/children":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "blk_toggle",
                            "type": "toggle",
                            "has_children": True,
                            "toggle": {"rich_text": [{"plain_text": "T"}]},
                        },
                        {
                            "id": "blk_para",
                            "type": "paragraph",
                            "has_children": False,
                            "paragraph": {"rich_text": [{"plain_text": "flat"}]},
                        },
                        # child_page must NOT be recursed (separate ingestion root).
                        {
                            "id": "blk_subpage",
                            "type": "child_page",
                            "has_children": True,
                            "child_page": {"title": "Subpage"},
                        },
                    ],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        if path == "/v1/blocks/blk_toggle/children":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "blk_inner",
                            "type": "paragraph",
                            "has_children": False,
                            "paragraph": {"rich_text": [{"plain_text": "hidden"}]},
                        }
                    ],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        return httpx.Response(404, json={"error": f"unmocked {path}"})

    settings = Settings(environment="local")
    ctx = ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.NOTION, access_token="x"
    )

    blocks = await notion._fetch_all_blocks("page_root", token)

    # Top-level: toggle + paragraph + child_page.
    assert [b["type"] for b in blocks] == ["toggle", "paragraph", "child_page"]
    # Toggle has children attached.
    toggle = blocks[0]
    assert toggle["_children"][0]["type"] == "paragraph"
    assert (
        toggle["_children"][0]["paragraph"]["rich_text"][0]["plain_text"] == "hidden"
    )
    # child_page was NOT recursed into — only toggle's children were fetched.
    assert "/v1/blocks/blk_subpage/children" not in calls
    # Markdown renders the nested content end-to-end.
    md = blocks_to_markdown(blocks)
    assert "hidden" in md
    assert "[child_page: Subpage]" in md


# ---------------------------------------------------------------------------
# _fetch_entity — 2026-03-11 endpoint dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_entity_dispatches_data_source_to_new_endpoint() -> None:
    """In 2026-03-11, `data_source.*` events must hit /v1/data_sources/{id},
    not the legacy /v1/databases/{id}. The schema/properties live on the
    data source record now, not on the parent database."""
    from engine.shared.models import IntegrationToken

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path == "/v1/data_sources/ds_42":
            return httpx.Response(
                200,
                json={
                    "object": "data_source",
                    "id": "ds_42",
                    "properties": {
                        "Name": {"id": "name", "type": "title"},
                        "Status": {"id": "status", "type": "select"},
                    },
                    "parent": {"type": "database_id", "database_id": "db_99"},
                },
            )
        return httpx.Response(404, json={"error": f"unmocked {path}"})

    settings = Settings(environment="local")
    ctx = ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.NOTION, access_token="x"
    )

    entity = await notion._fetch_entity("data_source", "ds_42", token)
    assert entity is not None
    assert entity["object"] == "data_source"
    assert entity["properties"]["Name"]["type"] == "title"
    # Critically: the call goes to /v1/data_sources/{id}, not /v1/databases/{id}.
    assert calls == ["/v1/data_sources/ds_42"]


@pytest.mark.asyncio
async def test_fetch_entity_database_uses_databases_endpoint() -> None:
    """Sanity: database events still hit /v1/databases/{id}; only the
    response shape (now a container with data_sources) changed."""
    from engine.shared.models import IntegrationToken

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path == "/v1/databases/db_99":
            return httpx.Response(
                200,
                json={
                    "object": "database",
                    "id": "db_99",
                    "title": [{"type": "text", "plain_text": "Engineering tasks",
                               "text": {"content": "Engineering tasks"}}],
                    "data_sources": [
                        {"id": "ds_42", "name": "Default"},
                        {"id": "ds_43", "name": "Archive"},
                    ],
                },
            )
        return httpx.Response(404, json={"error": f"unmocked {path}"})

    settings = Settings(environment="local")
    ctx = ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.NOTION, access_token="x"
    )

    entity = await notion._fetch_entity("database", "db_99", token)
    assert entity is not None
    assert calls == ["/v1/databases/db_99"]
    # New 2026-03-11 container shape — data_sources, no top-level properties.
    assert [ds["id"] for ds in entity["data_sources"]] == ["ds_42", "ds_43"]
    assert "properties" not in entity


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def _webhook_event(payload: dict) -> WebhookEvent:
    return WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        source_event_id="page:page_abc123:2026-04-22T12:00:00.000Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/notion/cust-1/2026/04/22/test.json",
        raw_payload=payload,
        headers={},
    )


@pytest.mark.asyncio
async def test_normalize_without_token_empty_body() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    event = _webhook_event(_load("page_properties_updated.json"))
    result = await notion.normalize(event, {})

    assert not result.is_empty
    assert len(result.documents) == 1
    doc = result.documents[0]

    assert doc.doc_id == "notion:page:page_abc123"
    assert doc.doc_type == DocType.NOTION_PAGE
    assert doc.body == ""
    assert doc.metadata["hydrated"] is False


@pytest.mark.asyncio
async def test_normalize_reads_entity_body_markdown_synth_bypass() -> None:
    """Synth corpus bypass: when the webhook payload's entity carries an
    inlined body_markdown (and properties.title), the handler must use them
    instead of the empty hydrated dict. Real Notion webhooks never set
    entity.body_markdown, so this fallback is a no-op on prod traffic.
    """
    payload = _load("page_properties_updated.json")
    # scripts/synth/output/notion.py inlines these fields on entity.
    payload["entity"]["body_markdown"] = "# Synth page\n\nBody content from synth corpus."
    payload["entity"]["properties"] = {
        "title": {
            "type": "title",
            "title": [{"type": "text", "plain_text": "Synth page",
                       "text": {"content": "Synth page"}}],
        },
    }

    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    event = _webhook_event(payload)

    # No hydration available — the synth path doesn't fetch from Notion's API.
    result = await notion.normalize(event, {})

    doc = result.documents[0]
    assert doc.title == "Synth page"
    assert "Body content from synth corpus." in doc.body
    assert doc.body_size_bytes > 0
    assert doc.source_system == SourceSystem.NOTION
    assert doc.author_id == "unknown"  # no hydration → no last_edited_by id

    # Workspace-level ACL always present.
    acl_rows = result.acl_snapshots
    assert len(acl_rows) == 1
    row = acl_rows[0]
    assert row.principal_type == PrincipalType.WORKSPACE
    assert row.principal_id == "ws_TEST"
    assert row.resource_type == "notion.page"
    assert row.resource_id == "page_abc123"
    assert row.permission == Permission.READ
    assert row.metadata["inherits"] is True

    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels
    assert (NodeLabel.PERSON, "unknown") in labels


@pytest.mark.asyncio
async def test_normalize_with_hydrated_content() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    event = _webhook_event(_load("page_properties_updated.json"))
    entity = _load("page_metadata.json")
    blocks = _load("blocks.json")["results"]

    from kb.handlers.notion import _extract_mentioned_user_ids

    hydrated = {
        "entity": entity,
        "body_markdown": blocks_to_markdown(blocks),
        "mentioned_user_ids": _extract_mentioned_user_ids(blocks),
        "permissions": entity["permissions"],
        "resource_type": "page",
    }

    result = await notion.normalize(event, hydrated)
    assert not result.is_empty
    doc = result.documents[0]

    # Body populated.
    assert "Incident runbook" in doc.body
    assert "Check Stripe dashboard" in doc.body
    assert doc.title == "Payments runbook"
    assert doc.author_id == "user_alice"
    assert doc.source_url == "https://www.notion.so/Payments-runbook-abc123"
    assert doc.body_size_bytes > 0
    assert doc.metadata["hydrated"] is True

    # Mentioned user → PERSON node + MENTIONS edge.
    person_ids = {n.canonical_id for n in result.graph_nodes if n.label == NodeLabel.PERSON}
    assert "user_bob" in person_ids
    assert "user_alice" in person_ids

    mentions_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.MENTIONS]
    assert any(e.to_canonical_id == "user_bob" for e in mentions_edges)

    authored_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.AUTHORED]
    assert any(e.from_canonical_id == "user_alice" for e in authored_edges)

    # ACL: workspace fallback + user + group rows.
    rows = result.acl_snapshots
    assert len(rows) == 3
    principal_pairs = {(r.principal_type, r.principal_id) for r in rows}
    assert (PrincipalType.WORKSPACE, "ws_TEST") in principal_pairs
    assert (PrincipalType.USER, "user_alice") in principal_pairs
    assert (PrincipalType.GROUP, "group_eng") in principal_pairs

    # Editor role → WRITE, reader → READ.
    user_row = next(r for r in rows if r.principal_type == PrincipalType.USER)
    assert user_row.permission == Permission.WRITE
    group_row = next(r for r in rows if r.principal_type == PrincipalType.GROUP)
    assert group_row.permission == Permission.READ

    # Parent chain captured in metadata so Phase 1 enforcement can walk it.
    assert all(r.metadata.get("inherits") is True for r in rows)
    assert all(r.metadata.get("parent_id") == "page_parent_999" for r in rows)


@pytest.mark.asyncio
async def test_normalize_propagates_in_trash_field() -> None:
    """2026-03-11 renamed `archived` → `in_trash`. Metadata key follows
    the new spec; legacy `archived` payloads still flow through via the
    fallback so cached / replayed envelopes don't lose the bit."""
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    event = _webhook_event(_load("page_properties_updated.json"))

    # Spec-compliant 2026-03-11 entity uses in_trash.
    entity_new = {"in_trash": True, "properties": {}}
    result_new = await notion.normalize(event, {"entity": entity_new})
    assert result_new.documents[0].metadata["in_trash"] is True

    # Legacy entity uses archived. Fallback must catch it.
    entity_legacy = {"archived": True, "properties": {}}
    result_legacy = await notion.normalize(event, {"entity": entity_legacy})
    assert result_legacy.documents[0].metadata["in_trash"] is True

    # Default (neither field set) is False.
    result_default = await notion.normalize(event, {"entity": {"properties": {}}})
    assert result_default.documents[0].metadata["in_trash"] is False


def test_database_schema_summary_renders_2026_container_shape() -> None:
    """`GET /v1/databases/{id}` in 2026-03-11 returns a container with
    `data_sources: [...]` (no top-level properties). The summary must
    include the data_sources list rather than fall back to an empty body."""
    from kb.handlers.notion import _database_schema_summary

    container_entity = {
        "object": "database",
        "id": "db_99",
        "description": [{"plain_text": "Engineering tickets"}],
        "data_sources": [
            {"id": "ds_42", "name": "Default"},
            {"id": "ds_43", "name": "Archive"},
        ],
    }
    summary = _database_schema_summary(container_entity)
    assert "Engineering tickets" in summary
    assert "Data sources:" in summary
    assert "- Default" in summary
    assert "- Archive" in summary


def test_database_schema_summary_renders_data_source_properties() -> None:
    """A data_source record (or a legacy database) carries `properties`
    directly — summarize them as before."""
    from kb.handlers.notion import _database_schema_summary

    ds_entity = {
        "object": "data_source",
        "id": "ds_42",
        "properties": {
            "Name": {"id": "name", "type": "title"},
            "Status": {"id": "status", "type": "select"},
        },
    }
    summary = _database_schema_summary(ds_entity)
    assert "Properties:" in summary
    assert "- Name (title)" in summary
    assert "- Status (select)" in summary


# ---------------------------------------------------------------------------
# signature verification
# ---------------------------------------------------------------------------


def test_verify_signature_dev_bypass() -> None:
    ctx = _make_ctx(env="local")
    notion = build_connector(SourceSystem.NOTION, ctx)
    assert notion.verify_signature({}, b"{}") is True


def test_verify_signature_prod_rejects_unsigned_unknown_caller() -> None:
    ctx = _make_ctx(env="main")
    notion = build_connector(SourceSystem.NOTION, ctx)
    assert notion.verify_signature({}, b"{}") is False


def test_verify_signature_valid_hmac() -> None:
    """Signature key is the subscription's verification_token, not the OAuth
    client secret. Notion's webhook docs (and the Sept 2025 upgrade) are
    explicit on this — the verification_token returned during the one-time
    handshake doubles as the HMAC signing secret."""
    import hashlib
    import hmac as hmac_mod

    from pydantic import SecretStr

    secret = "notion-verification-token"
    body = b'{"hello":"notion"}'
    digest = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()

    settings = Settings(
        environment="main",
        notion_webhook_verification_token=SecretStr(secret),
    )
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    notion = build_connector(SourceSystem.NOTION, ctx)

    assert notion.verify_signature({"X-Notion-Signature": digest}, body) is True
    assert notion.verify_signature({"X-Notion-Signature": digest}, body + b"x") is False
    assert (
        notion.verify_signature({"X-Notion-Signature": f"sha256={digest}"}, body) is True
    )


def test_verify_signature_prod_rejects_when_token_unset() -> None:
    """A signed payload with no token configured should still 401 — defense
    against a half-deployed env where the secret hasn't been set yet."""
    import hashlib
    import hmac as hmac_mod

    body = b'{"hello":"notion"}'
    digest = hmac_mod.new(b"any-secret", body, hashlib.sha256).hexdigest()

    settings = Settings(environment="main")  # no token
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    notion = build_connector(SourceSystem.NOTION, ctx)

    assert notion.verify_signature({"X-Notion-Signature": digest}, body) is False


def test_verify_signature_ignores_oauth_client_secret() -> None:
    """Pre-fix bug: signature was being checked against `notion_client_secret`.
    Make sure signing with the OAuth secret is now rejected when the
    verification token is the configured signing key."""
    import hashlib
    import hmac as hmac_mod

    from pydantic import SecretStr

    body = b'{"hello":"notion"}'
    oauth_secret = "oauth-client-secret"
    real_token = "verification-token"

    # Sign with the OAuth secret, not the verification token.
    bad_digest = hmac_mod.new(oauth_secret.encode(), body, hashlib.sha256).hexdigest()

    settings = Settings(
        environment="main",
        notion_client_secret=SecretStr(oauth_secret),
        notion_webhook_verification_token=SecretStr(real_token),
    )
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    notion = build_connector(SourceSystem.NOTION, ctx)

    assert notion.verify_signature({"X-Notion-Signature": bad_digest}, body) is False


# ---------------------------------------------------------------------------
# OAuth code exchange — called by /api/oauth/notion/exchange (admin_routes.py)
# after prbe-backend's gateway has verified the signed state and resolved the
# customer. Public install/callback live in prbe-backend, not here.
# ---------------------------------------------------------------------------


def _oauth_ctx_with_secret(
    *,
    handler,
    client_id: str | None = "ntn_test_client",
    client_secret: str | None = "ntn_test_secret",
) -> ConnectorContext:
    from pydantic import SecretStr

    settings = Settings(
        environment="local",
        notion_client_id=client_id,
        notion_client_secret=SecretStr(client_secret) if client_secret else None,
    )
    return ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_exchange_oauth_code_happy() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/oauth/token"
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_access_xyz",
                "token_type": "bearer",
                "refresh_token": "ntn_refresh_abc",
                "bot_id": "bot_42",
                "workspace_id": "ws_alpha",
                "workspace_name": "Acme Eng",
                "workspace_icon": "https://example.com/icon.png",
                "owner": {"user": {"id": "user_alice"}},
                "duplicated_template_id": None,
                "request_id": "req_1",
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    token = await notion.exchange_oauth_code(
        code="ntn-code-123",
        redirect_uri="https://example.com/oauth/notion/callback",
    )

    # httpx's auth=(cid, secret) tuple yields HTTP Basic auth.
    assert seen["auth"].startswith("Basic ")
    assert seen["body"] == {
        "grant_type": "authorization_code",
        "code": "ntn-code-123",
        "redirect_uri": "https://example.com/oauth/notion/callback",
    }

    assert token.access_token == "ntn_access_xyz"
    assert token.refresh_token == "ntn_refresh_abc"
    assert token.source_system == SourceSystem.NOTION
    # connector returns customer_id="" — the caller fills it in.
    assert token.customer_id == ""
    # Notion uses capability checkboxes, not OAuth scope strings.
    assert token.scope is None

    # install_metadata carries workspace info into identify_workspaces.
    assert token.install_metadata == {
        "workspace_id": "ws_alpha",
        "workspace_name": "Acme Eng",
        "workspace_icon": "https://example.com/icon.png",
        "bot_id": "bot_42",
        "owner": {"user": {"id": "user_alice"}},
    }


@pytest.mark.asyncio
async def test_exchange_oauth_code_4xx_raises_permanent() -> None:
    from engine.shared.exceptions import PermanentSourceError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(PermanentSourceError):
        await notion.exchange_oauth_code(
            code="bad", redirect_uri="https://x/cb"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_missing_code_raises() -> None:
    from engine.shared.exceptions import InvalidWebhookPayload

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("HTTP should not be called when code is None")

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(InvalidWebhookPayload):
        await notion.exchange_oauth_code(
            code=None, redirect_uri="https://x/cb"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_missing_secret_raises() -> None:
    from engine.shared.exceptions import MissingSecret

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("HTTP should not be called when secret is missing")

    ctx = _oauth_ctx_with_secret(handler=handler, client_secret=None)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(MissingSecret):
        await notion.exchange_oauth_code(
            code="x", redirect_uri="https://x/cb"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_no_refresh_token() -> None:
    """Notion's docs say refresh_token is nullable — connector must not
    crash when it's absent (older public integrations)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_access_xyz",
                "token_type": "bearer",
                "bot_id": "bot_42",
                "workspace_id": "ws_alpha",
                "workspace_name": "Acme Eng",
                "owner": {"user": {"id": "user_alice"}},
                # no refresh_token / workspace_icon
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = await notion.exchange_oauth_code(
        code="x", redirect_uri="https://x/cb"
    )
    assert token.access_token == "ntn_access_xyz"
    assert token.refresh_token is None
    assert token.install_metadata["workspace_id"] == "ws_alpha"
    assert token.install_metadata["workspace_icon"] is None


@pytest.mark.asyncio
async def test_exchange_oauth_code_malformed_200_raises_permanent() -> None:
    """Defensive: a Notion 200 that's missing the documented non-null
    `access_token` / `workspace_id` fields shouldn't be an unhandled
    KeyError → 500. Map to PermanentSourceError → 502."""
    from engine.shared.exceptions import PermanentSourceError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token_type": "bearer"})

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(PermanentSourceError):
        await notion.exchange_oauth_code(
            code="x", redirect_uri="https://x/cb"
        )


# ---------------------------------------------------------------------------
# identify_workspaces — reads install_metadata, no network
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_workspaces_from_install_metadata() -> None:
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(
            "identify_workspaces must NOT make HTTP calls — workspace info "
            "should come from token.install_metadata, populated by "
            "exchange_oauth_code"
        )

    settings = Settings(environment="local")
    ctx = ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    notion = build_connector(SourceSystem.NOTION, ctx)

    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        access_token="ntn_access_xyz",
        install_metadata={
            "workspace_id": "ws_alpha",
            "workspace_name": "Acme Eng",
            "workspace_icon": "https://example.com/icon.png",
            "bot_id": "bot_42",
            "owner": {"user": {"id": "user_alice"}},
        },
    )

    refs = await notion.identify_workspaces(token)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.external_id == "ws_alpha"
    assert ref.external_name == "Acme Eng"
    assert ref.metadata == {
        "bot_id": "bot_42",
        "owner_user_id": "user_alice",
        "workspace_icon": "https://example.com/icon.png",
    }


@pytest.mark.asyncio
async def test_identify_workspaces_returns_empty_when_metadata_missing() -> None:
    """A token loaded from DB doesn't have install_metadata (it's transient).
    Connector must return [] instead of crashing — the exchange caller
    already swallows that case (logs a warning, no mapping recorded)."""
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("identify_workspaces must not hit the network on missing metadata")

    settings = Settings(environment="local")
    ctx = ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    notion = build_connector(SourceSystem.NOTION, ctx)

    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        access_token="ntn_access_xyz",
        install_metadata=None,
    )
    assert await notion.identify_workspaces(token) == []


# ---------------------------------------------------------------------------
# exchange_oauth_code now persists expires_at when token rotation is enabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_oauth_code_persists_expires_at_when_rotated() -> None:
    """Notion's rotated-token integrations include `expires_in` (seconds)
    in the exchange response. The handler must surface it as a UTC datetime
    on `IntegrationToken.expires_at` so the refresh cron can pick the row
    up before the access_token dies."""
    from datetime import UTC, datetime

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_access_xyz",
                "refresh_token": "ntn_refresh_abc",
                "expires_in": 3600,
                "workspace_id": "ws_alpha",
                "bot_id": "bot_42",
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    before = datetime.now(UTC)
    token = await notion.exchange_oauth_code(
        code="x", redirect_uri="https://example.com/cb"
    )
    assert token.expires_at is not None
    delta = (token.expires_at - before).total_seconds()
    assert 3590 <= delta <= 3610


@pytest.mark.asyncio
async def test_exchange_oauth_code_legacy_long_lived_no_expires_at() -> None:
    """Legacy long-lived integrations omit expires_in; expires_at stays None
    so the refresh cron doesn't pick the row up."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_access_xyz",
                "workspace_id": "ws_alpha",
                "bot_id": "bot_42",
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = await notion.exchange_oauth_code(
        code="x", redirect_uri="https://example.com/cb"
    )
    assert token.expires_at is None


# ---------------------------------------------------------------------------
# exchange_refresh_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_refresh_token_no_refresh_token_raises_permanent() -> None:
    """Legacy tokens minted before rotation have no refresh_token; caller
    flips auth_failed and prompts reconnect."""
    from engine.shared.exceptions import PermanentSourceError
    from engine.shared.models import IntegrationToken

    ctx = _oauth_ctx_with_secret(handler=lambda req: httpx.Response(200, json={}))
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        access_token="ntn_legacy",
        refresh_token=None,
    )
    with pytest.raises(PermanentSourceError, match="without a stored refresh_token"):
        await notion.exchange_refresh_token(token)


@pytest.mark.asyncio
async def test_exchange_refresh_token_success_rotates() -> None:
    from engine.shared.models import IntegrationToken

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_new",
                "refresh_token": "ntn_new_refresh",
                "expires_in": 3600,
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        access_token="ntn_old",
        refresh_token="ntn_old_refresh",
    )
    new = await notion.exchange_refresh_token(old)

    assert seen["body"] == {
        "grant_type": "refresh_token",
        "refresh_token": "ntn_old_refresh",
    }
    assert new.access_token == "ntn_new"
    assert new.refresh_token == "ntn_new_refresh"
    assert new.customer_id == "cust-1"
    assert new.expires_at is not None


@pytest.mark.asyncio
async def test_exchange_refresh_token_4xx_raises_permanent() -> None:
    from engine.shared.exceptions import PermanentSourceError
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        access_token="x",
        refresh_token="rotten",
    )
    with pytest.raises(PermanentSourceError):
        await notion.exchange_refresh_token(old)


@pytest.mark.asyncio
async def test_exchange_refresh_token_5xx_raises_transient() -> None:
    from engine.shared.exceptions import TransientSourceError
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        access_token="x",
        refresh_token="r",
    )
    with pytest.raises(TransientSourceError):
        await notion.exchange_refresh_token(old)


# ---------------------------------------------------------------------------
# verify_token_health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_token_health_healthy_returns_true() -> None:
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/users/me"
        return httpx.Response(200, json={"id": "bot_42", "type": "bot"})

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.NOTION, access_token="x"
    )
    assert await notion.verify_token_health(token) is True


@pytest.mark.asyncio
async def test_verify_token_health_401_returns_false() -> None:
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"code": "unauthorized", "message": "Invalid token"}
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.NOTION, access_token="x"
    )
    assert await notion.verify_token_health(token) is False


@pytest.mark.asyncio
async def test_verify_token_health_5xx_raises_transient() -> None:
    from engine.shared.exceptions import TransientSourceError
    from engine.shared.models import IntegrationToken

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.NOTION, access_token="x"
    )
    with pytest.raises(TransientSourceError):
        await notion.verify_token_health(token)

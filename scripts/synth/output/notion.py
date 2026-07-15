"""NotionWrapper — serialize a SynthDoc to a Notion page.content_updated webhook envelope.

The output must round-trip through NotionConnector._parse_notion_webhook.
_is_notion_webhook() checks: isinstance(entity, dict) and isinstance(type, str).
_parse_notion_webhook reads: type, entity.type, entity.id, workspace_id; it
falls back to top-level `timestamp` for the source-event-id derivation when
data lacks last_edited_time (real Notion never sets it on data).

Picked `page.content_updated` over the more event-specific
`page.properties_updated` because synth output is whole-page replay — the
content-updated event is the closest single-event spec for "this page's
body changed".

Synth-only inlining (the notion-handler bypass):
    Real Notion webhooks notify-only — content lives behind a Notion API call
    that the prod handler makes via fetch_supplementary(integration_token).
    Synth has no live OAuth token, so we inline the would-be hydrated content
    on `entity` itself: `entity.properties` (title) and `entity.body_markdown`
    (pre-rendered block content). The prod handler reads these as a fallback
    when its API fetch returns nothing; on real webhook traffic neither field
    is present and behavior is unchanged. See services/ingestion/handlers/
    notion.py::normalize for the reader-side comment.

v1 is minimal: title property + plain-text paragraph blocks. Plan 3 can extend
to richer block shapes when LLM-generated content warrants it.
"""

from __future__ import annotations

from datetime import datetime

import orjson

# blocks_to_markdown is the same renderer the prod handler uses post-fetch,
# so synth's inlined body_markdown matches what real Notion ingestion would
# have produced for the same block tree.
from kb.handlers.notion import blocks_to_markdown
from scripts.synth.output.base import SynthDoc

_SYNTH_WORKSPACE_ID = "ws-synth"
_SYNTH_WORKSPACE_NAME = "Synth"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _user_slug(doc: SynthDoc) -> str:
    if not doc.personas:
        return "user-synth-unknown"
    return doc.personas[0].replace(":", "-").replace("@", "")


def _title_from_doc(doc: SynthDoc) -> str:
    """Extract first line of text as the page title."""
    first_line = doc.text.splitlines()[0].strip() if doc.text else ""
    title = first_line.lstrip("#").strip()
    return title[:200] if title else f"Synth page {doc.id}"


def _blocks_from_text(text: str) -> list[dict]:
    """Convert plain text to minimal Notion block list (paragraph per line).

    Heading lines (## prefix) become heading_2 blocks.
    Other lines become paragraph blocks.
    Empty lines are skipped.
    """
    blocks: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            content = stripped[3:].strip()
            btype = "heading_2"
        elif stripped.startswith("# "):
            content = stripped[2:].strip()
            btype = "heading_1"
        else:
            content = stripped
            btype = "paragraph"
        blocks.append({
            "type": btype,
            "id": f"block-{len(blocks)}",
            btype: {
                "rich_text": [
                    {
                        "type": "text",
                        "plain_text": content,
                        "text": {"content": content},
                    }
                ]
            },
        })
    return blocks


def wrap(doc: SynthDoc) -> bytes:
    """Produce a Notion page.content_updated webhook envelope as JSON bytes.

    Inlines properties + pre-rendered body_markdown on `entity` so the prod
    handler can ingest synth content without a live OAuth fetch (see module
    docstring for the bypass contract).
    """
    page_id = doc.page_id or f"page-{doc.source_event_id}"
    iso_ts = _iso(doc.occurred_at)
    title = _title_from_doc(doc)
    blocks = _blocks_from_text(doc.text)

    title_property = {
        "type": "title",
        "title": [
            {
                "type": "text",
                "plain_text": title,
                "text": {"content": title},
            }
        ],
    }

    payload = {
        "id": doc.source_event_id,
        "type": "page.content_updated",
        "timestamp": iso_ts,
        "workspace_id": _SYNTH_WORKSPACE_ID,
        "workspace_name": _SYNTH_WORKSPACE_NAME,
        "entity": {
            "type": "page",
            "id": page_id,
            # Synth-only: inlined hydration (see module docstring).
            "properties": {"title": title_property},
            "body_markdown": blocks_to_markdown(blocks),
        },
        "data": {
            "last_edited_time": iso_ts,
            "last_edited_by": {"id": _user_slug(doc)},
            "updated_properties": ["title"],
            # data.* is left for legacy compatibility with any consumer that
            # reads webhook bodies before the entity-side inlining landed.
            "properties": {"title": title_property},
            "blocks": blocks,
        },
    }

    return orjson.dumps(payload)

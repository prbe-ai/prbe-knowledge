"""Per-customer postmortem template resolution.

The postmortem agent calls ``get_effective_template(customer_id)`` to
discover the template body it should render the draft against.
Resolution order:

1. ``customer_postmortem_templates`` row in ``inline`` mode → use
   ``body_markdown`` directly (``source='inline_override'``).
2. ``customer_postmortem_templates`` row in ``doc_ref`` mode → fetch
   the referenced document's body (reassembled from ``chunks.content``
   in ``chunk_index`` order, filtered to live + ``visibility='approved'``).
   If the referenced doc isn't readable, log a warning and fall through.
   (``source='doc_ref_override'``).
3. Otherwise → ``shared.templates.postmortem.DEFAULT_POSTMORTEM_TEMPLATE``
   (``source='default'``).

This module is intentionally independent of ``wiki_review_state`` — the
two concerns share only the schemas package.

RLS: every read goes through ``with_tenant(customer_id)``. The
``documents`` body fetch piggybacks on the same connection so the
tenant GUC is set when scanning chunks.

Note on schema: ``prbe-knowledge.documents`` has no ``body`` column —
document bodies live in ``chunks.content`` (one row per chunk,
re-joined at read time). This mirrors the ``/sources/{doc_id}``
endpoint's reassembly path in ``services/retrieval/main.py``.
"""
from __future__ import annotations

from typing import Any

from shared.db import with_tenant
from shared.logging import get_logger
from shared.schemas.postmortem_template import (
    TemplateEffectiveResponse,
    TemplateRow,
    TemplateUpsertRequest,
)
from shared.templates.postmortem import DEFAULT_POSTMORTEM_TEMPLATE

log = get_logger(__name__)


def _row_to_template(row: Any) -> TemplateRow:
    return TemplateRow(
        customer_id=row["customer_id"],
        mode=row["mode"],
        body_markdown=row["body_markdown"],
        ref_doc_id=row["ref_doc_id"],
        updated_at=row["updated_at"],
    )


async def get_override(customer_id: str) -> TemplateRow | None:
    """Return the customer's stored template override row, or None."""
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT customer_id, mode, body_markdown, ref_doc_id, updated_at "
            "FROM customer_postmortem_templates "
            "WHERE customer_id = $1",
            customer_id,
        )
    return _row_to_template(row) if row else None


async def upsert_override(req: TemplateUpsertRequest) -> TemplateRow:
    """Insert/update the customer's template override.

    For ``mode='doc_ref'``, validates that the referenced doc is
    readable by this customer (exists + ``visibility='approved'``)
    BEFORE writing the row. Without this check a misconfigured override
    would silently fall back to the default template at render time —
    surfacing the error at upsert keeps the failure mode loud and
    immediate. Raises ``ValueError`` on a missing/non-approved ref.
    """
    async with with_tenant(req.customer_id) as conn:
        if req.mode == "doc_ref":
            assert req.ref_doc_id is not None  # enforced by pydantic validator
            ref_exists = await conn.fetchval(
                "SELECT 1 FROM documents "
                "WHERE customer_id = $1 AND doc_id = $2 "
                "AND visibility = 'approved' "
                "LIMIT 1",
                req.customer_id, req.ref_doc_id,
            )
            if ref_exists is None:
                raise ValueError(
                    f"ref_doc_id '{req.ref_doc_id}' not readable for customer"
                )

        row = await conn.fetchrow(
            """
            INSERT INTO customer_postmortem_templates
                (customer_id, mode, body_markdown, ref_doc_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (customer_id) DO UPDATE
              SET mode = EXCLUDED.mode,
                  body_markdown = EXCLUDED.body_markdown,
                  ref_doc_id = EXCLUDED.ref_doc_id,
                  updated_at = now()
            RETURNING customer_id, mode, body_markdown, ref_doc_id, updated_at;
            """,
            req.customer_id, req.mode, req.body_markdown, req.ref_doc_id,
        )
    assert row is not None  # INSERT … RETURNING always returns a row
    return _row_to_template(row)


async def _fetch_doc_body(
    conn: Any, customer_id: str, doc_id: str,
) -> str | None:
    """Reassemble a doc's body from its live, approved content chunks.

    Mirrors the ``/sources/{doc_id}`` path in ``services/retrieval/main.py``:
    1. Look up the live document version.
    2. Pull content chunks for that version (``kind='content'``,
       ``valid_to IS NULL``, version between first_seen and last_seen).
    3. Join with the same separator (``"\n\n"``).

    The visibility gate is on ``documents``: the override should only
    resolve against approved wiki docs (draft revisions in flight
    aren't a stable template target).

    Returns ``None`` if the doc isn't found or is missing content
    chunks (treat as unresolvable and fall through to the default).
    """
    doc = await conn.fetchrow(
        """
        SELECT version FROM documents
        WHERE customer_id = $1 AND doc_id = $2
          AND valid_to IS NULL
          AND visibility = 'approved'
        ORDER BY version DESC
        LIMIT 1
        """,
        customer_id, doc_id,
    )
    if doc is None:
        return None
    chunk_rows = await conn.fetch(
        """
        SELECT content FROM chunks
        WHERE customer_id = $1
          AND doc_id = $2
          AND valid_to IS NULL
          AND kind = 'content'
          AND $3 BETWEEN first_seen_version AND last_seen_version
        ORDER BY chunk_index
        """,
        customer_id, doc_id, doc["version"],
    )
    if not chunk_rows:
        return None
    body = "\n\n".join(r["content"] for r in chunk_rows)
    return body or None


async def get_effective_template(
    customer_id: str,
) -> TemplateEffectiveResponse:
    """Resolve the template the postmortem agent should render against.

    Falls through to the bundled default when no override is set, when
    an inline override's body is unexpectedly empty (defensive — the
    table check constraint should prevent this), or when a doc_ref
    override points at a doc that isn't readable. The last case logs
    a warning so the override misconfiguration surfaces in ops.
    """
    async with with_tenant(customer_id) as conn:
        override_row = await conn.fetchrow(
            "SELECT customer_id, mode, body_markdown, ref_doc_id, updated_at "
            "FROM customer_postmortem_templates "
            "WHERE customer_id = $1",
            customer_id,
        )
        if override_row is None:
            return TemplateEffectiveResponse(
                body_markdown=DEFAULT_POSTMORTEM_TEMPLATE,
                source="default",
            )

        mode = override_row["mode"]
        if mode == "inline":
            body = override_row["body_markdown"]
            if body:
                return TemplateEffectiveResponse(
                    body_markdown=body,
                    source="inline_override",
                )
            # Defensive: the DB check constraint forbids inline + null body,
            # but if a future migration relaxes that, fall through cleanly
            # rather than crashing.
            log.warning(
                "postmortem.template_inline_empty",
                customer_id=customer_id,
            )
            return TemplateEffectiveResponse(
                body_markdown=DEFAULT_POSTMORTEM_TEMPLATE,
                source="default",
            )

        if mode == "doc_ref":
            ref_doc_id = override_row["ref_doc_id"]
            body = await _fetch_doc_body(conn, customer_id, ref_doc_id)
            if body:
                return TemplateEffectiveResponse(
                    body_markdown=body,
                    source="doc_ref_override",
                    resolved_ref_doc_id=ref_doc_id,
                )
            log.warning(
                "postmortem.template_ref_unresolved",
                customer_id=customer_id,
                ref_doc_id=ref_doc_id,
            )
            return TemplateEffectiveResponse(
                body_markdown=DEFAULT_POSTMORTEM_TEMPLATE,
                source="default",
            )

        # Unknown mode (shouldn't happen — check constraint covers it).
        log.warning(
            "postmortem.template_unknown_mode",
            customer_id=customer_id,
            mode=mode,
        )
        return TemplateEffectiveResponse(
            body_markdown=DEFAULT_POSTMORTEM_TEMPLATE,
            source="default",
        )

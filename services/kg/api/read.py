"""Read endpoints for the debugging knowledge graph.

Two endpoints, both tenant-scoped via RLS:

  * ``GET /classes/{class_id}`` — single-class envelope (frontmatter + body)
    deserialized through the ``BugClass`` Pydantic model. 404 when the row
    doesn't exist *for the authenticated tenant* (cross-tenant lookups are
    silently filtered by the RLS policy on ``app.current_customer_id`` and
    surface as 404, not 200). See spec §5.2 for the envelope shape and
    §12.3 for the tenant-isolation contract.
  * ``GET /classes`` — list view returning ``{id, description}`` only. The
    body is excluded by design: it's opaque markdown that can be many KB
    per class, and a list view doesn't need it. Ordered by ``class_id``
    so the response is deterministic for the dashboard staff UI.

Auth: handlers depend on ``authenticate_query`` from
``services/retrieval/auth.py``, which already implements both accepted
auth shapes (``Authorization: Bearer <key>`` and
``X-Internal-Knowledge-Key + X-Prbe-Customer``). Reusing it directly
avoids duplicating the secret-comparison and customer-lookup logic.

DB: handlers acquire a connection via ``shared.db.with_tenant(customer_id)``,
which sets the ``app.current_customer_id`` GUC inside a transaction so
the RLS policy on ``kg_classes`` filters by the authenticated tenant.
The ``customer_id`` column is also included as defense-in-depth in
``WHERE`` clauses — RLS is the source of truth, but the explicit
predicate makes the query self-documenting.

This module deliberately stays read-only. Writes (PUT) land in a sibling
module in a later task.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.kg.schema import BugClass
from services.retrieval.auth import authenticate_query
from shared.db import with_tenant

router = APIRouter()


class ClassListItem(BaseModel):
    """A single row in the list-classes response.

    ``description`` is sourced from ``frontmatter->>'description'``, which
    the schema enforces as a non-empty string at write-time. Defaults to
    an empty string here only as a safety net for ill-formed legacy rows.
    """

    id: str
    description: str


class ClassListResponse(BaseModel):
    """Envelope for ``GET /classes``. Wrapping the list in ``{items: [...]}``
    leaves room for future pagination metadata without a breaking change."""

    items: list[ClassListItem]


def _decode_jsonb(value: Any) -> Any:
    """Normalize asyncpg's JSONB return value to a Python object.

    asyncpg returns JSONB as either ``str``/``bytes`` or ``dict`` depending
    on whether a JSONB codec is registered upstream — we've observed both
    in this repo (``shared/tokens.py:_load_jsonb`` handles the str case;
    the kg path here has seen dict). Handle both so this module doesn't
    care which path the connection took.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return value


@router.get("/classes/{class_id}", response_model=BugClass)
async def get_class(
    class_id: str,
    customer_id: str = Depends(authenticate_query),
) -> BugClass:
    """Return one class as a ``BugClass`` envelope, or 404."""
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT frontmatter, body
              FROM kg_classes
             WHERE customer_id = $1
               AND class_id    = $2
            """,
            customer_id,
            class_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="class not found")
    return BugClass.model_validate(
        {
            "frontmatter": _decode_jsonb(row["frontmatter"]),
            "body": row["body"],
        }
    )


@router.get("/classes", response_model=ClassListResponse)
async def list_classes(
    customer_id: str = Depends(authenticate_query),
) -> ClassListResponse:
    """Return all classes for the authenticated tenant — id + description only.

    No body in the response: see module docstring for the rationale.
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT class_id,
                   frontmatter->>'description' AS description
              FROM kg_classes
             WHERE customer_id = $1
             ORDER BY class_id
            """,
            customer_id,
        )
    return ClassListResponse(
        items=[
            ClassListItem(id=r["class_id"], description=r["description"] or "")
            for r in rows
        ]
    )

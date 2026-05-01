"""Write endpoint for the debugging knowledge graph.

One handler — ``PUT /classes/{class_id}`` — does a full upsert of a
``BugClass`` envelope, gated by:

  1. The same ``authenticate_query`` dep the read API uses, returning the
     authenticated ``customer_id``.
  2. A per-tenant Postgres advisory lock (``tenant_xact_lock``), held for
     the duration of the transaction. This is the single-writer
     enforcement called out in spec §7.2 — concurrent PUTs against the
     same tenant serialize on this lock; nothing else is needed.
  3. ``kg_check`` link validation against the live class universe for the
     authenticated tenant. Frontmatter ``related`` ids and body
     ``[[wiki-link]]`` targets must all resolve (or be the class's own
     id, since a class can self-reference).
  4. A path-id ↔ payload-id consistency check that fails with 400 *before*
     acquiring the lock — there's no point holding a tenant-wide lock for
     a request that's broken on the way in.

Status-code semantics:

  * 201 Created if no row existed for ``(customer_id, class_id)`` before
    this transaction; 200 OK on update. The existence check runs inside
    the same transaction, after the advisory lock is held — doing it
    outside the lock would re-introduce a TOCTOU race that the lock
    exists to prevent.
  * 422 if ``kg_check`` raises ``KgCheckError``; the exception's message
    (which includes the unresolved class ids) becomes the response detail.
  * 400 on path/payload id mismatch.
  * 401 on missing/invalid auth (raised by ``authenticate_query``).

DB shape: the upsert uses ``ON CONFLICT (customer_id, class_id) DO UPDATE``
and explicitly bumps ``updated_at = NOW()`` — the column has
``DEFAULT NOW()`` but defaults only fire on INSERT, so without the
explicit set the conflict path silently keeps the original mtime and
downstream maintenance logic gets misled.

Status code mechanism: handler returns ``fastapi.responses.JSONResponse``
directly. ``response_model`` is intentionally not set — Starlette's
``(payload, status_code)`` tuple doesn't compose with ``response_model``
cleanly, and JSONResponse keeps the read/write paths' return shapes
visible at the call site without surprise.

See spec §5.2 (envelope shape), §7.2 (single-writer + advisory lock),
§12.3 (tenant isolation contract).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from services.kg.advisory_lock import tenant_xact_lock
from services.kg.kg_check import KgCheckError, check_class
from services.kg.schema import BugClass
from services.retrieval.auth import authenticate_query
from shared.db import with_tenant

router = APIRouter()


@router.put("/classes/{class_id}")
async def put_class(
    class_id: str,
    payload: BugClass,
    customer_id: str = Depends(authenticate_query),
) -> JSONResponse:
    """Upsert one class for the authenticated tenant.

    See module docstring for the full contract. The ordering inside the
    transaction is deliberate:

        with_tenant(customer_id):           # opens tx + sets RLS GUC
            tenant_xact_lock(customer_id):  # serialize writes per tenant
                load class universe         # for kg_check
                check_class(payload, ...)   # 422 on broken refs
                check existence             # determines 200 vs 201
                upsert                      # UPDATE bumps updated_at
    """
    # Cheap pre-check: bail before touching the DB if the path and payload
    # disagree. No point taking a tenant-wide lock for a malformed request.
    if payload.frontmatter.id != class_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"path id {class_id!r} != payload id {payload.frontmatter.id!r}"
            ),
        )

    async with (
        with_tenant(customer_id) as conn,
        tenant_xact_lock(conn, customer_id=customer_id),
    ):
        # Universe = every existing class_id for this tenant, plus the
        # path id of the class being upserted. Including the path id
        # lets a brand-new class self-reference (e.g. as a regression
        # marker) without hitting kg_check on the very first write.
        universe_rows = await conn.fetch(
            "SELECT class_id FROM kg_classes WHERE customer_id = $1",
            customer_id,
        )
        universe: set[str] = {r["class_id"] for r in universe_rows} | {class_id}

        try:
            check_class(payload, universe=universe)
        except KgCheckError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        existing = await conn.fetchrow(
            """
            SELECT 1
              FROM kg_classes
             WHERE customer_id = $1
               AND class_id    = $2
            """,
            customer_id,
            class_id,
        )
        status_code = 200 if existing is not None else 201

        await conn.execute(
            """
            INSERT INTO kg_classes (customer_id, class_id, frontmatter, body)
            VALUES ($1, $2, $3::jsonb, $4)
            ON CONFLICT (customer_id, class_id) DO UPDATE
              SET frontmatter = EXCLUDED.frontmatter,
                  body        = EXCLUDED.body,
                  -- DEFAULT NOW() only fires on INSERT; the conflict
                  -- branch must bump updated_at explicitly.
                  updated_at  = NOW()
            """,
            customer_id,
            class_id,
            payload.frontmatter.model_dump_json(),
            payload.body,
        )

    return JSONResponse(content={"status": "ok"}, status_code=status_code)

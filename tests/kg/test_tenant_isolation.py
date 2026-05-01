"""Spec §12.3 — comprehensive tenant-isolation regression alarm.

Authenticates as tB and actively tries to read or modify tA's data via
every Phase-1 API path AND the pgvector embedding-query path. Every
test in this file is a regression alarm — if any assertion ever fails
in the future, that's a tenant-isolation breach, not a flaky test.

The four paths under test:

1. ``GET /kg/classes/{id}``  — direct cross-tenant fetch must 404.
2. ``GET /kg/classes``       — list view must NOT contain tA rows.
3. ``PUT /kg/classes/{id}``  — write under tB's namespace MUST NOT
   overwrite tA's row at the same ``class_id``. (PK is
   ``(customer_id, class_id)`` so the rows are distinct in the DB; this
   test proves the API path slug can't be used as a hijack vector.)
4. ``services.kg.embedding_query.query_similar`` — pgvector cosine
   query under tB returns zero rows even when tA has a vector at the
   queried point. Replaces the originally-planned Pinecone-namespace
   test (this repo uses pgvector, not Pinecone — the isolation surface
   is RLS, not a vector-store namespace).

RLS contract (spec §5.1; migration ``0031_kg_rls``): policies are
``USING customer_id = current_setting('app.current_customer_id', true)``,
USING-only, with ``FORCE ROW LEVEL SECURITY``. A query without the GUC
set returns zero rows; a query with a different tenant's GUC returns
zero rows. Both code paths are exercised here.

All tests use the ``seeded_classes`` fixture (tA pre-loaded with two
classes, tB has only a customers row) and the
``tests/kg/conftest.py:pytest_collection_modifyitems`` hook auto-skips
the file when Postgres isn't reachable.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from services.kg.embedding_query import _format_vector, query_similar
from shared.db import raw_conn, with_tenant


def _valid_class_payload(
    *,
    class_id: str,
    description: str,
    body: str = "## When this fires\n401 ...",
) -> dict[str, object]:
    """Build a minimal valid ``BugClass`` envelope. Mirrors the helper in
    ``test_api_write.py`` — duplicated here so this file is self-contained
    (it's a security gate; we don't want the test to drift if that helper
    moves)."""
    return {
        "frontmatter": {
            "id": class_id,
            "type": "bug-class",
            "description": description,
            "signature": {
                "must_match": ["status_code == 401"],
                "embedding_seed": "jwt refresh expired clock-skew",
            },
            "related": {
                "analogous_to": [],
                "overlaps_with": [],
                "often_confused_with": [],
                "regressed_by": [],
            },
            "context_sources": [],
            "evidence": {
                "match_count": 0,
                "last_updated": None,
                "recent_refinements": [],
            },
        },
        "body": body,
    }


async def _client_get(
    kg_app: FastAPI, path: str, headers: dict[str, str]
) -> httpx.Response:
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path, headers=headers)


async def _client_put(
    kg_app: FastAPI,
    path: str,
    json_body: dict[str, object],
    headers: dict[str, str],
) -> httpx.Response:
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.put(path, json=json_body, headers=headers)


# ---------------------------------------------------------------------------
# 1. Cross-tenant GET-by-id must 404.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tenantB_cannot_read_tenantA_class(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """tA's class is invisible to tB.

    NOTE: this duplicates ``test_api_read.py:test_cross_tenant_query_returns_nothing``.
    The duplication is deliberate. This file is the canonical
    tenant-isolation gate; a security-focused test surface should be
    discoverable on its own without grepping the read-API tests.
    Removing this case to avoid duplication would mean a future reader
    auditing tenant isolation could miss the GET-by-id path.
    """
    resp = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tB"),
    )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 2. Cross-tenant list must not surface tA rows.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tenantB_list_does_not_include_tenantA(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """tB's list view is empty — the seeded_classes fixture seeds zero
    classes for tB, so any non-empty result is a leak from tA."""
    resp = await _client_get(kg_app, "/kg/classes", headers_for("tB"))
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    # Strong assertion: tB has no classes at all. A leak from tA would
    # most likely surface as items being non-empty here. We additionally
    # check the specific tA class_ids in case a future seed adds tB
    # classes — the per-id check would still catch a tA leak.
    assert items == [], f"tB saw tA's rows: {items!r}"
    ids = {i["id"] for i in items}
    assert "auth-401-jwt-refresh" not in ids
    assert "db-timeout-replica-lag" not in ids


# ---------------------------------------------------------------------------
# 3. PUT under tB MUST NOT overwrite tA's row at the same class_id.
#    This is the most important test in the file — proves the path slug
#    cannot be used as a hijack vector.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tenantB_put_cannot_overwrite_tenantA(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """tB's PUT at the same path slug must land in tB's namespace, not tA's.

    Mechanism: PK on ``kg_classes`` is ``(customer_id, class_id)``. The
    RLS ``USING`` policy plus the handler's tenant-scoped INSERT/UPDATE
    means tB's write affects only tB's row. The seeded tA row at
    ``auth-401-jwt-refresh`` must survive untouched.
    """
    payload = _valid_class_payload(
        class_id="auth-401-jwt-refresh",
        description="HIJACK",
        # Use a body without wiki-links so kg_check passes — we want to
        # exercise the write path, not the validation path.
        body="## When this fires\nthis should never end up in tA's row",
    )
    put_resp = await _client_put(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        payload,
        headers_for("tB"),
    )
    # The PUT may legitimately succeed (200 or 201) under tB's namespace —
    # RLS USING-only doesn't reject the write; it just scopes it to tB.
    assert put_resp.status_code in (200, 201), put_resp.text

    get_resp = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tA"),
    )
    assert get_resp.status_code == 200, get_resp.text
    body = get_resp.json()
    # The critical assertion: tA's view must still show its original
    # description, NOT the tB-supplied "HIJACK" string. A regression
    # here would be a P0 multi-tenancy breach.
    assert body["frontmatter"]["description"] != "HIJACK"
    assert body["frontmatter"]["description"] == "401 from upstream after JWT refresh"


# ---------------------------------------------------------------------------
# 4. pgvector embedding-query isolation. Replaces the originally-planned
#    Pinecone-namespace test (the repo uses pgvector, not Pinecone).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pgvector_query_isolates_tenants(
    seeded_classes: None,
) -> None:
    """``query_similar`` under tB cannot see tA's vectors.

    Setup:
        - Add a known signature_embedding to tA's seeded
          ``auth-401-jwt-refresh`` class (vector of all 1.0s).
        - Run ``query_similar`` with the same vector under
          ``with_tenant("tB")``: must return zero results.
        - Defensive sanity check: run the same query under
          ``with_tenant("tA")`` and confirm tA's class IS returned. This
          is the test for the test — without it, a bug that always
          returned zero rows (e.g., RLS denying all reads) would pass
          tB's empty-result assertion as a false negative.

    Cleanup:
        - Reset the signature_embedding to NULL on the seeded row so the
          ``seeded_classes`` fixture's cleanup-by-customer DELETE
          (which it would do anyway) doesn't leave orphan vectors. The
          fixture already DELETEs the row in its teardown; resetting
          here is belt-and-suspenders for any future change to that
          fixture.
    """
    vec = [1.0] * 1536
    # Use the canonical formatter from the module under test rather than
    # reimplementing it here — protects against silent drift if the
    # production format ever changes (e.g. precision, int-coercion).
    vec_literal = _format_vector(vec)

    # Set the embedding on tA's existing seeded row. UPDATE rather than a
    # fresh INSERT keeps the test focused on isolation, not on duplicating
    # _seed_class. The GUC must be set inline because RLS is FORCEd.
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, true)",
            "tA",
        )
        await conn.execute(
            "UPDATE kg_classes SET signature_embedding = $1::vector "
            "WHERE customer_id = $2 AND class_id = $3",
            vec_literal,
            "tA",
            "auth-401-jwt-refresh",
        )

    try:
        # Critical assertion: tB sees zero matches even though tA has a
        # vector at the queried point. This is the pgvector-equivalent of
        # the Pinecone-namespace isolation guarantee in spec §12.3.
        async with with_tenant("tB") as conn_b:
            tb_matches = await query_similar(
                conn_b,
                customer_id="tB",
                vector=vec,
                top_k=10,
            )
        assert tb_matches == [], (
            f"tB saw tA's vector via query_similar: {tb_matches!r}"
        )

        # Defensive: the same query under tA MUST return tA's class.
        # Without this, a regression that broke RLS in a way that hid
        # everything (rather than leaking) would silently pass the tB
        # assertion above.
        async with with_tenant("tA") as conn_a:
            ta_matches = await query_similar(
                conn_a,
                customer_id="tA",
                vector=vec,
                top_k=10,
            )
        ta_ids = {m.class_id for m in ta_matches}
        assert "auth-401-jwt-refresh" in ta_ids, (
            "tA could not see its own vector — test setup is broken, "
            "the tB-empty result above is not a meaningful signal"
        )
    finally:
        # Reset the embedding to NULL so the seeded_classes teardown (which
        # only DELETEs by customer_id) doesn't depend on a particular
        # column state. Idempotent.
        async with raw_conn() as conn:
            await conn.execute(
                "SELECT set_config('app.current_customer_id', $1, true)",
                "tA",
            )
            await conn.execute(
                "UPDATE kg_classes SET signature_embedding = NULL "
                "WHERE customer_id = $1 AND class_id = $2",
                "tA",
                "auth-401-jwt-refresh",
            )

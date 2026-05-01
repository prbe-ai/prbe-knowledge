"""Tests for the KG candidates API: GET /kg/candidates.

Covers the contract guarantees from spec §7.3 and the staff dashboard's
triage view (plan Task 25):

  * Default ``status=pending`` filter.
  * ``?status=`` filter restricted to the four-state enum.
  * ``?limit=`` clamped 1..500 by FastAPI's Query validator.
  * Order is ``created_at DESC``.
  * RLS-scoped: tA's GET only returns tA's rows.
  * ``notes_embedding`` is never returned (large; not useful for the UI).

All live-DB tests rely on ``seeded_classes`` to set up tA/tB customers
plus the truncate/cleanup lifecycle; candidates seeded inline get
cleaned up via the ``ON DELETE CASCADE`` on ``customers`` when
``_cleanup`` deletes the customer rows at fixture teardown.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from shared.db import close_pool, raw_conn

# Shape of the ``headers_for`` fixture from conftest: customer_id -> headers.
HeadersFor = Callable[[str], dict[str, str]]


async def _client_get(
    kg_app: FastAPI,
    path: str,
    headers: dict[str, str],
) -> httpx.Response:
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path, headers=headers)


async def _seed_candidate(
    *,
    customer_id: str,
    payload_hash: str = "hash-default",
    payload: dict[str, Any] | None = None,
    status: str = "pending",
    repeat_count: int = 1,
    created_at: datetime | None = None,
    notes_embedding: list[float] | None = None,
) -> str:
    """Insert one ``kg_candidates`` row directly via raw_conn.

    Mirrors the inline-GUC pattern from ``conftest._seed_class`` so RLS
    lets the INSERT through under FORCE ROW LEVEL SECURITY. Returns the
    server-generated ``candidate_id`` as a string for assertions.
    """
    if payload is None:
        payload = {"type": "test", "note": "sample"}
    columns = ["customer_id", "payload_hash", "payload", "status", "repeat_count"]
    values: list[Any] = [customer_id, payload_hash, json.dumps(payload), status, repeat_count]
    placeholders = ["$1", "$2", "$3::jsonb", "$4", "$5"]

    if created_at is not None:
        columns.append("created_at")
        values.append(created_at)
        placeholders.append(f"${len(values)}")

    if notes_embedding is not None:
        # pgvector accepts a stringified bracket-list literal at the SQL layer.
        # asyncpg has no built-in vector codec, so we cast text → vector here.
        columns.append("notes_embedding")
        # Format like '[0.1,0.2,...]'.
        values.append("[" + ",".join(repr(float(x)) for x in notes_embedding) + "]")
        placeholders.append(f"${len(values)}::vector")

    sql = (
        f"INSERT INTO kg_candidates ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders)}) "
        f"RETURNING candidate_id"
    )
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, true)",
            customer_id,
        )
        row = await conn.fetchrow(sql, *values)
        assert row is not None
        return str(row["candidate_id"])


# ---------------------------------------------------------------------------
# Status / limit filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_candidates_returns_pending_by_default(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Three candidates seeded (pending, accepted, pending). Default filter
    returns the two pending rows."""
    await _seed_candidate(customer_id="tA", payload_hash="h1", status="pending")
    await _seed_candidate(customer_id="tA", payload_hash="h2", status="accepted")
    await _seed_candidate(customer_id="tA", payload_hash="h3", status="pending")

    resp = await _client_get(kg_app, "/kg/candidates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_returned"] == 2
    assert len(body["items"]) == 2
    assert all(item["status"] == "pending" for item in body["items"])


@pytest.mark.asyncio
async def test_list_candidates_status_filter(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """``?status=accepted`` returns the one accepted row."""
    await _seed_candidate(customer_id="tA", payload_hash="h1", status="pending")
    await _seed_candidate(customer_id="tA", payload_hash="h2", status="accepted")
    await _seed_candidate(customer_id="tA", payload_hash="h3", status="pending")

    resp = await _client_get(
        kg_app, "/kg/candidates?status=accepted", headers_for("tA")
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_returned"] == 1
    assert body["items"][0]["status"] == "accepted"


@pytest.mark.asyncio
async def test_list_candidates_invalid_status_422(
    kg_app: FastAPI,
    headers_for: HeadersFor,
) -> None:
    """Status outside the four-state enum → 422 from FastAPI's Literal validator."""
    await close_pool()  # no DB needed; validation runs first
    resp = await _client_get(
        kg_app, "/kg/candidates?status=unknown_value", headers_for("tA")
    )
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_list_candidates_limit_clamps(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Seed 5 pending candidates; ``?limit=2`` returns 2."""
    for i in range(5):
        await _seed_candidate(
            customer_id="tA",
            payload_hash=f"h{i}",
            status="pending",
        )

    resp = await _client_get(
        kg_app, "/kg/candidates?limit=2", headers_for("tA")
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_returned"] == 2
    assert len(body["items"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_limit", ["0", "-1", "501"])
async def test_list_candidates_invalid_limit_422(
    kg_app: FastAPI,
    headers_for: HeadersFor,
    bad_limit: str,
) -> None:
    """``limit`` out of [1, 500] → 422 from FastAPI's Query(ge=1, le=500)."""
    await close_pool()
    resp = await _client_get(
        kg_app, f"/kg/candidates?limit={bad_limit}", headers_for("tA")
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Tenant scoping + ordering + projection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_candidates_tenant_scoped(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Seed for both tA and tB; tA's GET must not surface tB's rows.

    ``seeded_classes`` already creates both customer rows. The inline
    GUC inside ``_seed_candidate`` lets the RLS-protected INSERT
    through for whichever tenant we name.
    """
    await _seed_candidate(customer_id="tA", payload_hash="hA")
    await _seed_candidate(customer_id="tB", payload_hash="hB1")
    await _seed_candidate(customer_id="tB", payload_hash="hB2")

    resp = await _client_get(kg_app, "/kg/candidates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_returned"] == 1
    assert body["items"][0]["payload_hash"] == "hA"


@pytest.mark.asyncio
async def test_list_candidates_orders_by_created_at_desc(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Seed 3 candidates with explicit created_at; verify newest-first."""
    base = datetime.now(tz=UTC) - timedelta(hours=1)
    await _seed_candidate(
        customer_id="tA",
        payload_hash="oldest",
        created_at=base,
    )
    await _seed_candidate(
        customer_id="tA",
        payload_hash="middle",
        created_at=base + timedelta(minutes=10),
    )
    await _seed_candidate(
        customer_id="tA",
        payload_hash="newest",
        created_at=base + timedelta(minutes=20),
    )

    resp = await _client_get(kg_app, "/kg/candidates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert [i["payload_hash"] for i in items] == ["newest", "middle", "oldest"]


@pytest.mark.asyncio
async def test_notes_embedding_not_in_response(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """A row with a non-NULL notes_embedding still gets returned, but the
    serialized response must NOT include a ``notes_embedding`` key — the
    vector is large (per-row 6KB-ish) and not useful for the UI."""
    embedding = [0.01] * 1536
    await _seed_candidate(
        customer_id="tA",
        payload_hash="with-emb",
        notes_embedding=embedding,
    )

    resp = await _client_get(kg_app, "/kg/candidates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) == 1
    assert "notes_embedding" not in items[0]

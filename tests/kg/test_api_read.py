"""Tests for the KG read API: GET /kg/classes/{id} and GET /kg/classes.

Covers the four contract guarantees from spec §5.2 / §12.3:

  1. ``test_get_class_returns_envelope`` — a stored class round-trips back as
     a ``BugClass``-shaped JSON envelope (`frontmatter` + `body`).
  2. ``test_get_unknown_class_404`` — missing class_id is a 404, not a 500.
  3. ``test_list_classes_returns_metadata_only`` — the list view returns
     id + description only, never the body. The body would blow up the
     payload (markdown can be many KB) and isn't useful for a list.
  4. ``test_cross_tenant_query_returns_nothing`` — the security-critical
     RLS check: tenant tA's class is invisible to tenant tB even when tB
     guesses the exact class_id. The 404 surfaces because RLS filters
     the row out before the WHERE clause sees it (USING-only policy on
     ``app.current_customer_id``).

All four require a live Postgres + the kg_classes table. The
``tests/kg/conftest.py:pytest_collection_modifyitems`` hook auto-skips
when Postgres isn't reachable, so this file is safe to ship in CI envs
that don't run docker-compose.

Shared fixtures (``kg_app``, ``headers_for``, ``seeded_classes``,
``_patch_settings``) live in ``tests/kg/conftest.py`` so the read- and
write-API tests share the same seeding / auth pattern.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from shared.db import close_pool


async def _client_get(
    kg_app: FastAPI,
    path: str,
    headers: dict[str, str],
) -> httpx.Response:
    """One-shot GET via ASGITransport; mirrors tests/test_query_auth.py."""
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path, headers=headers)


@pytest.mark.asyncio
async def test_get_class_returns_envelope(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    resp = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tA"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["frontmatter"]["id"] == "auth-401-jwt-refresh"
    assert body["frontmatter"]["type"] == "bug-class"
    assert "body" in body
    assert isinstance(body["body"], str)


@pytest.mark.asyncio
async def test_get_unknown_class_404(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    resp = await _client_get(
        kg_app,
        "/kg/classes/does-not-exist",
        headers_for("tA"),
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_list_classes_returns_metadata_only(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    resp = await _client_get(kg_app, "/kg/classes", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    items = payload["items"]
    # tA has two seeded classes, ordered by class_id ascending.
    assert len(items) == 2
    assert items[0]["id"] == "auth-401-jwt-refresh"
    assert items[1]["id"] == "db-timeout-replica-lag"
    assert all("description" in i and i["description"] for i in items)
    # Body must NOT leak into the list view.
    assert all("body" not in i for i in items)


@pytest.mark.asyncio
async def test_cross_tenant_query_returns_nothing(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """Security-critical: tA's class must be invisible to tB.

    RLS filters the row out of the SELECT before the WHERE clause sees
    it; the handler then surfaces "row not found" as a 404. If this ever
    returned 200 with tA's payload, that's a tenant-isolation breach.
    """
    resp = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tB"),
    )
    assert resp.status_code == 404, resp.text


# Auth tests don't need the DB, but they DO depend on the auth dep being
# wired correctly. Keep them here so the read-API contract is checked
# end-to-end in one file.


@pytest.mark.asyncio
async def test_missing_auth_returns_401(kg_app: FastAPI) -> None:
    """No headers → 401, not 200/404. Belt-and-suspenders for the auth dep."""
    # No DB needed; reset the pool reference so an env without Postgres
    # doesn't error out on pool acquisition before the auth check runs.
    await close_pool()
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/kg/classes/anything")
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_missing_auth_on_list_returns_401(kg_app: FastAPI) -> None:
    await close_pool()
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/kg/classes")
    assert resp.status_code == 401, resp.text

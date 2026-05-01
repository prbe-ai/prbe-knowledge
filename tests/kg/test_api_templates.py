"""Tests for the KG templates API: GET /kg/templates and
POST /kg/templates/{id}/apply.

Covers the contract guarantees from spec §5.7 / §7.2 and the staff
dashboard's onboarding flow (plan Task 24):

  * List view returns every loaded template as ``{id, description, domain}``
    sorted by id, with ``domain`` parsed from the leading id segment.
  * Apply is upsert-shaped (201 first time, 200 on idempotent re-apply),
    404 for unknown template ids, and 422 when kg_check fails because
    a referenced template hasn't been applied yet for this tenant.
  * Tenant scoping: applying for tA must not surface to tB.

Live-DB tests rely on the ``seeded_classes`` fixture pulling in
``live_db``; the conftest auto-skip wrapper handles environments without
a running Postgres. The pure list / auth tests don't need the DB.

Shared fixtures (``kg_app``, ``headers_for``, ``seeded_classes``,
``_patch_settings``) live in ``tests/kg/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from services.kg.templates._loader import load_all_templates
from shared.db import close_pool

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


async def _client_post(
    kg_app: FastAPI,
    path: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.post(path, headers=headers, json=json_body or {})


# ---------------------------------------------------------------------------
# GET /kg/templates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_templates_returns_envelope(
    kg_app: FastAPI,
    headers_for: HeadersFor,
) -> None:
    """200 with one item per loaded template — id + description + domain."""
    # No DB needed for the template list (it loads from disk). Reset the
    # pool so envs without Postgres can still run this test.
    await close_pool()
    resp = await _client_get(kg_app, "/kg/templates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    expected_count = len(load_all_templates())
    assert len(items) == expected_count
    assert expected_count == 23  # locks the documented library size
    for item in items:
        assert set(item.keys()) == {"id", "description", "domain"}
        assert item["id"]
        assert item["description"]
        assert item["domain"]


@pytest.mark.asyncio
async def test_list_templates_sorted_by_id(
    kg_app: FastAPI,
    headers_for: HeadersFor,
) -> None:
    """Items must be in ascending order by id, deterministic for the UI."""
    await close_pool()
    resp = await _client_get(kg_app, "/kg/templates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    ids = [i["id"] for i in resp.json()["items"]]
    assert ids == sorted(ids)


@pytest.mark.asyncio
async def test_list_templates_domain_parsed_from_id(
    kg_app: FastAPI,
    headers_for: HeadersFor,
) -> None:
    """Domain is the leading dash-segment of the id."""
    await close_pool()
    resp = await _client_get(kg_app, "/kg/templates", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    for item in resp.json()["items"]:
        assert item["domain"] == item["id"].split("-", 1)[0]


@pytest.mark.asyncio
async def test_list_templates_requires_auth(kg_app: FastAPI) -> None:
    """No headers → 401 (consistent with the rest of the kg surface)."""
    await close_pool()
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/kg/templates")
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# POST /kg/templates/{id}/apply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_template_creates_class(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Apply a refless template for tA → 201, then GET it back as that tenant.

    Picks ``auth-token-leaked`` because its frontmatter has empty
    ``often_confused_with`` and the body has no cross-template wiki-links;
    kg_check passes against tA's pre-seeded universe (which doesn't include
    this id) plus the template's own id.
    """
    resp = await _client_post(
        kg_app,
        "/kg/templates/auth-token-leaked/apply",
        headers_for("tA"),
    )
    assert resp.status_code == 201, resp.text
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["class_id"] == "auth-token-leaked"

    fetched = await _client_get(
        kg_app,
        "/kg/classes/auth-token-leaked",
        headers_for("tA"),
    )
    assert fetched.status_code == 200, fetched.text
    body = fetched.json()
    assert body["frontmatter"]["id"] == "auth-token-leaked"
    assert body["frontmatter"]["type"] == "bug-class"


@pytest.mark.asyncio
async def test_apply_template_already_exists_returns_200(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Idempotent re-apply: second call returns 200, not 201."""
    first = await _client_post(
        kg_app,
        "/kg/templates/auth-token-leaked/apply",
        headers_for("tA"),
    )
    assert first.status_code == 201, first.text

    second = await _client_post(
        kg_app,
        "/kg/templates/auth-token-leaked/apply",
        headers_for("tA"),
    )
    assert second.status_code == 200, second.text
    assert second.json()["class_id"] == "auth-token-leaked"


@pytest.mark.asyncio
async def test_apply_template_404_for_unknown(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Unknown template id → 404, not 422 (it's a missing resource, not bad data)."""
    resp = await _client_post(
        kg_app,
        "/kg/templates/nonexistent-template/apply",
        headers_for("tA"),
    )
    assert resp.status_code == 404, resp.text
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_apply_template_422_when_kg_check_fails(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Two-step assertion: ``auth-401-jwt-refresh`` references
    ``auth-403-rbac`` via ``often_confused_with``. Applying it BEFORE
    ``auth-403-rbac`` is in the tenant's class set must fail 422 with
    the missing id surfaced. After applying the dependency first, the
    second apply succeeds — proving the kg_check universe is the
    tenant's existing classes (per-tenant), not the global template
    universe.

    Note: ``seeded_classes`` pre-seeds tA with a stub ``auth-401-jwt-refresh``
    row whose frontmatter has empty ``often_confused_with``. The first
    apply attempt still fails 422 because ``check_class`` evaluates the
    INCOMING template's refs (which DO reference auth-403-rbac), not the
    pre-seeded row's. Once auth-403-rbac is applied, the retry overwrites
    the seeded stub — so it returns 200 (update), not 201 (insert).
    """
    first = await _client_post(
        kg_app,
        "/kg/templates/auth-401-jwt-refresh/apply",
        headers_for("tA"),
    )
    assert first.status_code == 422, first.text
    assert "auth-403-rbac" in first.json()["detail"]

    # Apply the dependency, then retry the original.
    dep = await _client_post(
        kg_app,
        "/kg/templates/auth-403-rbac/apply",
        headers_for("tA"),
    )
    assert dep.status_code == 201, dep.text

    retry = await _client_post(
        kg_app,
        "/kg/templates/auth-401-jwt-refresh/apply",
        headers_for("tA"),
    )
    # 200 because seeded_classes pre-seeds auth-401-jwt-refresh; this
    # is an update, not an insert.
    assert retry.status_code == 200, retry.text


@pytest.mark.asyncio
async def test_apply_is_tenant_scoped(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for: HeadersFor,
) -> None:
    """Apply for tA → tB cannot see the class.

    Cross-tenant lookup goes through the same RLS policy as the rest of
    the read API, so a 404 here is the security-critical outcome.
    """
    apply_resp = await _client_post(
        kg_app,
        "/kg/templates/auth-token-leaked/apply",
        headers_for("tA"),
    )
    assert apply_resp.status_code == 201, apply_resp.text

    cross = await _client_get(
        kg_app,
        "/kg/classes/auth-token-leaked",
        headers_for("tB"),
    )
    assert cross.status_code == 404, cross.text

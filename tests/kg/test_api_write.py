"""Tests for the KG write API: PUT /kg/classes/{class_id}.

Covers the four contract guarantees from spec §5.2 / §7.2:

  1. ``test_put_creates_class`` — a fresh class for tA gets 201, and the
     row round-trips back through the read endpoint.
  2. ``test_put_updates_class`` — PUT against an existing class_id returns
     200 and the new ``description`` is visible to the read endpoint.
  3. ``test_put_rejects_broken_wiki_link`` — a body that references an
     unknown class_id is 422; the offending id appears in the response
     detail so the dashboard can surface it.
  4. ``test_put_path_id_must_match_payload_id`` — path / payload id
     mismatch is 400 (caught before the lock is acquired).

All four require a live Postgres + the kg_classes table. The
``tests/kg/conftest.py:pytest_collection_modifyitems`` hook auto-skips
when Postgres isn't reachable, mirroring the read-API tests.

Shared fixtures (``kg_app``, ``headers_for``, ``seeded_classes``,
``_patch_settings``) live in ``tests/kg/conftest.py``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport


def _valid_class_payload(
    *,
    class_id: str,
    description: str = "401 from upstream after JWT refresh",
    body: str = "## When this fires\n401 ...",
) -> dict[str, Any]:
    """Build a minimal valid ``BugClass`` envelope as plain JSON.

    Mirrors the shape of ``conftest._seed_class`` so payloads written via
    the API and rows seeded directly look the same when round-tripped.
    """
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


async def _client_put(
    kg_app: FastAPI,
    path: str,
    json_body: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.put(path, json=json_body, headers=headers)


async def _client_get(
    kg_app: FastAPI,
    path: str,
    headers: dict[str, str],
) -> httpx.Response:
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path, headers=headers)


@pytest.mark.asyncio
async def test_put_creates_class(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """Fresh class_id → 201, and the read endpoint can fetch it back."""
    payload = _valid_class_payload(
        class_id="db-deadlock-new",
        description="DB deadlock during concurrent writes",
        body="## When this fires\nlock cycles ...",
    )
    resp = await _client_put(
        kg_app,
        "/kg/classes/db-deadlock-new",
        payload,
        headers_for("tA"),
    )
    assert resp.status_code == 201, resp.text

    fetched = await _client_get(
        kg_app,
        "/kg/classes/db-deadlock-new",
        headers_for("tA"),
    )
    assert fetched.status_code == 200, fetched.text
    body = fetched.json()
    assert body["frontmatter"]["id"] == "db-deadlock-new"
    assert body["frontmatter"]["description"] == "DB deadlock during concurrent writes"
    assert body["body"] == "## When this fires\nlock cycles ..."


@pytest.mark.asyncio
async def test_put_updates_class(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """Existing class_id → 200, and the read endpoint reflects the update."""
    payload = _valid_class_payload(
        class_id="auth-401-jwt-refresh",
        description="updated description after refinement",
    )
    resp = await _client_put(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        payload,
        headers_for("tA"),
    )
    assert resp.status_code == 200, resp.text

    fetched = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tA"),
    )
    assert fetched.status_code == 200, fetched.text
    assert (
        fetched.json()["frontmatter"]["description"]
        == "updated description after refinement"
    )


@pytest.mark.asyncio
async def test_put_rejects_broken_wiki_link(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """A wiki-link to an unknown class_id is a 422 with the offending id surfaced."""
    payload = _valid_class_payload(
        class_id="db-deadlock-new",
        body="See [[ghost]] for the related class.",
    )
    resp = await _client_put(
        kg_app,
        "/kg/classes/db-deadlock-new",
        payload,
        headers_for("tA"),
    )
    assert resp.status_code == 422, resp.text
    assert "ghost" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_put_path_id_must_match_payload_id(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """Path and payload class_id must match — 400 otherwise."""
    payload = _valid_class_payload(class_id="aaa")
    resp = await _client_put(
        kg_app,
        "/kg/classes/bbb",
        payload,
        headers_for("tA"),
    )
    assert resp.status_code == 400, resp.text

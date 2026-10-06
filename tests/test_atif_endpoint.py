"""GET /trajectory/{doc_id}: an agent session as ATIF, paged by step.

Tenancy is the document row's: read under the caller's tenant, so another
tenant's doc id, a deleted document or a doc that is not an agent session is a
404 before R2 is touched. A session whose trajectory has not been written yet
answers 404 with `reason: not_built` (the reader falls back to text).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import orjson
import pytest
from httpx import ASGITransport

from engine.ingest.atif.build import build_trajectory
from engine.ingest.atif.store import trajectory_key
from engine.shared.config import Settings, get_settings
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.storage import StorageNotFound
from tests.test_sources import _seed_customer

SESSION = "11111111-2222-3333-4444-555555555555"


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.gets: list[str] = []

    async def bucket_for(self, customer_id: str) -> str:
        return f"bucket-{customer_id}"

    async def get(self, bucket: str, key: str) -> bytes:
        self.gets.append(key)
        try:
            return self.objects[(bucket, key)]
        except KeyError:
            raise StorageNotFound(key) from None


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    import engine.retrieval.main as retrieval_main

    fake = FakeStore()
    monkeypatch.setattr(retrieval_main, "get_store", lambda: fake)
    return fake


async def _customer(customer_id: str) -> str:
    # The /sources suite's seeding: a customer row and its plaintext api key.
    return await _seed_customer(customer_id)


async def _doc(customer_id: str, doc_id: str, *, doc_type: str = "claude_code.session",
               source: str = "claude_code", deleted: bool = False,
               visibility: str = "approved", superseded: bool = False) -> None:
    now = datetime.now(UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id, source_system, source_id, source_url,
                doc_class, doc_type, content_type, content_hash, title,
                body_size_bytes, body_token_count, created_at, updated_at,
                valid_from, ingested_at, acl, metadata, deleted_at, visibility, valid_to
            ) VALUES ($1, 1, $2, $3, $4, 'https://x', 'raw_source', $5,
                      'application/json', 'h', 't', 1, 0, $6, $6, $6, $6, '{}'::jsonb,
                      '{}'::jsonb, $7, $8, $9)
            """,
            doc_id, customer_id, source, SESSION, doc_type, now, now if deleted else None,
            visibility, now if superseded else None,
        )


def _trajectory(n_users: int) -> dict[str, Any]:
    events = [{"line_no": i, "raw": {"type": "user", "message": {"content": f"turn {i}"}}}
              for i in range(n_users)]
    return build_trajectory(events, session_id=SESSION, agent_name="claude_code").trajectory


async def _get(path: str, api_key: str | None) -> httpx.Response:
    from engine.retrieval.main import app

    await close_pool()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with (
        httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return await client.get(path, headers=headers)


async def test_requires_bearer(live_db, settings, store) -> None:
    resp = await _get("/trajectory/x", None)
    await init_pool(settings)
    assert resp.status_code == 401


async def test_pages_steps_and_hides_render_provenance(live_db, settings, store) -> None:
    key = await _customer("cust-a")
    doc_id = f"claude_code:cust-a:{SESSION}"
    await _doc("cust-a", doc_id)
    store.objects[("bucket-cust-a", trajectory_key("claude_code", "cust-a", SESSION))] = (
        orjson.dumps(_trajectory(5))
    )

    first = await _get(f"/trajectory/{doc_id}?step_limit=2", key)
    last = await _get(f"/trajectory/{doc_id}?step_from=5&step_limit=2", key)
    await init_pool(settings)

    assert first.status_code == 200, first.text
    body = first.json()
    assert (body["total_steps"], body["step_from"], body["next_step_from"]) == (5, 1, 3)
    assert body["session_id"] == SESSION and "session_complete" not in body
    steps = body["trajectory"]["steps"]
    assert [s["message"] for s in steps] == ["turn 0", "turn 1"]
    assert "extra" not in body["trajectory"], "render provenance is engine-internal"
    assert all("extra" not in s for s in steps)
    assert last.json()["next_step_from"] is None
    assert [s["step_id"] for s in last.json()["trajectory"]["steps"]] == [5]


async def test_not_built_is_a_404_with_a_reason(live_db, settings, store) -> None:
    key = await _customer("cust-b")
    doc_id = f"claude_code:cust-b:{SESSION}"
    await _doc("cust-b", doc_id)
    resp = await _get(f"/trajectory/{doc_id}", key)
    await init_pool(settings)
    assert resp.status_code == 404
    assert resp.json()["reason"] == "not_built"


@pytest.mark.parametrize(
    "case", ["other tenant", "deleted", "not a session", "unknown", "draft", "superseded"]
)
async def test_unreadable_documents_never_reach_storage(live_db, settings, store, case) -> None:
    key = await _customer("cust-c")
    await _customer("cust-d")
    doc_id = f"claude_code:cust-d:{SESSION}"
    if case == "other tenant":
        await _doc("cust-d", doc_id)
    elif case == "deleted":
        doc_id = f"claude_code:cust-c:{SESSION}"
        await _doc("cust-c", doc_id, deleted=True)
    elif case == "not a session":
        doc_id = "slack:T:C:1"
        await _doc("cust-c", doc_id, doc_type="slack_message", source="slack")
    elif case in ("draft", "superseded"):
        doc_id = f"claude_code:cust-c:{SESSION}"
        await _doc("cust-c", doc_id, visibility="draft" if case == "draft" else "approved",
                   superseded=case == "superseded")
    resp = await _get(f"/trajectory/{doc_id}", key)
    await init_pool(settings)
    assert resp.status_code == 404
    assert "reason" not in resp.json()
    assert store.gets == []


async def test_an_unreadable_object_is_not_built(live_db, settings, store) -> None:
    key = await _customer("cust-e")
    doc_id = f"claude_code:cust-e:{SESSION}"
    await _doc("cust-e", doc_id)
    store.objects[("bucket-cust-e", trajectory_key("claude_code", "cust-e", SESSION))] = b"{oops"
    resp = await _get(f"/trajectory/{doc_id}", key)
    await init_pool(settings)
    assert resp.status_code == 404 and resp.json()["reason"] == "not_built"


async def test_a_page_stops_at_its_byte_budget(live_db, settings, store, monkeypatch) -> None:
    import engine.retrieval.main as retrieval_main

    monkeypatch.setattr(retrieval_main, "_TRAJECTORY_PAGE_MAX_BYTES", 150)
    key = await _customer("cust-f")
    doc_id = f"claude_code:cust-f:{SESSION}"
    await _doc("cust-f", doc_id)
    store.objects[("bucket-cust-f", trajectory_key("claude_code", "cust-f", SESSION))] = (
        orjson.dumps(_trajectory(5))
    )
    resp = await _get(f"/trajectory/{doc_id}?step_limit=5", key)
    await init_pool(settings)
    body = resp.json()
    shown = len(body["trajectory"]["steps"])
    assert 1 <= shown < 5 and body["next_step_from"] == 1 + shown

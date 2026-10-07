"""GET /trajectory/{doc_id}: an agent session as ATIF, paged by step.

Tenancy is the document row's: read under the caller's tenant, so another
tenant's doc id, a deleted document or a doc that is not an agent session is a
404 before R2 is touched. A session whose trajectory has not been written yet
answers 404 with `reason: not_built` (the reader falls back to text).

Paging reads the object once per version (engine/retrieval/trajectory_cache.py):
every page revalidates with a conditional GET, so a rewritten (live) or removed
trajectory is never served from the cache, and tenants never share an entry.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import httpx
import orjson
import pytest
from httpx import ASGITransport

from engine.ingest.atif.build import build_trajectory
from engine.ingest.atif.store import trajectory_key
from engine.retrieval.trajectory_cache import trajectory_cache
from engine.shared.config import Settings, get_settings
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.storage import ObjectRead, StorageNotFound
from tests.test_sources import _seed_customer

SESSION = "11111111-2222-3333-4444-555555555555"


class FakeStore:
    """S3's conditional GET over a dict; the ETag is the body's MD5."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.gets: list[str] = []
        #: The ETag each read revalidated with (None: it held no copy).
        self.sent_etags: list[str | None] = []
        #: Body bytes each read transferred (0 for a 304 or a 404).
        self.transferred: list[int] = []

    async def bucket_for(self, customer_id: str) -> str:
        return f"bucket-{customer_id}"

    async def get_if_changed(self, bucket: str, key: str, etag: str | None) -> ObjectRead:
        self.gets.append(key)
        self.sent_etags.append(etag)
        try:
            body = self.objects[(bucket, key)]
        except KeyError:
            self.transferred.append(0)
            raise StorageNotFound(key) from None
        current = f'"{hashlib.md5(body).hexdigest()}"'
        if etag == current:
            self.transferred.append(0)
            return ObjectRead(body=None, etag=current)
        self.transferred.append(len(body))
        return ObjectRead(body=body, etag=current)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _empty_cache() -> None:
    trajectory_cache.clear()


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
    return (await _get_many([path], api_key))[0]


async def _get_many(paths: list[str], api_key: str | None) -> list[httpx.Response]:
    """Requests in order, in one app process (one cache)."""
    from engine.retrieval.main import app

    await close_pool()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with (
        httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return [await client.get(path, headers=headers) for path in paths]


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


async def test_the_page_budget_counts_utf8_bytes(live_db, settings, store, monkeypatch) -> None:
    import engine.retrieval.main as retrieval_main

    trajectory = _trajectory(4)
    for step in trajectory["steps"]:
        step["message"] = "漢" * 100  # 100 characters, 300 bytes
    from engine.ingest.atif.store import strip_provenance

    one = len(orjson.dumps(strip_provenance(trajectory)["steps"][0]))
    monkeypatch.setattr(retrieval_main, "_TRAJECTORY_PAGE_MAX_BYTES", one * 2)
    key = await _customer("cust-g")
    doc_id = f"claude_code:cust-g:{SESSION}"
    await _doc("cust-g", doc_id)
    store.objects[("bucket-cust-g", trajectory_key("claude_code", "cust-g", SESSION))] = (
        orjson.dumps(trajectory)
    )
    resp = await _get(f"/trajectory/{doc_id}?step_limit=4", key)
    await init_pool(settings)
    assert len(resp.json()["trajectory"]["steps"]) == 2


async def test_paging_reads_the_object_once_per_version(live_db, settings, store) -> None:
    """Every page revalidates (a 304, no body); a rewrite -- the live session's
    next refresh -- is served from the next page on, never the old copy."""
    key = await _customer("cust-h")
    doc_id = f"claude_code:cust-h:{SESSION}"
    await _doc("cust-h", doc_id)
    obj = ("bucket-cust-h", trajectory_key("claude_code", "cust-h", SESSION))
    store.objects[obj] = orjson.dumps(_trajectory(5))
    pages = [f"/trajectory/{doc_id}?step_from={n}&step_limit=2" for n in (1, 3, 5)]
    walk = await _get_many(pages, key)
    assert [r.status_code for r in walk] == [200, 200, 200]
    assert [s["message"] for r in walk for s in r.json()["trajectory"]["steps"]] == [
        f"turn {i}" for i in range(5)
    ]
    assert store.transferred == [len(store.objects[obj]), 0, 0], "one body for three pages"

    store.objects[obj] = orjson.dumps(_trajectory(7))  # the session grew
    grown = await _get_many([f"/trajectory/{doc_id}?step_from=5&step_limit=5"] * 2, key)
    await init_pool(settings)
    body = grown[0].json()
    assert body["total_steps"] == 7
    assert [s["message"] for s in body["trajectory"]["steps"]] == ["turn 4", "turn 5", "turn 6"]
    assert grown[1].json() == body
    assert store.transferred[3:] == [len(store.objects[obj]), 0]


async def test_a_removed_trajectory_is_never_served_from_cache(live_db, settings, store) -> None:
    key = await _customer("cust-i")
    doc_id = f"claude_code:cust-i:{SESSION}"
    await _doc("cust-i", doc_id)
    obj = ("bucket-cust-i", trajectory_key("claude_code", "cust-i", SESSION))
    store.objects[obj] = orjson.dumps(_trajectory(3))
    first = await _get(f"/trajectory/{doc_id}", key)
    del store.objects[obj]  # a resume or a failed write removes it
    second = await _get(f"/trajectory/{doc_id}", key)
    await init_pool(settings)
    assert first.status_code == 200
    assert second.status_code == 404 and second.json()["reason"] == "not_built"


async def test_tenants_never_share_a_cached_trajectory(live_db, settings, store) -> None:
    """Two tenants, the same session id: B never gets A's cached copy."""
    key_a = await _customer("cust-j")
    key_b = await _customer("cust-k")
    doc_a, doc_b = f"claude_code:cust-j:{SESSION}", f"claude_code:cust-k:{SESSION}"
    await _doc("cust-j", doc_a)
    await _doc("cust-k", doc_b)
    traj_a = _trajectory(2)
    traj_a["steps"][0]["message"] = "tenant j secret"
    store.objects[("bucket-cust-j", trajectory_key("claude_code", "cust-j", SESSION))] = (
        orjson.dumps(traj_a)
    )
    a = await _get(f"/trajectory/{doc_a}", key_a)
    b_none = await _get(f"/trajectory/{doc_b}", key_b)
    # B's read held no copy: A's cached entry is not B's, not even to revalidate.
    assert store.sent_etags == [None, None]
    store.objects[("bucket-cust-k", trajectory_key("claude_code", "cust-k", SESSION))] = (
        orjson.dumps(_trajectory(2))
    )
    b = await _get(f"/trajectory/{doc_b}", key_b)
    # Tenant B naming tenant A's document is the document lookup's 404.
    b_on_a = await _get(f"/trajectory/{doc_a}", key_b)
    await init_pool(settings)
    assert a.json()["trajectory"]["steps"][0]["message"] == "tenant j secret"
    assert b_none.status_code == 404 and b_none.json()["reason"] == "not_built"
    assert b.status_code == 200
    assert b.json()["trajectory"]["steps"][0]["message"] == "turn 0"
    assert "tenant j secret" not in b.text
    assert b_on_a.status_code == 404 and "tenant j secret" not in b_on_a.text


async def test_step_from_past_the_end_and_out_of_range(live_db, settings, store) -> None:
    key = await _customer("cust-l")
    doc_id = f"claude_code:cust-l:{SESSION}"
    await _doc("cust-l", doc_id)
    store.objects[("bucket-cust-l", trajectory_key("claude_code", "cust-l", SESSION))] = (
        orjson.dumps(_trajectory(3))
    )
    past, absurd = await _get_many(
        [f"/trajectory/{doc_id}?step_from=9", f"/trajectory/{doc_id}?step_from={2**64}"], key
    )
    await init_pool(settings)
    assert past.status_code == 200
    body = past.json()
    assert body["trajectory"]["steps"] == [] and body["next_step_from"] is None
    assert body["total_steps"] == 3
    assert absurd.status_code == 422, "refused, not a 500 from the encoder"

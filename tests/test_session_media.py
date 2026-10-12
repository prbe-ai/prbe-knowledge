"""Pasted images of captured sessions (kb/session_media.py), against real
Postgres + MinIO.

Sessions go through the REAL pipeline: the receipt door (kb/session_receipts.
accept), the worker's pass (Normalizer.process_queue_row: fetch_supplementary
-> after_fetch -> normalize -> persist, only the LLM extraction stubbed), the
deletion (kb/session_deletion) and the routes over ASGI. Uploads are real
presigned PUTs to MinIO, so the signed headers are checked by a store, not by
this file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from pathlib import Path

import httpx
import orjson
import pytest
import pytest_asyncio

from engine.ingest.atif.fragment import fragment
from engine.ingest.atif.store import trajectory_key
from engine.shared import claude_code_extraction as ext
from engine.shared import db as db_module
from engine.shared.config import get_settings
from engine.shared.constants import SourceSystem
from engine.shared.session_suppression import media_staging_prefix, session_folder
from engine.shared.storage import get_store
from kb import session_deletion as sd
from kb import session_media as sm
from kb import session_receipts as sr
from tests.test_session_deletion import (  # noqa: F401  # pytest fixture, used by name
    _identity,
    _mine,
    env,
)

CC = SourceSystem.CLAUDE_CODE
ALICE = "11111111-1111-4111-8111-111111111111"
EMAIL = "alice@example.test"
ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"pasted screenshot bytes " * 40


def _sid() -> str:
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def _stub_extraction(monkeypatch):
    """The model call is the only thing stubbed (a protocol-3 pass hands it
    the fragments' Lines too)."""

    async def fake_extract(*, session_id, events, cwd=None, agent="claude_code", cache=None,
                           lines=None):
        return ext.UnitBundle()

    monkeypatch.setattr("kb.handlers.claude_code._ext.extract_units_from_session", fake_extract)


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _headers(customer: str) -> dict:
    return {"X-Internal-Knowledge-Key": "test-internal-key", "X-Prbe-Customer": customer}


def _client() -> httpx.AsyncClient:
    from kb.ingestion_app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest_asyncio.fixture
async def media(env, monkeypatch):  # noqa: F811
    """Tenant A has images (and protocol 3 at fragment versions 1 and 2);
    tenant B does not."""
    (a, b), store = env
    monkeypatch.setenv("SESSION_MEDIA_CUSTOMERS", f"other, {a}")
    monkeypatch.setenv("SESSION_PROTOCOL3_CUSTOMERS", a)
    monkeypatch.setenv("SESSION_FRAGMENT_VERSIONS", "1,2")
    # As research runs: a protocol-2 session's trajectory is the fold of its
    # fragments, which shows images (the frozen builder cannot).
    monkeypatch.setenv("SESSION_ATIF_BUILDER", "fold")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield a, b, store
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _settings_changed(monkeypatch, **env_values: str) -> None:
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def ask(customer: str, sid: str, body: bytes = PNG, *, media_type: str = "image/png",
              sha256: str | None = None, size: int | None = None) -> httpx.Response:
    async with _client() as client:
        return await client.post(
            f"/api/sessions/claude_code/{sid}/media",
            headers=_headers(customer),
            json={"sha256": sha256 or _sha(body), "size_bytes": len(body) if size is None else size,
                  "media_type": media_type},
        )


async def put(answer: dict, body: bytes, headers: dict | None = None) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.request(
            answer["method"], answer["url"], content=body,
            headers=answer["headers"] if headers is None else headers,
        )


async def upload(customer: str, sid: str, body: bytes = PNG, media_type: str = "image/png") -> None:
    answer = await ask(customer, sid, body, media_type=media_type)
    assert answer.status_code == 200 and answer.json()["status"] == "upload", answer.text
    stored = await put(answer.json(), body)
    assert stored.status_code == 200, stored.text


def _image_event(i: int, digests: list[tuple[str, str]], text: str = "what is this?") -> dict:
    return {"line_no": i, "raw": {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": text},
        *({"type": "image", "mimeType": mime, "bytes": 999, "sha256": sha} for sha, mime in digests),
    ]}}}


def _reply(i: int) -> dict:
    return {"line_no": i, "raw": {"type": "assistant", "inference_id": f"m{i}",
                                  "message": {"role": "assistant", "content": [
                                      {"type": "text", "text": "a chart"}]}}}


def _batches(sid: str, events: list[dict], *, protocol: int = 2, version: int = 2) -> list[dict]:
    """One batch of `events` and its finalize, as a tap sends them."""
    stream, n = str(uuid.uuid4()), len(events)
    prefix = hashlib.sha256(sid.encode()).hexdigest()
    first = dict(
        protocol_version=protocol, session_id=sid, stream_id=stream, batch_seq=0,
        source_byte_start=0, source_byte_end=n * 100, source_line_start=0, source_line_end=n,
        event_start=0, event_end=n, prefix_sha256=prefix, cwd="/work", **_identity(ALICE, EMAIL),
    )
    end = dict(
        protocol_version=protocol, session_id=sid, stream_id=stream, batch_seq=1,
        source_byte_start=n * 100, source_byte_end=n * 100, source_line_start=n,
        source_line_end=n, event_start=n, event_end=n, prefix_sha256=prefix, finalize=True,
        **_identity(ALICE, EMAIL),
    )
    if protocol == 3:
        first.update(fragments=[fragment(e, version) for e in events], fragment_version=version)
        end["fragment_version"] = version
    else:
        first["events"], end["events"] = events, []
    return [first, end]


async def accept_all(customer: str, batches: list[dict]) -> None:
    for body in batches:
        result = await sr.accept(body, customer, CC, get_store())
        assert result["status"] == "accepted", result


async def row(customer: str, sid: str, sha: str):
    async with db_module.with_tenant(customer) as conn:
        return await conn.fetchrow(
            "SELECT * FROM session_media WHERE customer_id = $1 AND session_id = $2 AND sha256 = $3",
            customer, sid, sha,
        )


async def objects(store, customer: str, prefix: str = "raw/") -> list[str]:
    return sorted(await store.list_keys(await store.bucket_for(customer), prefix))


def _final(customer: str, sid: str, sha: str, ext: str = "png") -> str:
    return f"{session_folder(CC.value, customer, sid)}media/{sha}.{ext}"


def _staged(customer: str, sid: str, sha: str) -> str:
    return f"{media_staging_prefix(CC.value, customer, sid)}{sha}"


async def delete(customer: str, sids: list[str]) -> dict:
    refs = [sd.SessionRef(CC.value, s) for s in sids]
    await sd.record_sessions(customer, refs, deletion_id=str(uuid.uuid4()), reason="test",
                             ticket=None, selector={"by": "id"})
    return await sd.run_deletion(customer, refs, grace_s=0)


# --- schema --------------------------------------------------------------------------


def test_the_bootstrap_schema_carries_the_migrations_ddl_verbatim() -> None:
    # Prod runs migration 0149 (kb/session_media_schema.sql); CI and fresh
    # installs build from db/schema.sql. They must not drift.
    ddl = (ROOT / "kb/session_media_schema.sql").read_text().strip()
    assert ddl in (ROOT / "db/schema.sql").read_text()
    migration = (ROOT / "db/migrations/versions/20261012_0149_session_media.py").read_text()
    assert "kb/session_media_schema.sql" in migration


def test_the_engine_schema_copy_names_an_uploaded_images_digest() -> None:
    import jsonschema

    schema = json.loads((ROOT / "engine/ingest/probe_events/probe-events-1.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)

    def event(block: dict) -> dict:
        return {"type": "user", "message": {"role": "user", "content": [block]}}

    block = {"type": "image", "mimeType": "image/png", "bytes": 3}
    assert not list(validator.iter_errors(event(block)))
    assert not list(validator.iter_errors(event({**block, "sha256": "a" * 64})))
    for bad in ("A" * 64, "a" * 63, 5):
        assert list(validator.iter_errors(event({**block, "sha256": bad}))), bad
    assert list(validator.iter_errors(event({**block, "path": "x"}))), "still closed"


@pytest.mark.asyncio
async def test_the_app_role_sees_its_own_tenants_rows_only(media) -> None:
    """The tests connect as a superuser, which FORCE RLS never binds. The
    production role is not one: it needs the grant, and it must see a row
    only under that tenant's GUC."""
    a, b, _store = media
    sid = _sid()
    assert (await ask(a, sid)).status_code == 200
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "DO $$ BEGIN CREATE ROLE app NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        await conn.execute((ROOT / "kb/session_media_schema.sql").read_text())
        assert await conn.fetchval(
            "SELECT relforcerowsecurity FROM pg_class WHERE relname = 'session_media'"
        )
        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert await conn.fetchval(
                "SELECT has_table_privilege('app', 'session_media', $1)", privilege
            ), privilege
    async with db_module.with_tenant(a) as conn:
        await conn.execute("SET LOCAL ROLE app")
        assert await conn.fetchval("SELECT count(*) FROM session_media") == 1
    async with db_module.with_tenant(b) as conn:
        await conn.execute("SET LOCAL ROLE app")
        assert await conn.fetchval("SELECT count(*) FROM session_media") == 0


# --- the upload route ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_upload_url_signs_length_type_and_digest_for_the_staging_key(media) -> None:
    a, _b, store = media
    sid = _sid()
    answer = await ask(a, sid)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["status"] == "upload" and body["method"] == "PUT"
    assert body["headers"] == {
        "Content-Length": str(len(PNG)),
        "Content-Type": "image/png",
        "x-amz-checksum-sha256": sm.checksum_b64(_sha(PNG)),
    }
    signed = httpx.URL(body["url"]).params["X-Amz-SignedHeaders"].split(";")
    assert {"content-length", "content-type", "x-amz-checksum-sha256"} <= set(signed)
    assert httpx.URL(body["url"]).params["X-Amz-Expires"] == "900"
    assert f"/{_staged(a, sid, _sha(PNG))}" in httpx.URL(body["url"]).path
    # The store refuses anything but the bytes the URL was signed for.
    other = b"X" * len(PNG)
    assert (await put(body, other)).status_code == 400
    assert (await put(body, PNG, {k: v for k, v in body["headers"].items()
                                  if k != "x-amz-checksum-sha256"})).status_code in (400, 403)
    assert (await put(body, PNG[:-1], {**body["headers"],
                                       "Content-Length": str(len(PNG) - 1)})).status_code in (400, 403)
    assert await objects(store, a) == []
    assert (await put(body, PNG)).status_code == 200
    assert await objects(store, a) == [_staged(a, sid, _sha(PNG))], "staged, not in the session"
    record = await row(a, sid, _sha(PNG))
    assert (record["size_bytes"], record["media_type"], record["promoted_at"]) == (
        len(PNG), "image/png", None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"sha256": "A" * 64}, 422),
        ({"sha256": "a" * 63}, 422),
        ({"sha256": "../" + "a" * 61}, 422),
        ({"size": 0}, 422),
        ({"media_type": "image/svg+xml"}, 415),
        ({"media_type": "text/html"}, 415),
        ({"size": 5 * 1024**3 + 1}, 413),
    ],
    ids=str,
)
async def test_a_malformed_or_unsupported_upload_is_refused(media, change, status) -> None:
    a, _b, store = media
    answer = await ask(a, _sid(), **change)
    assert answer.status_code == status, answer.text
    if status == 415:
        assert answer.json()["detail"] == {
            "reason": "media_type_unsupported",
            "media_types": ["image/gif", "image/jpeg", "image/png", "image/webp"],
        }
    if status == 413:
        assert answer.json()["detail"] == {"reason": "media_too_large", "max_bytes": 5 * 1024**3}
    async with db_module.with_tenant(a) as conn:
        assert await conn.fetchval("SELECT count(*) FROM session_media") == 0
    assert await objects(store, a) == []


@pytest.mark.asyncio
async def test_identity_and_auth_are_checked(media) -> None:
    a, _b, _store = media
    async with _client() as client:
        body = {"sha256": _sha(PNG), "size_bytes": len(PNG), "media_type": "image/png"}
        unauth = await client.post(f"/api/sessions/claude_code/{_sid()}/media", json=body,
                                   headers={"X-Prbe-Customer": a})
        assert unauth.status_code == 401
        for path in ("/api/sessions/claude_code/not-a-uuid/media",
                     f"/api/sessions/slack/{_sid()}/media"):
            assert (await client.post(path, json=body, headers=_headers(a))).status_code == 422
        for wrong in ({**body, "size_bytes": "12"}, {**body, "size_bytes": True},
                      {k: v for k, v in body.items() if k != "sha256"}):
            refused = await client.post(f"/api/sessions/claude_code/{_sid()}/media", json=wrong,
                                        headers=_headers(a))
            assert refused.status_code == 422, wrong
        # The read route builds a key from the same three values.
        for path in (f"/api/sessions/claude_code/{_sid()}/media/{'A' * 64}",
                     f"/api/sessions/claude_code/{_sid()}/media/..%2F{'a' * 61}",
                     f"/api/sessions/claude_code/not-a-uuid/media/{'a' * 64}",
                     f"/api/sessions/slack/{_sid()}/media/{'a' * 64}"):
            got = await client.get(path, headers=_headers(a))
            assert got.status_code in (404, 422) and got.json().get("detail") != {
                "reason": "media_not_found"
            }, path


@pytest.mark.asyncio
async def test_a_customer_without_images_is_refused_and_not_advertised(media, monkeypatch) -> None:
    a, b, _store = media
    sid = _sid()
    refused = await ask(b, sid)
    assert refused.status_code == 403 and refused.json()["detail"] == {"reason": "media_disabled"}
    async with _client() as client:
        for customer, advertised in ((a, True), (b, False)):
            read = await client.get(f"/api/sessions/claude_code/{sid}/receipts",
                                    headers=_headers(customer))
            assert read.status_code == 200
            assert read.json()["accepts"].get("media") is (True if advertised else None)
    # The flip: every customer.
    _settings_changed(monkeypatch, SESSION_MEDIA_ALL="true", SESSION_MEDIA_CUSTOMERS="")
    assert (await ask(b, sid)).status_code == 200
    # Withdrawn: new uploads stop (images already stored stay readable, below).
    _settings_changed(monkeypatch, SESSION_MEDIA_ALL="false")
    assert (await ask(a, _sid())).status_code == 403


@pytest.mark.asyncio
async def test_a_held_tenant_takes_no_upload(media) -> None:
    a, _b, _store = media
    async with db_module.raw_conn() as conn:
        await conn.execute("UPDATE customers SET status = 'terminated' WHERE customer_id = $1", a)
    refused = await ask(a, _sid())
    assert refused.status_code == 409 and refused.json()["detail"]["reason"] == "tenant_not_active"


@pytest.mark.asyncio
async def test_a_deleted_session_is_refused_410(media) -> None:
    a, _b, store = media
    sid = _sid()
    await delete(a, [sid])
    refused = await ask(a, sid)
    assert refused.status_code == 410
    assert refused.json()["detail"]["reason"] == "session_deleted"
    assert await objects(store, a) == []


@pytest.mark.asyncio
async def test_the_daily_budget_counts_each_image_once_per_window(media, monkeypatch) -> None:
    a, b, _store = media
    _settings_changed(monkeypatch, SESSION_MEDIA_DAILY_BYTES=str(2 * len(PNG) + 10),
                      SESSION_MEDIA_ALL="true")
    s1, s2 = _sid(), _sid()
    second = PNG + b"2"
    assert (await ask(a, s1)).status_code == 200
    # Asking again for the same image (an expired URL) does not count it twice.
    assert (await ask(a, s1)).status_code == 200
    assert (await ask(a, s2, second)).status_code == 200
    over = await ask(a, s2, PNG + b"3")
    assert over.status_code == 429
    assert over.json()["detail"] == {
        "reason": "media_budget",
        "limit_bytes": 2 * len(PNG) + 10,
        "used_bytes": 2 * len(PNG) + 1,
        "window_s": 86400,
    }
    # Per customer: tenant B's budget is its own.
    assert (await ask(b, _sid())).status_code == 200
    # Uploads older than the window no longer count.
    async with db_module.with_tenant(a) as conn:
        await conn.execute("UPDATE session_media SET created_at = now() - interval '25 hours'")
    assert (await ask(a, s2, PNG + b"3")).status_code == 200


# --- promotion (the worker) --------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_accepted_batch_promotes_its_images_and_the_trajectory_points_at_them(
    media,
) -> None:
    a, _b, store = media
    sid = _sid()
    jpeg = b"\xff\xd8\xff" + b"jpeg bytes" * 30
    await upload(a, sid, PNG)
    await upload(a, sid, jpeg, "image/jpeg")
    events = [_image_event(0, [(_sha(PNG), "image/png"), (_sha(jpeg), "image/jpeg")]), _reply(1)]
    await accept_all(a, _batches(sid, events))
    # Accepting the batch moves nothing: only the worker promotes.
    assert await store.exists(await store.bucket_for(a), _staged(a, sid, _sha(PNG)))
    assert (await row(a, sid, _sha(PNG)))["promoted_at"] is None

    await _mine(a, CC, sid)

    bucket = await store.bucket_for(a)
    png_key, jpeg_key = _final(a, sid, _sha(PNG)), _final(a, sid, _sha(jpeg), "jpg")
    assert await store.get(bucket, png_key) == PNG
    assert (await store.head(bucket, png_key)).content_type == "image/png"
    assert (await store.head(bucket, jpeg_key)).content_type == "image/jpeg"
    assert await objects(store, a, media_staging_prefix(CC.value, a, sid)) == [], "staging emptied"
    assert (await row(a, sid, _sha(PNG)))["promoted_at"] is not None
    # The stored trajectory names the images where they are.
    trajectory = orjson.loads(await store.get(bucket, trajectory_key(CC.value, a, sid)))
    first = trajectory["steps"][0]["message"]
    assert [p.get("source", {}).get("path") for p in first] == [
        None, f"media/{_sha(PNG)}.png", f"media/{_sha(jpeg)}.jpg"
    ]
    assert png_key == f"{session_folder(CC.value, a, sid)}{first[1]['source']['path']}"
    # Another pass promotes nothing twice.
    assert await sm.promote_named_media(a, CC.value, sid, [_sha(PNG), _sha(jpeg)]) == {}
    # Already stored: the next request for it uploads nothing.
    have = await ask(a, sid)
    assert have.status_code == 200 and have.json() == {"status": "have"}

    # Read: a short-lived URL that serves the stored type.
    async with _client() as client:
        got = await client.get(f"/api/sessions/claude_code/{sid}/media/{_sha(PNG)}",
                               headers=_headers(a))
        assert got.status_code == 200, got.text
        read = got.json()
        assert (read["media_type"], read["size_bytes"]) == ("image/png", len(PNG))
        assert httpx.URL(read["url"]).params["X-Amz-Expires"] == "300"
        assert httpx.URL(read["url"]).params["response-content-type"] == "image/png"
        missing = await client.get(f"/api/sessions/claude_code/{sid}/media/{'0' * 64}",
                                   headers=_headers(a))
        assert missing.status_code == 404
        assert missing.json()["detail"] == {"reason": "media_not_found"}
        other_tenant = await client.get(f"/api/sessions/claude_code/{sid}/media/{_sha(PNG)}",
                                        headers=_headers(_b))
        assert other_tenant.status_code == 404
    async with httpx.AsyncClient() as plain:
        fetched = await plain.get(read["url"])
    assert fetched.status_code == 200 and fetched.content == PNG
    assert fetched.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_a_protocol_3_version_2_session_promotes_and_folds_its_images(media) -> None:
    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    events = [_image_event(0, [(_sha(PNG), "image/png")]), _reply(1)]
    await accept_all(a, _batches(sid, events, protocol=3, version=2))
    await _mine(a, CC, sid)
    bucket = await store.bucket_for(a)
    assert await store.exists(bucket, _final(a, sid, _sha(PNG)))
    trajectory = orjson.loads(await store.get(bucket, trajectory_key(CC.value, a, sid)))
    assert trajectory["steps"][0]["message"][1] == {
        "type": "image",
        "source": {"media_type": "image/png", "path": f"media/{_sha(PNG)}.png"},
    }
    # The door scrubbed the batch before storing it, and the digest survived.
    [stored] = [k for k in await objects(store, a, f"raw/{CC.value}/{a}/sessions-v2/{sid}/")
                if k.split("/")[-1].startswith("0-")]
    payload = json.loads(await store.get(bucket, stored))["payload"]
    assert payload["fragments"][0]["parts"][1]["sha256"] == _sha(PNG)


@pytest.mark.asyncio
async def test_a_version_1_session_names_no_image_and_promotes_nothing(media) -> None:
    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    events = [_image_event(0, [(_sha(PNG), "image/png")]), _reply(1)]
    await accept_all(a, _batches(sid, events, protocol=3, version=1))
    await _mine(a, CC, sid)
    assert not await store.exists(await store.bucket_for(a), _final(a, sid, _sha(PNG)))
    assert (await row(a, sid, _sha(PNG)))["promoted_at"] is None


@pytest.mark.asyncio
async def test_an_image_named_before_its_bytes_arrive_is_promoted_on_a_later_pass(media) -> None:
    """A missing staged object is not an error: the image keeps its
    placeholder and the next pass of the session promotes it."""
    a, _b, store = media
    sid = _sid()
    answer = (await ask(a, sid)).json()
    events = [_image_event(0, [(_sha(PNG), "image/png")]), _reply(1)]
    batches = _batches(sid, events)
    await accept_all(a, batches[:1])
    await _mine(a, CC, sid)  # no bytes yet: nothing raised, nothing promoted
    assert (await row(a, sid, _sha(PNG)))["promoted_at"] is None
    assert (await put(answer, PNG)).status_code == 200
    await accept_all(a, batches[1:])
    await _mine(a, CC, sid)
    assert await store.get(await store.bucket_for(a), _final(a, sid, _sha(PNG))) == PNG


@pytest.mark.asyncio
async def test_a_staged_object_that_is_not_what_was_signed_is_never_promoted(media) -> None:
    a, _b, store = media
    sid = _sid()
    assert (await ask(a, sid)).status_code == 200
    bucket = await store.bucket_for(a)
    # Written past the URL (a store that ignored the signed length).
    await store.put(bucket, _staged(a, sid, _sha(PNG)), PNG + b"tail", content_type="image/png")
    outcome = await sm.promote_named_media(a, CC.value, sid, [_sha(PNG)])
    assert outcome == {sm.Promotion.MISMATCH: 1}
    assert not await store.exists(bucket, _final(a, sid, _sha(PNG)))


@pytest.mark.asyncio
async def test_an_image_another_session_uploaded_is_not_this_sessions(media) -> None:
    a, _b, store = media
    mine, other = _sid(), _sid()
    await upload(a, other, PNG)
    # `mine` names the digest, but the upload (row and bytes) is `other`'s.
    assert await sm.promote_named_media(a, CC.value, mine, [_sha(PNG)]) == {}
    assert await objects(store, a) == [_staged(a, other, _sha(PNG))]


# --- deletion ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promotion_is_refused_for_a_deleted_session(media) -> None:
    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    await sd.record_sessions(a, [sd.SessionRef(CC.value, sid)], deletion_id=str(uuid.uuid4()),
                             reason="r", ticket=None, selector={"by": "id"})
    outcome = await sm.promote_named_media(a, CC.value, sid, [_sha(PNG)])
    assert outcome == {sm.Promotion.DELETED: 1}
    assert not await store.exists(await store.bucket_for(a), _final(a, sid, _sha(PNG)))


@pytest.mark.asyncio
async def test_deletion_removes_promoted_and_staged_images_and_their_rows(media) -> None:
    a, _b, store = media
    sid, keep = _sid(), _sid()
    later = PNG + b"not named yet"
    await upload(a, sid, PNG)
    await accept_all(a, _batches(sid, [_image_event(0, [(_sha(PNG), "image/png")]), _reply(1)]))
    await _mine(a, CC, sid)
    await upload(a, sid, later)  # staged, never named
    await upload(a, keep, PNG)   # another session's copy of the same image
    bucket = await store.bucket_for(a)
    assert await store.exists(bucket, _final(a, sid, _sha(PNG)))

    report = await sd.plan(a, [sd.SessionRef(CC.value, sid)])
    [planned] = report["sessions"]
    assert planned["rows"]["session_media"] == 2
    listed = await sd.SessionRef(CC.value, sid).raw_keys(store, bucket, a)
    assert {_final(a, sid, _sha(PNG)), _staged(a, sid, _sha(later))} <= listed

    outcome = (await delete(a, [sid]))["sessions"][f"claude_code:{sid}"]
    assert outcome["verified"], outcome
    assert outcome["rows_deleted"]["session_media"] == 2
    assert not [k for k in await objects(store, a) if sid in k]
    assert await objects(store, a, media_staging_prefix(CC.value, a, keep)) == [
        _staged(a, keep, _sha(PNG))
    ], "another session's staged image is untouched"
    assert await row(a, keep, _sha(PNG)) is not None
    # Reading it now: 410, as every route answers for a deleted session.
    async with _client() as client:
        gone = await client.get(f"/api/sessions/claude_code/{sid}/media/{_sha(PNG)}",
                                headers=_headers(a))
    assert gone.status_code == 410 and gone.json()["detail"]["reason"] == "session_deleted"


@pytest.mark.asyncio
async def test_a_late_upload_after_deletion_is_never_promoted_and_is_swept(media) -> None:
    """A URL handed out before the deletion still works for 15 minutes. Its
    bytes land in staging, which nothing promotes for a deleted session, and
    the sweep removes them (found by listing: the deletion removed the row)."""
    a, _b, store = media
    sid = _sid()
    answer = (await ask(a, sid)).json()
    outcome = (await delete(a, [sid]))["sessions"][f"claude_code:{sid}"]
    assert outcome["verified"], outcome
    assert (await put(answer, PNG)).status_code == 200  # the late PUT
    bucket = await store.bucket_for(a)
    assert await store.exists(bucket, _staged(a, sid, _sha(PNG)))
    # Nothing can name it: the door refuses the session, promotion finds no row.
    with pytest.raises(Exception) as refused:
        await accept_all(a, _batches(sid, [_image_event(0, [(_sha(PNG), "image/png")])]))
    assert getattr(refused.value, "status_code", None) == 410
    assert await sm.promote_named_media(a, CC.value, sid, [_sha(PNG)]) == {}
    assert not await store.exists(bucket, _final(a, sid, _sha(PNG)))
    # The next hourly sweep takes it: its session's deletion is recent, so it
    # does not wait out the 24-hour staging window.
    swept = await sm.sweep_media_staging()
    assert swept["objects_deleted"] == 1
    assert not await store.exists(bucket, _staged(a, sid, _sha(PNG)))


@pytest.mark.asyncio
async def test_the_worker_sweeps_staged_images_of_a_session_deleted_mid_pass(media) -> None:
    from engine.ingest.worker import Worker
    from engine.shared.session_suppression import SessionDeleted

    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    await Worker._sweep_deleted(
        SessionDeleted("deleted", customer_id=a, source=CC.value, session_ids={sid})
    )
    assert await objects(store, a, media_staging_prefix(CC.value, a, sid)) == []


def test_a_session_id_cannot_name_the_staging_folder() -> None:
    assert not sd.valid_session_id("media-staging")


# --- the sweep ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sweep_takes_old_staging_and_unpromoted_rows_and_leaves_held_tenants(
    media,
) -> None:
    a, b, store = media
    s1, s2 = _sid(), _sid()
    await upload(a, s1, PNG)
    await upload(a, s2, PNG + b"x")
    await accept_all(a, _batches(s2, [_image_event(0, [(_sha(PNG + b"x"), "image/png")])]))
    await _mine(a, CC, s2)  # promoted: its row stays
    # Tenant B, under legal hold, keeps everything.
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE customers SET metadata = jsonb_build_object('legal_hold', 'case 7') "
            "WHERE customer_id = $1", b,
        )
    b_bucket = await store.bucket_for(b)
    held = _staged(b, _sid(), _sha(PNG))
    await store.put(b_bucket, held, PNG, content_type="image/png")
    async with db_module.with_tenant(a) as conn:
        await conn.execute("UPDATE session_media SET created_at = now() - interval '25 hours'")

    dry = await sm.sweep_media_staging(older_than_s=0, dry_run=True)
    assert (dry["objects"], dry["rows"], dry.get("objects_deleted", 0)) == (1, 1, 0)
    assert dry["customers_held"] == 1
    swept = await sm.sweep_media_staging(older_than_s=0)
    assert (swept["objects_deleted"], swept["rows"]) == (1, 1)
    assert await row(a, s1, _sha(PNG)) is None, "an image never promoted is forgotten"
    assert (await row(a, s2, _sha(PNG + b"x")))["promoted_at"] is not None
    assert await store.exists(await store.bucket_for(a), _final(a, s2, _sha(PNG + b"x")))
    assert await store.exists(b_bucket, held)


@pytest.mark.asyncio
async def test_the_completer_cron_runs_the_sweep(media, capsys, monkeypatch) -> None:
    """The hourly session-completer CronJob is the sweep's schedule."""
    from scripts import cron_session_completer

    async def keep_the_tests_pool() -> None:
        return None

    monkeypatch.setattr(db_module, "close_pool", keep_the_tests_pool)
    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    await cron_session_completer._main(["--idle-minutes", "1440", "--dry-run"])
    out = capsys.readouterr().out
    assert "media staging sweep:" in out and "dry_run=True" in out
    assert await objects(store, a) == [_staged(a, sid, _sha(PNG))], "a dry run deletes nothing"


# --- owed promotions (named, not promoted) ---------------------------------------------------


@pytest.mark.asyncio
async def test_an_accepted_batch_stamps_its_images_owed_in_the_same_commit(media) -> None:
    a, _b, _store = media
    p2, p3, early = _sid(), _sid(), _sid()
    await upload(a, p2, PNG)
    await upload(a, p3, PNG)
    events = [_image_event(0, [(_sha(PNG), "image/png")]), _reply(1)]
    await accept_all(a, _batches(p2, events)[:1])
    await accept_all(a, _batches(p3, events, protocol=3, version=2)[:1])
    for sid in (p2, p3):
        record = await row(a, sid, _sha(PNG))
        assert record["named_at"] is not None and record["promoted_at"] is None, sid
    # Named before its upload was asked for: the worker's pass stamps it.
    await accept_all(a, _batches(early, events)[:1])
    await upload(a, early, PNG)
    assert (await row(a, early, _sha(PNG)))["named_at"] is None
    await _mine(a, CC, early)
    record = await row(a, early, _sha(PNG))
    assert record["named_at"] is not None and record["promoted_at"] is not None


@pytest.mark.asyncio
async def test_a_promotion_the_final_pass_failed_is_retried_by_the_sweep(media, monkeypatch) -> None:
    """The session's last pass cannot promote (the store fails), logs it and
    completes; nothing else would come back to it. The hourly sweep does."""
    from engine.shared.exceptions import StorageUnavailable

    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    await accept_all(a, _batches(sid, [_image_event(0, [(_sha(PNG), "image/png")]), _reply(1)]))

    async def store_down(*args, **kwargs):
        raise StorageUnavailable("copy failed: 503")

    with monkeypatch.context() as patch:
        patch.setattr(store, "copy", store_down)
        await _mine(a, CC, sid)  # the completing pass: no exception reaches it
    bucket = await store.bucket_for(a)
    assert not await store.exists(bucket, _final(a, sid, _sha(PNG)))
    assert (await row(a, sid, _sha(PNG)))["promoted_at"] is None
    # A day later (older_than_s=0): the retry promotes before anything is swept.
    swept = await sm.sweep_media_staging(older_than_s=0)
    assert swept["retry_promoted"] == 1 and swept["owed"] == 0
    assert await store.get(bucket, _final(a, sid, _sha(PNG))) == PNG
    assert (await row(a, sid, _sha(PNG)))["promoted_at"] is not None
    assert await objects(store, a, media_staging_prefix(CC.value, a, sid)) == []


@pytest.mark.asyncio
async def test_an_owed_image_keeps_its_staged_bytes_for_7_days_then_is_given_up(
    media, monkeypatch
) -> None:
    from engine.shared.exceptions import StorageUnavailable

    a, _b, store = media
    sid, unnamed = _sid(), _sid()
    await upload(a, sid, PNG)
    await upload(a, unnamed, PNG)
    await accept_all(a, _batches(sid, [_image_event(0, [(_sha(PNG), "image/png")])])[:1])

    async def store_down(*args, **kwargs):
        raise StorageUnavailable("copy failed: 503")

    monkeypatch.setattr(store, "copy", store_down)
    bucket = await store.bucket_for(a)
    # Past 24 hours, still failing: the owed image's bytes and row are kept;
    # the one no batch named goes as before.
    swept = await sm.sweep_media_staging(older_than_s=0)
    assert (swept["retry_failed"], swept["owed"], swept["objects_deleted"]) == (1, 1, 1)
    assert await store.exists(bucket, _staged(a, sid, _sha(PNG)))
    assert not await store.exists(bucket, _staged(a, unnamed, _sha(PNG)))
    assert await row(a, sid, _sha(PNG)) is not None
    # Named 8 days ago: given up, bytes and row removed.
    async with db_module.with_tenant(a) as conn:
        await conn.execute(
            "UPDATE session_media SET named_at = now() - interval '8 days', "
            "created_at = now() - interval '8 days'"
        )
    swept = await sm.sweep_media_staging(older_than_s=0)
    assert (swept["owed"], swept["abandoned"], swept["objects_deleted"]) == (0, 1, 1)
    assert not await store.exists(bucket, _staged(a, sid, _sha(PNG)))
    assert await row(a, sid, _sha(PNG)) is None


# --- the copy and the session lock -------------------------------------------------------------


class _HeldCopy:
    """Holds the store's server-side copy inside its worker thread until
    released: the shape of a slow copy."""

    def __init__(self, store, monkeypatch) -> None:
        self.entered, self.release = threading.Event(), threading.Event()
        real = store._client.copy

        def held(*args, **kwargs):
            self.entered.set()
            assert self.release.wait(30), "the test never released the copy"
            return real(*args, **kwargs)

        monkeypatch.setattr(store._client, "copy", held)

    async def wait_entered(self) -> None:
        for _ in range(500):
            if self.entered.is_set():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the copy never started")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_the_pass", [False, True], ids=["finishes", "cancelled"])
async def test_a_deletion_never_finishes_before_a_running_copy_lands(
    media, monkeypatch, cancel_the_pass: bool
) -> None:
    """The copy runs in a thread. If the promotion's await were cancelled (a
    worker shutting down) and gave up its lock before that thread ended, a
    deletion could list the session's folder, find nothing and finish, and the
    copy would then land `media/<sha>.<ext>` for good. The lock is held until
    the copy has landed, so the deletion always sees it and removes it."""
    a, _b, store = media
    sid = _sid()
    await upload(a, sid, PNG)
    held = _HeldCopy(store, monkeypatch)
    promotion = asyncio.create_task(sm.promote_named_media(a, CC.value, sid, [_sha(PNG)]))
    try:
        await held.wait_entered()
        if cancel_the_pass:
            promotion.cancel()
        deletion = asyncio.create_task(delete(a, [sid]))
        done, _ = await asyncio.wait({deletion}, timeout=1.0)
        assert not done, "the deletion waited for the session lock the copy holds"
    finally:
        held.release.set()
    if cancel_the_pass:
        with pytest.raises(asyncio.CancelledError):
            await promotion
    else:
        assert await promotion == {sm.Promotion.PROMOTED: 1}
    outcome = (await deletion)["sessions"][f"claude_code:{sid}"]
    assert outcome["verified"], outcome
    assert not [k for k in await objects(store, a) if sid in k], "nothing of the session remains"


@pytest.mark.asyncio
async def test_an_image_object_landing_after_a_deletion_is_swept(media) -> None:
    """A copy the store finished after its client timed out can land after the
    deletion listed the folder. The next sweep removes it."""
    a, _b, store = media
    sid, kept = _sid(), _sid()
    await upload(a, kept, PNG)
    await accept_all(a, _batches(kept, [_image_event(0, [(_sha(PNG), "image/png")])]))
    await _mine(a, CC, kept)
    outcome = (await delete(a, [sid]))["sessions"][f"claude_code:{sid}"]
    assert outcome["verified"], outcome
    bucket = await store.bucket_for(a)
    late = _final(a, sid, _sha(PNG))
    await store.put(bucket, late, PNG, content_type="image/png")
    swept = await sm.sweep_media_staging()
    assert swept["objects_deleted"] == 1
    assert not await store.exists(bucket, late)
    assert await store.exists(bucket, _final(a, kept, _sha(PNG))), "a live session's image stays"

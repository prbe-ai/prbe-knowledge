"""Immutable session acceptance on the existing webhook and queue.

The session advisory lock covers validation, the content-addressed R2 write,
receipt and queue transaction. A crash before commit leaves only an unreferenced
object; a crash after commit replays its receipt without rewriting any object.
Protocol 1 cannot write into a protocol-2 stream, even with a different device.

Protocol 3 is protocol 2 with each event sent as an ATIF fragment
(`fragments`, keyed by the same event ordinals) and the sanitized `events`
optional. A stream is pinned to the protocol of its first batch, and a new
stream may start on protocol 3 only while this customer is advertised it
(`accepts`); an open protocol-3 stream stays accepted after that is withdrawn.
"""

from __future__ import annotations

import hashlib
import json
import re
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from engine.ingest.connectedness import is_source_connected
from engine.ingest.payload_redaction import redact_payload_async
from engine.shared.config import Settings, get_settings
from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from engine.shared.session_suppression import (
    DELETED_REASON,
    DELETED_STATUS,
    deleted_sessions,
    lock_session,
)
from engine.shared.source_registry import ingestion_priority_for
from kb.admin_routes import verify_internal_knowledge_key

router = APIRouter(prefix="/api/sessions", dependencies=[Depends(verify_internal_knowledge_key)])
EMPTY_HASH = hashlib.sha256(b"").hexdigest()
_SOURCES = {"claude_code", "codex", "pi", "kimi_code"}
#: Session upload protocols: 2 sends sanitized probe-events/1 `events`; 3 sends
#: one ATIF `fragments` entry per event (and `events` only when asked to).
PROTOCOL_EVENTS = 2
PROTOCOL_FRAGMENTS = 3
SESSION_PROTOCOLS = (PROTOCOL_EVENTS, PROTOCOL_FRAGMENTS)


def fragment_ordinal(fragment: object) -> int | None:
    """The event ordinal a fragment covers: its Line's `line_no`
    (engine/ingest/atif/fragment.py). This door checks only that the ordinals
    are contiguous; `fold` validates everything else."""
    line = fragment.get("line") if isinstance(fragment, dict) else None
    ordinal = line.get("line_no") if isinstance(line, dict) else None
    return ordinal if type(ordinal) is int else None

_FIELDS = (
    "session_id",
    "batch_seq",
    "cwd",
    "events",
    "fragments",
    "fragment_version",
    "finalize",
    "protocol_version",
    "stream_id",
    "source_byte_start",
    "source_byte_end",
    "source_line_start",
    "source_line_end",
    "event_start",
    "event_end",
    "prefix_sha256",
    "provenance",
    "snapshot_byte_end",
    "snapshot_sha256",
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def fragment_versions(settings: Settings | None = None) -> list[int]:
    """Fragment versions this engine's fold reads. A typo in the deploy drops
    that entry rather than taking ingestion down."""
    s = settings or get_settings()
    return sorted(
        {int(v) for v in _csv(s.session_fragment_versions) if v.isascii() and v.isdigit()}
    )


def protocol3_enabled(customer_id: str, settings: Settings | None = None) -> bool:
    """May this customer START a protocol-3 stream now?"""
    s = settings or get_settings()
    return s.session_protocol3_all or customer_id in _csv(s.session_protocol3_customers)


def accepts(customer_id: str, settings: Settings | None = None) -> dict:
    """What a client may start a NEW stream with. An existing stream keeps the
    protocol it was pinned to, whatever this says."""
    s = settings or get_settings()
    protocols = [PROTOCOL_EVENTS]
    if protocol3_enabled(customer_id, s):
        protocols.append(PROTOCOL_FRAGMENTS)
    return {
        "protocols": protocols,
        "fragment_versions": fragment_versions(s),
        "events": s.session_protocol3_events,
    }


def payload_protocol(payload: dict) -> int:
    """The protocol of a payload `validate_payload` accepted."""
    return (
        PROTOCOL_FRAGMENTS if payload["protocol_version"] == PROTOCOL_FRAGMENTS else PROTOCOL_EVENTS
    )


def canonical_payload(payload: dict) -> bytes:
    """Identity enrichment and request timestamps are not source event identity."""
    return json.dumps(
        {k: payload[k] for k in _FIELDS if k in payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def _check_ordinals(items: list, key: str, start: int, end: int, what: str) -> None:
    if [item.get(key) if isinstance(item, dict) else None for item in items] != list(
        range(start, end)
    ):
        raise ValueError(f"{what} ordinals are not contiguous")


def _validate_fragments(payload: dict) -> None:
    """Protocol 3's batch body: one fragment per covered event ordinal.

    Only the envelope is checked here. The fragment's own fields are the
    client's word and `fold` treats them as untrusted.
    """
    start, end = payload["event_start"], payload["event_end"]
    fragments = payload.get("fragments", [])
    if not isinstance(fragments, list) or end - start != len(fragments):
        raise ValueError("fragment coverage does not match payload")
    if payload.get("finalize") and fragments:
        raise ValueError("finalize cannot carry fragments")
    ordinals = [fragment_ordinal(f) for f in fragments]
    if None in ordinals:
        raise ValueError("fragment is not an object with an integer ordinal")
    if ordinals != list(range(start, end)):
        raise ValueError("fragment ordinals are not contiguous")
    if fragments and "fragment_version" not in payload:
        raise ValueError("fragments carry no fragment_version")
    if "fragment_version" in payload:
        version = payload["fragment_version"]
        if type(version) is not int or version not in fragment_versions():
            raise ValueError("unsupported fragment version")
    # Optional in protocol 3 (the canary also sends them), and when present they
    # cover exactly the fragments' ordinals.
    if "events" in payload:
        events = payload["events"]
        if not isinstance(events, list) or len(events) != len(fragments):
            raise ValueError("event coverage does not match payload")
        if payload.get("finalize") and events:
            raise ValueError("finalize cannot carry events")
        _check_ordinals(events, "line_no", start, end, "retained-event")


def validate_payload(payload: dict) -> None:
    try:
        UUID(payload["session_id"])
        UUID(payload["stream_id"])
        protocol = payload["protocol_version"]
        if protocol == PROTOCOL_FRAGMENTS and type(protocol) is int:
            sends_fragments = True
        elif protocol == PROTOCOL_EVENTS:
            sends_fragments = False
        else:
            raise ValueError("unsupported codec")
        for name in (
            "batch_seq",
            "source_byte_start",
            "source_byte_end",
            "source_line_start",
            "source_line_end",
            "event_start",
            "event_end",
        ):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError(f"invalid {name}")
        for unit in ("source_byte", "source_line", "event"):
            if payload[f"{unit}_end"] < payload[f"{unit}_start"]:
                raise ValueError(f"invalid {unit} interval")
        if not re.fullmatch(r"[a-f0-9]{64}", payload["prefix_sha256"]):
            raise ValueError("invalid prefix digest")
        snapshot_end = payload.get("snapshot_byte_end")
        if snapshot_end is not None or payload.get("snapshot_sha256") is not None:
            if type(snapshot_end) is not int or snapshot_end < payload["source_byte_end"]:
                raise ValueError("invalid historical snapshot boundary")
            if not re.fullmatch(r"[a-f0-9]{64}", payload.get("snapshot_sha256", "")):
                raise ValueError("invalid historical snapshot digest")
        if sends_fragments:
            _validate_fragments(payload)
        else:
            events = payload.get("events", [])
            if not isinstance(events, list) or payload["event_end"] - payload["event_start"] != len(
                events
            ):
                raise ValueError("event coverage does not match payload")
            if payload.get("finalize") and events:
                raise ValueError("finalize cannot carry events")
        if payload.get("finalize") and any(
            payload[f"{unit}_start"] != payload[f"{unit}_end"]
            for unit in ("source_byte", "source_line", "event")
        ):
            raise ValueError("finalize must certify the already accepted cursor")
        if not sends_fragments:
            _check_ordinals(
                events, "line_no", payload["event_start"], payload["event_end"], "retained-event"
            )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(422, f"invalid transcript protocol: {exc}") from exc


async def _lock(conn, customer: str, source: str, session: str) -> None:
    await lock_session(conn, customer, source, session)


async def refuse_deleted(conn, customer: str, source: str, session: str) -> None:
    """410 for a session a customer had deleted. Caller holds the session lock.

    Checked before any byte is written: a capture client keeps a session's
    transcript on disk and re-sends it (retries, reconnects, a fresh stream from
    batch 0), and accepting it would restore exactly what was erased.
    """
    if await deleted_sessions(conn, customer, source, [session]):
        raise HTTPException(
            DELETED_STATUS,
            {
                "reason": DELETED_REASON,
                "message": "this session was deleted at the customer's request; "
                "do not resend it",
                "source": source,
                "session_id": session,
            },
        )


async def _legacy_exists(conn, customer: str, source: str, session: str) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM ingestion_queue WHERE customer_id=$1 "
            "AND source_system=$2 AND source_event_id=$3) OR EXISTS(SELECT 1 FROM documents "
            "WHERE customer_id=$1 AND doc_id=$4)",
            customer,
            source,
            session,
            f"{source}:{customer}:{session}",
        )
    )


async def reject_legacy_writer(conn, customer: str, source: str, session: str) -> None:
    if await conn.fetchval(
        "SELECT 1 FROM session_streams WHERE customer_id=$1 AND source_system=$2 AND session_id=$3",
        customer,
        source,
        session,
    ):
        raise HTTPException(409, "session uses protocol 2; upgrade the capture producer")


async def accept_legacy(payload, envelope, customer, source, store, key, enqueue) -> bool:
    """Fence old producers atomically with first protocol-2 stream acceptance."""
    if not await is_source_connected(customer, source):
        return False
    # Authentication/parse happened upstream. Never persist the original
    # envelope, even for an old capture client that did not redact locally.
    envelope = json.dumps(await redact_payload_async(json.loads(envelope))).encode()
    async with with_tenant(customer) as conn:
        await _lock(conn, customer, source.value, payload["session_id"])
        await refuse_deleted(conn, customer, source.value, payload["session_id"])
        await reject_legacy_writer(conn, customer, source.value, payload["session_id"])
        bucket = await store.bucket_for(customer)
        await store.ensure_bucket(bucket)
        await store.put(bucket, key, envelope)
        await _enqueue_agent(conn, customer, source, payload["session_id"], key)
        return True


async def _enqueue_agent(conn, customer, source, sid, key):
    await conn.execute(
        "INSERT INTO ingestion_queue(customer_id,source_system,source_event_id,payload_s3_key,"
        "payload_s3_keys,status,priority,version,enqueued_at) VALUES($1,$2,$3,$4,ARRAY[$4],'pending',$5,1,now()) "
        "ON CONFLICT(customer_id,source_system,source_event_id) DO UPDATE SET "
        "payload_s3_keys=ingestion_queue.payload_s3_keys || EXCLUDED.payload_s3_keys,"
        # A row being processed stays `processing`: the version bump is what
        # tells the running worker its payload grew, and its CAS miss returns
        # the row to pending (Worker._mark_done). Resetting it here let a
        # second worker claim the same session mid-extraction and mine it
        # twice -- the same fix kb/ingestion_app._enqueue already carries.
        "status=CASE WHEN ingestion_queue.status='processing' THEN 'processing' ELSE 'pending' END,"
        "version=ingestion_queue.version+1,completed_at=NULL,error=NULL,enqueued_at=now()",
        customer,
        source.value,
        sid,
        key,
        ingestion_priority_for(source.value),
    )


def _receipt(row) -> dict:
    return {
        k: row[k]
        for k in (
            "batch_seq",
            "body_sha256",
            "source_byte_end",
            "source_line_end",
            "event_end",
            "prefix_sha256",
            "finalized",
        )
    }


async def accept(payload: dict, customer: str, source: SourceSystem, store) -> dict:
    validate_payload(payload)
    protocol = payload_protocol(payload)
    if not await is_source_connected(customer, source):
        raise HTTPException(409, "session capture source is disconnected")
    sid = payload["session_id"]
    digest = hashlib.sha256(canonical_payload(payload)).hexdigest()
    # The receipt identifies the original request, which clients retry byte
    # for byte. Only the stored representation changes, not that identity.
    stored_payload = await redact_payload_async(payload)
    # Date-independent and content-addressed: no other body can overwrite this key.
    key = f"raw/{source.value}/{customer}/sessions-v2/{sid}/{payload['batch_seq']}-{digest}.json"
    async with with_tenant(customer) as conn:
        await _lock(conn, customer, source.value, sid)
        await refuse_deleted(conn, customer, source.value, sid)
        stream = await conn.fetchrow(
            "SELECT * FROM session_streams WHERE customer_id=$1 "
            "AND source_system=$2 AND session_id=$3",
            customer,
            source.value,
            sid,
        )
        if stream is None:
            if await _legacy_exists(conn, customer, source.value, sid):
                raise HTTPException(
                    409, "legacy transcript requires reconciliation; no bytes replaced"
                )
            if payload["batch_seq"] != 0:
                raise HTTPException(409, "unknown stream; first sequence must be zero")
            # Only a NEW stream asks: one already pinned to 3 keeps being
            # accepted after the customer stops being advertised it.
            if protocol == PROTOCOL_FRAGMENTS and not protocol3_enabled(customer):
                raise HTTPException(409, "protocol 3 not enabled")
            await conn.execute(
                "INSERT INTO session_streams(customer_id,source_system,session_id,stream_id,"
                "protocol_version,prefix_sha256,uploader_device_id) VALUES($1,$2,$3,$4,$5,$6,$7)",
                customer,
                source.value,
                sid,
                payload["stream_id"],
                protocol,
                EMPTY_HASH,
                stored_payload.get("device_id"),
            )
            stream = {
                "stream_id": payload["stream_id"],
                "protocol_version": protocol,
                "last_seq": -1,
                "source_byte_end": 0,
                "source_line_end": 0,
                "event_end": 0,
            }
        if stream["stream_id"] != payload["stream_id"]:
            raise HTTPException(409, "session owned by another stream; reconcile receipts first")
        if stream["protocol_version"] != protocol:
            raise HTTPException(409, "protocol mismatch")
        previous = await conn.fetchrow(
            "SELECT * FROM session_batch_receipts WHERE customer_id=$1 "
            "AND source_system=$2 AND session_id=$3 AND batch_seq=$4",
            customer,
            source.value,
            sid,
            payload["batch_seq"],
        )
        if previous:
            if previous["body_sha256"] != digest:
                raise HTTPException(409, "batch identity already accepted with different content")
            return {
                "status": "duplicate",
                "protocol_version": protocol,
                "receipt": _receipt(previous),
            }
        if payload["batch_seq"] != stream["last_seq"] + 1 or any(
            payload[f"{unit}_start"] != stream[f"{unit}_end"]
            for unit in ("source_byte", "source_line", "event")
        ):
            raise HTTPException(409, "batch does not continue the acknowledged source/event cursor")
        if payload.get("finalize") and payload["prefix_sha256"] != stream.get(
            "prefix_sha256", EMPTY_HASH
        ):
            raise HTTPException(409, "finalize prefix differs from the accepted source")
        snapshot_end = payload.get("snapshot_byte_end")
        snapshot_hash = payload.get("snapshot_sha256")
        if (
            stream.get("snapshot_byte_end") is not None
            and not stream.get("finalized")
            and (snapshot_end, snapshot_hash)
            != (
                stream["snapshot_byte_end"],
                stream["snapshot_sha256"],
            )
        ):
            raise HTTPException(409, "historical snapshot changed before completion")
        if snapshot_end is not None:
            if payload.get("finalize") and payload["source_byte_end"] != snapshot_end:
                raise HTTPException(409, "historical snapshot is not fully accepted")
            if (
                payload["source_byte_end"] == snapshot_end
                and payload["prefix_sha256"] != snapshot_hash
            ):
                raise HTTPException(
                    409, "accepted source differs from the declared historical snapshot"
                )
        bucket = await store.bucket_for(customer)
        await store.ensure_bucket(bucket)
        # Include authenticated uploader metadata for display, never as original authorship proof.
        envelope = json.dumps({"payload": stored_payload}, sort_keys=True, separators=(",", ":")).encode()
        await store.put(bucket, key, envelope)
        await _enqueue_agent(conn, customer, source, sid, key)
        row = await conn.fetchrow(
            "INSERT INTO session_batch_receipts(customer_id,source_system,session_id,batch_seq,body_sha256,"
            "payload_key,source_byte_end,source_line_end,event_end,prefix_sha256,finalized) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING *",
            customer,
            source.value,
            sid,
            payload["batch_seq"],
            digest,
            key,
            payload["source_byte_end"],
            payload["source_line_end"],
            payload["event_end"],
            payload["prefix_sha256"],
            bool(payload.get("finalize")),
        )
        await conn.execute(
            "UPDATE session_streams SET last_seq=$4,source_byte_end=$5,source_line_end=$6,"
            "event_end=$7,prefix_sha256=$8,finalized=$9,snapshot_byte_end=$10,snapshot_sha256=$11 "
            "WHERE customer_id=$1 AND source_system=$2 AND session_id=$3",
            customer,
            source.value,
            sid,
            payload["batch_seq"],
            payload["source_byte_end"],
            payload["source_line_end"],
            payload["event_end"],
            payload["prefix_sha256"],
            bool(payload.get("finalize")),
            snapshot_end,
            snapshot_hash,
        )
        return {"status": "accepted", "protocol_version": protocol, "receipt": _receipt(row)}


@router.get("/{source}/{session_id}/receipts")
async def receipts(
    source: str,
    session_id: str,
    x_prbe_customer: str = Header(...),
    after: int = Query(-1, ge=-1),
    limit: int = Query(200, ge=1, le=1000),
) -> dict:
    if source not in _SOURCES:
        raise HTTPException(422, "unsupported transcript source")
    try:
        UUID(session_id)
    except ValueError as exc:
        raise HTTPException(422, "malformed session id") from exc
    async with with_tenant(x_prbe_customer) as conn:
        stream = await conn.fetchrow(
            "SELECT * FROM session_streams WHERE customer_id=$1 "
            "AND source_system=$2 AND session_id=$3",
            x_prbe_customer,
            source,
            session_id,
        )
        # Asked first, not only when no stream is left: between the deletion
        # being recorded and its rows going, the stream still exists, and
        # `ready` would invite the client to keep sending. Said outright rather
        # than "absent": an absent session is one a client should start
        # uploading, and this one must never be.
        deleted = bool(await deleted_sessions(conn, x_prbe_customer, source, [session_id]))
        if deleted or not stream:
            if deleted:
                state = "deleted"
            elif await _legacy_exists(conn, x_prbe_customer, source, session_id):
                state = "legacy"
            else:
                state = "absent"
            return {
                # Old clients refuse anything but 2 here; `accepts` is what a
                # new stream may start on.
                "protocol_version": PROTOCOL_EVENTS,
                "accepts": accepts(x_prbe_customer),
                "customer_id": x_prbe_customer,
                "source": source,
                "session_id": session_id,
                "state": state,
                "receipts": [],
            }
        rows = await conn.fetch(
            "SELECT * FROM session_batch_receipts WHERE customer_id=$1 AND source_system=$2 "
            "AND session_id=$3 AND batch_seq>$4 ORDER BY batch_seq LIMIT $5",
            x_prbe_customer,
            source,
            session_id,
            after,
            limit,
        )
        return {
            # The protocol this stream is pinned to, for its whole life.
            "protocol_version": stream["protocol_version"],
            "accepts": accepts(x_prbe_customer),
            "customer_id": x_prbe_customer,
            "source": source,
            "session_id": session_id,
            "state": "ready",
            "stream": {
                k: stream[k]
                for k in (
                    "stream_id",
                    "last_seq",
                    "source_byte_end",
                    "source_line_end",
                    "event_end",
                    "prefix_sha256",
                    "finalized",
                    "snapshot_byte_end",
                    "snapshot_sha256",
                )
            },
            "receipts": [_receipt(r) for r in rows],
            "more": bool(rows and rows[-1]["batch_seq"] < stream["last_seq"]),
        }

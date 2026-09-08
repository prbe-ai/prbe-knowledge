"""Immutable session acceptance on the existing webhook and queue.

The session advisory lock covers validation, the content-addressed R2 write,
receipt and queue transaction. A crash before commit leaves only an unreferenced
object; a crash after commit replays its receipt without rewriting any object.
Protocol 1 cannot write into a protocol-2 stream, even with a different device.
"""

from __future__ import annotations

import hashlib
import json
import re
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from engine.ingest.connectedness import is_source_connected
from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from engine.shared.source_registry import ingestion_priority_for
from kb.admin_routes import verify_internal_knowledge_key

router = APIRouter(prefix="/api/sessions", dependencies=[Depends(verify_internal_knowledge_key)])
EMPTY_HASH = hashlib.sha256(b"").hexdigest()
_SOURCES = {"claude_code", "codex", "pi"}
_FIELDS = (
    "session_id",
    "batch_seq",
    "cwd",
    "events",
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


def canonical_payload(payload: dict) -> bytes:
    """Identity enrichment and request timestamps are not source event identity."""
    return json.dumps(
        {k: payload[k] for k in _FIELDS if k in payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def validate_payload(payload: dict) -> None:
    try:
        UUID(payload["session_id"])
        UUID(payload["stream_id"])
        if payload["protocol_version"] != 2:
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
        if [e.get("line_no") if isinstance(e, dict) else None for e in events] != list(
            range(payload["event_start"], payload["event_end"])
        ):
            raise ValueError("retained-event ordinals are not contiguous")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(422, f"invalid transcript protocol: {exc}") from exc


async def _lock(conn, customer: str, source: str, session: str) -> None:
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"session-stream:{customer}:{source}:{session}",
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
    async with with_tenant(customer) as conn:
        await _lock(conn, customer, source.value, payload["session_id"])
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
        "payload_s3_keys=ingestion_queue.payload_s3_keys || EXCLUDED.payload_s3_keys, status='pending',"
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
    if not await is_source_connected(customer, source):
        raise HTTPException(409, "session capture source is disconnected")
    sid = payload["session_id"]
    digest = hashlib.sha256(canonical_payload(payload)).hexdigest()
    # Date-independent and content-addressed: no other body can overwrite this key.
    key = f"raw/{source.value}/{customer}/sessions-v2/{sid}/{payload['batch_seq']}-{digest}.json"
    async with with_tenant(customer) as conn:
        await _lock(conn, customer, source.value, sid)
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
            await conn.execute(
                "INSERT INTO session_streams(customer_id,source_system,session_id,stream_id,"
                "protocol_version,prefix_sha256,uploader_device_id) VALUES($1,$2,$3,$4,2,$5,$6)",
                customer,
                source.value,
                sid,
                payload["stream_id"],
                EMPTY_HASH,
                payload.get("device_id"),
            )
            stream = {
                "stream_id": payload["stream_id"],
                "last_seq": -1,
                "source_byte_end": 0,
                "source_line_end": 0,
                "event_end": 0,
            }
        if stream["stream_id"] != payload["stream_id"]:
            raise HTTPException(409, "session owned by another stream; reconcile receipts first")
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
            return {"status": "duplicate", "protocol_version": 2, "receipt": _receipt(previous)}
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
        envelope = json.dumps({"payload": payload}, sort_keys=True, separators=(",", ":")).encode()
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
        return {"status": "accepted", "protocol_version": 2, "receipt": _receipt(row)}


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
        if not stream:
            legacy = await _legacy_exists(conn, x_prbe_customer, source, session_id)
            return {
                "protocol_version": 2,
                "customer_id": x_prbe_customer,
                "source": source,
                "session_id": session_id,
                "state": "legacy" if legacy else "absent",
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
            "protocol_version": 2,
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

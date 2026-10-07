"""Strip stored coding-agent session batches down to probe-events/1.

Taps before 0.9.10 uploaded more than the consent screen allows: a second copy
of every tool's output (`toolUseResult`), files and CLAUDE.md attachments,
unknown content blocks whole, pasted images as base64, Codex's raw shell
`action`. Nothing reads those fields -- not the index, not the trajectory, not
the dashboard -- but they sit in R2 for as long as the customer stays. This
rewrites each stored batch with every event projected onto probe-events/1
(engine/ingest/probe_events): what tap 0.9.11 sends, nothing more.

Safe by construction, checked per batch:
  * Only ever removes keys (engine/ingest/probe_events/project.py).
  * A batch is rewritten only if its rendered Lines -- the indexed text, the
    evidence spans and the extraction input -- are identical before and after.
    Otherwise it is left alone and reported (`lines_changed`), so search,
    extraction caches and trajectories never move.
  * The receipt is not touched: it pins the hash of the client's original
    request (kb/session_receipts.py), never the stored representation, which
    the server already rewrites when it redacts.
  * A protocol-3 batch (ATIF `fragments`, kb/session_receipts.py) keeps its
    fragments exactly: they are allow-listed by construction (fragment.py) and
    the session is read from them. Its `events`, present only while a canary
    asks for them, are projected like any others -- and only if each still
    fragments to what it did, since they are the evidence the fragment shadow
    compares the client's fragments with (`fragments_changed` otherwise). One
    with no `events` reads `no_events`: nothing to strip.
  * `--drop-protocol3-events` instead REMOVES a protocol-3 batch's `events`
    (the canary's duplicate of its fragments, sent only while
    SESSION_PROTOCOL3_EVENTS was on): only for a batch DECLARED protocol 3, of
    an ENDED session, whose events fragment to byte-identical fragments
    (`fragments_differ` otherwise) and whose fragments are not degraded
    (`degraded`). What is lost is canary evidence only: the worker reads a
    protocol-3 session from its fragments. Protocol-2 batches are untouched.
  * Nothing is rewritten for a tenant under legal hold or not active, nor for a
    deleted session (engine/shared/legal_hold.purge_blocked_reason, checked per
    tenant and again per batch). A session deleted or purged while its batch
    was being rewritten has the rewritten object removed again: the strip
    never resurrects a sweep.

DRY RUN unless --write. Output is keys, counts, byte sizes and the key paths
dropped (a histogram per tenant), never content. Review it before --write.

    scripts/atif_sessions_job.sh is the runner (MODULE=scripts.strip_session_payloads):
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --customer probe
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --customer probe --write
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --all-tenants --write
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --all-tenants \
        --drop-protocol3-events --write
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from typing import Any

import orjson

from engine.ingest.atif.fragment import fragment
from engine.ingest.atif.uploaded import fragment_lines
from engine.ingest.probe_events.project import dropped_paths, project_event
from engine.shared import storage
from engine.shared.constants import AGENT_SESSION_SOURCES
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.exceptions import StorageNotFound
from engine.shared.legal_hold import purge_blocked_reason
from engine.shared.session_signals import SessionProtocol, is_cron_marker_key
from engine.shared.session_suppression import deleted_sessions, session_of_event_id
from engine.shared.transcript_render import lines_from_events
from scripts.atif_sessions import _emit, _queue_row, _queue_rows, _tenants

#: A tenant or session a background job must not rewrite (legal hold, not
#: active, missing) is skipped; one whose data is GONE (deleted, purged) also
#: has a copy we put back removed again.
_GONE_STATUSES = ("missing", "status:deleted")


def strip_batch(
    body: bytes, *, drop_fragment_events: bool = False
) -> tuple[bytes | None, str, int, list[str]]:
    """(new body or None, outcome, bytes removed, dropped key paths) for one batch."""
    try:
        envelope = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None, "unreadable", 0, []
    if not isinstance(envelope, dict):
        return None, "unreadable", 0, []
    payload = envelope.get("payload", envelope)
    if drop_fragment_events:
        return _drop_fragment_events(envelope, payload, body)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or not events:
        return None, "no_events", 0, []
    stripped = []
    dropped: list[str] = []
    for e in events:
        if isinstance(e, dict) and isinstance(e.get("raw"), dict):
            raw = project_event(e["raw"])
            dropped.extend(dropped_paths(e["raw"], raw))
            stripped.append({**e, "raw": raw})
        else:
            stripped.append(e)
    if stripped == events:
        return None, "clean", 0, []
    if lines_from_events(stripped) != lines_from_events(events):
        return None, "lines_changed", 0, dropped
    if "fragments" in payload and _fragments_of(stripped) != _fragments_of(events):
        return None, "fragments_changed", 0, dropped
    # Everything else in the payload -- a protocol-3 batch's fragments included
    # -- is kept as it is.
    new_payload = {**payload, "events": stripped}
    new_envelope = {**envelope, "payload": new_payload} if "payload" in envelope else new_payload
    new_body = json.dumps(new_envelope, sort_keys=True, separators=(",", ":")).encode()
    return new_body, "stripped", len(body) - len(new_body), dropped


def _drop_fragment_events(
    envelope: dict[str, Any], payload: Any, body: bytes
) -> tuple[bytes | None, str, int, list[str]]:
    """A protocol-3 batch without its canary `events`, when they add nothing.

    Protocol 3 is the batch's DECLARED protocol (the worker reads a protocol-2
    batch from its events whatever else it carries). The events go only when
    they fragment to byte-identical fragments (canonical JSON: `1` is not
    `true`) and no fragment is degraded -- a degraded line's events are the copy
    a fixed renderer could still re-render."""
    if not isinstance(payload, dict) or SessionProtocol.of(payload) is not SessionProtocol.FRAGMENTS:
        return None, "not_protocol3", 0, []
    fragments = payload.get("fragments")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return None, "no_events", 0, []
    if not isinstance(fragments, list) or fragment_lines(fragments).degraded:
        return None, "degraded", 0, []
    if _canonical(_fragments_of(events)) != _canonical(fragments):
        return None, "fragments_differ", 0, []
    new_payload = {k: v for k, v in payload.items() if k != "events"}
    new_envelope = {**envelope, "payload": new_payload} if "payload" in envelope else new_payload
    new_body = json.dumps(new_envelope, sort_keys=True, separators=(",", ":")).encode()
    return new_body, "events_dropped", len(body) - len(new_body), ["events"]


async def _stream_finalized(customer_id: str, source: str, session_id: str) -> bool:
    async with with_tenant(customer_id) as conn:
        return bool(
            await conn.fetchval(
                "SELECT finalized FROM session_streams WHERE customer_id=$1 "
                "AND source_system=$2 AND session_id=$3",
                customer_id,
                source,
                session_id,
            )
        )


def _canonical(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _fragments_of(events: list[Any]) -> list[dict[str, Any]]:
    return [fragment(e) for e in events if isinstance(e, dict)]


async def _blocked(customer_id: str, source: str, session_id: str) -> str | None:
    """Why this session must not be rewritten now (legal hold, tenant not
    active or missing, session deleted, queue row gone), or None."""
    async with get_pool().acquire() as conn:
        reason = await purge_blocked_reason(conn, customer_id)
    if reason is not None:
        return reason
    if await _queue_row(customer_id, source, session_id) is None:
        return "purged"
    async with with_tenant(customer_id) as conn:
        if await deleted_sessions(conn, customer_id, source, [session_id]):
            return "deleted"
    return None


async def _blocked_after_put(customer_id: str, source: str, session_id: str) -> str | None:
    """`_blocked` after a put, retried; "unverified" when it cannot be read, so
    the key is reported for an operator to recheck rather than silently kept."""
    for attempt in range(3):
        try:
            return await _blocked(customer_id, source, session_id)
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1))
    return "unverified"


async def strip(args: argparse.Namespace) -> None:
    store = storage.get_store()
    counts: dict[str, int] = defaultdict(int)
    paths: Counter[str] = Counter()
    removed = 0

    async def one_key(customer_id: str, bucket: str, row: dict[str, Any], key: str) -> None:
        nonlocal removed
        source, session_id = row["source_system"], row["source_event_id"]
        ident = {"customer": customer_id, "source": source, "session_id": session_id, "key": key}
        try:
            body = await store.get(bucket, key)
        except StorageNotFound:
            counts["missing"] += 1
            return
        new_body, outcome, saved, dropped = strip_batch(
            body, drop_fragment_events=args.drop_protocol3_events
        )
        if outcome == "events_dropped" and not await _stream_finalized(
            customer_id, source, session_id
        ):
            # A running session: its events may still be compared on its
            # completing pass. Left for a later run.
            new_body, outcome, saved, dropped = None, "skipped:running", 0, []
        counts[outcome] += 1
        paths.update(dropped)
        if outcome == "lines_changed":
            _emit({"kind": "batch", **ident, "lines_changed": True})
        if new_body is None:
            return
        removed += saved
        if not args.write:
            return
        if (reason := await _blocked(customer_id, source, session_id)) is not None:
            counts[f"skipped:{reason}"] += 1
            return
        put_ok = False
        try:
            await store.put(bucket, key, new_body)
            put_ok = True
        finally:
            # Whatever the put did (a timed-out put may still have landed): if
            # the session's data is gone now -- its deletion or purge swept while
            # we wrote -- remove what we put back. A legal hold or a terminated
            # tenant keeps the stripped copy: that state is reversible.
            after = await _blocked_after_put(customer_id, source, session_id)
            if after == "unverified":
                counts["unverified"] += 1
                _emit({"kind": "batch", **ident, "put_ok": put_ok, "post_put_unverified": True})
            elif after in ("deleted", "purged", *_GONE_STATUSES):
                await store.delete(bucket, key)
                counts["overtaken"] += 1
                _emit({"kind": "batch", **ident, "overtaken": after})
            elif put_ok:
                counts["written"] += 1

    async def drain(customer_id: str, bucket: str, work: asyncio.Queue) -> None:
        while True:
            item = await work.get()
            if item is None:
                return
            row, key = item
            try:
                await one_key(customer_id, bucket, row, key)
            except Exception as exc:
                counts[f"error:{type(exc).__name__}"] += 1
                _emit({"kind": "batch", "customer": customer_id, "key": key,
                       "error": type(exc).__name__})

    for customer_id in await _tenants(args.customer, args.all_tenants):
        async with get_pool().acquire() as conn:
            reason = await purge_blocked_reason(conn, customer_id)
        if reason is not None:
            _emit({"kind": "tenant", "customer": customer_id, "skipped": reason})
            continue
        bucket = await store.bucket_for(customer_id)
        before, before_removed, before_paths = dict(counts), removed, Counter(paths)
        all_rows = await _queue_rows(customer_id, args.sources)
        rows = [r for r in all_rows if session_of_event_id(r["source_event_id"]) == r["source_event_id"]]
        work: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
        workers = [asyncio.create_task(drain(customer_id, bucket, work)) for _ in range(args.concurrency)]
        for row in rows:
            for key in dict.fromkeys(k for k in (row["payload_s3_keys"] or []) if k):
                if not is_cron_marker_key(key):
                    await work.put((row, key))
        for _ in workers:
            await work.put(None)
        await asyncio.gather(*workers)
        tenant_paths = paths - before_paths
        _emit({
            "kind": "tenant", "customer": customer_id, "sessions": len(rows),
            "legacy_event_rows": len(all_rows) - len(rows),
            "bytes_removed": removed - before_removed,
            "dropped_paths": dict(tenant_paths.most_common(40)),
            **{k: counts[k] - before.get(k, 0) for k in counts if counts[k] - before.get(k, 0)},
        })
    _emit({"kind": "summary", "write": args.write, "bytes_removed": removed,
           "dropped_paths": dict(paths.most_common(60)), **dict(sorted(counts.items()))})


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("strip")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--customer", action="append")
    scope.add_argument("--all-tenants", action="store_true")
    p.add_argument("--sources", nargs="*", default=sorted(s.value for s in AGENT_SESSION_SOURCES))
    p.add_argument("--write", action="store_true")
    p.add_argument("--concurrency", type=_positive, default=8)
    p.add_argument("--drop-protocol3-events", action="store_true",
                   help="remove ended protocol-3 sessions' canary events (see module doc)")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from engine.shared.config import get_settings

    await init_pool(get_settings())
    try:
        await strip(args)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())

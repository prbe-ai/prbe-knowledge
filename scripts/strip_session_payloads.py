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
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from typing import Any

import orjson

from engine.ingest.probe_events.project import dropped_paths, project_event
from engine.shared import storage
from engine.shared.constants import AGENT_SESSION_SOURCES
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.exceptions import StorageNotFound
from engine.shared.legal_hold import purge_blocked_reason
from engine.shared.session_signals import is_cron_marker_key
from engine.shared.session_suppression import deleted_sessions, session_of_event_id
from engine.shared.transcript_render import lines_from_events
from scripts.atif_sessions import _emit, _queue_row, _queue_rows, _tenants

#: A tenant or session a background job must not rewrite (legal hold, not
#: active, missing) is skipped; one whose data is GONE (deleted, purged) also
#: has a copy we put back removed again.
_GONE_STATUSES = ("missing", "status:deleted")


def strip_batch(body: bytes) -> tuple[bytes | None, str, int, list[str]]:
    """(new body or None, outcome, bytes removed, dropped key paths) for one batch."""
    try:
        envelope = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None, "unreadable", 0, []
    if not isinstance(envelope, dict):
        return None, "unreadable", 0, []
    payload = envelope.get("payload", envelope)
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
    new_payload = {**payload, "events": stripped}
    new_envelope = {**envelope, "payload": new_payload} if "payload" in envelope else new_payload
    new_body = json.dumps(new_envelope, sort_keys=True, separators=(",", ":")).encode()
    return new_body, "stripped", len(body) - len(new_body), dropped


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
        new_body, outcome, saved, dropped = strip_batch(body)
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
            after = await _blocked(customer_id, source, session_id)
            if after in ("deleted", "purged", *_GONE_STATUSES):
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("strip")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--customer", action="append")
    scope.add_argument("--all-tenants", action="store_true")
    p.add_argument("--sources", nargs="*", default=sorted(s.value for s in AGENT_SESSION_SOURCES))
    p.add_argument("--write", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
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

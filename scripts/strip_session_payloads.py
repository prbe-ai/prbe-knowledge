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
  * A session deleted, purged or closed while its batch was being rewritten has
    the rewritten object removed again: the strip never resurrects a sweep.

DRY RUN unless --write. Output is keys, counts and byte sizes, never content.

    scripts/atif_sessions_job.sh is the runner (MODULE=scripts.strip_session_payloads):
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --customer probe
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --customer probe --write
    MODULE=scripts.strip_session_payloads scripts/atif_sessions_job.sh strip --all-tenants --write
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from typing import Any

import orjson

from engine.ingest.probe_events import project_event
from engine.shared import storage
from engine.shared.constants import AGENT_SESSION_SOURCES
from engine.shared.db import close_pool, init_pool, with_tenant
from engine.shared.exceptions import StorageNotFound
from engine.shared.session_signals import is_cron_marker_key
from engine.shared.session_suppression import deleted_sessions, session_of_event_id
from engine.shared.tenant_status import refusal_for
from engine.shared.transcript_render import lines_from_events
from scripts.atif_sessions import _emit, _queue_row, _queue_rows, _tenants


def strip_batch(body: bytes) -> tuple[bytes | None, str, int]:
    """(new body or None, outcome, bytes removed) for one stored batch."""
    try:
        envelope = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None, "unreadable", 0
    if not isinstance(envelope, dict):
        return None, "unreadable", 0
    payload = envelope.get("payload", envelope)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or not events:
        return None, "no_events", 0
    stripped = [
        {**e, "raw": project_event(e["raw"])} if isinstance(e, dict) and "raw" in e else e
        for e in events
    ]
    if stripped == events:
        return None, "clean", 0
    if lines_from_events(stripped) != lines_from_events(events):
        return None, "lines_changed", 0
    new_payload = {**payload, "events": stripped}
    new_envelope = {**envelope, "payload": new_payload} if "payload" in envelope else new_payload
    new_body = json.dumps(new_envelope, sort_keys=True, separators=(",", ":")).encode()
    return new_body, "stripped", len(body) - len(new_body)


async def _closed(customer_id: str, source: str, session_id: str) -> bool:
    """Deleted, purged (queue row gone) or tenant no longer active."""
    if await refusal_for(customer_id) is not None:
        return True
    if await _queue_row(customer_id, source, session_id) is None:
        return True
    async with with_tenant(customer_id) as conn:
        return bool(await deleted_sessions(conn, customer_id, source, [session_id]))


async def strip(args: argparse.Namespace) -> None:
    store = storage.get_store()
    counts: dict[str, int] = defaultdict(int)
    removed = 0
    sem = asyncio.Semaphore(args.concurrency)

    async def one_key(customer_id: str, bucket: str, row: dict[str, Any], key: str) -> None:
        nonlocal removed
        source, session_id = row["source_system"], row["source_event_id"]
        ident = {"customer": customer_id, "source": source, "session_id": session_id, "key": key}
        async with sem:
            try:
                try:
                    body = await store.get(bucket, key)
                except StorageNotFound:
                    counts["missing"] += 1
                    return
                new_body, outcome, saved = strip_batch(body)
                counts[outcome] += 1
                if outcome == "lines_changed":
                    _emit({"kind": "batch", **ident, "lines_changed": True})
                if new_body is None:
                    return
                removed += saved
                if not args.write:
                    return
                if await _closed(customer_id, source, session_id):
                    counts["closed"] += 1
                    return
                await store.put(bucket, key, new_body)
                if await _closed(customer_id, source, session_id):
                    # Its deletion or purge swept the folder while we wrote:
                    # remove the copy we put back, as the sweep would have.
                    await store.delete(bucket, key)
                    counts["overtaken"] += 1
                    _emit({"kind": "batch", **ident, "overtaken": True})
                    return
                counts["written"] += 1
            except Exception as exc:
                counts[f"error:{type(exc).__name__}"] += 1
                _emit({"kind": "batch", **ident, "error": type(exc).__name__})

    for customer_id in await _tenants(args.customer, args.all_tenants):
        if (refusal := await refusal_for(customer_id)) is not None:
            _emit({"kind": "tenant", "customer": customer_id, "skipped": refusal["status"]})
            continue
        bucket = await store.bucket_for(customer_id)
        before = dict(counts)
        rows = [
            r for r in await _queue_rows(customer_id, args.sources)
            if session_of_event_id(r["source_event_id"]) == r["source_event_id"]
        ]
        await asyncio.gather(*(
            one_key(customer_id, bucket, row, key)
            for row in rows
            for key in dict.fromkeys(k for k in (row["payload_s3_keys"] or []) if k)
            if not is_cron_marker_key(key)
        ))
        _emit({"kind": "tenant", "customer": customer_id, "sessions": len(rows),
               **{k: counts[k] - before.get(k, 0) for k in counts if counts[k] - before.get(k, 0)}})
    _emit({"kind": "summary", "write": args.write, "bytes_removed": removed,
           **dict(sorted(counts.items()))})


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

"""Agent sessions as ATIF (engine/ingest/atif): the replay gate and the backfill.

    replay    READ-ONLY. Re-renders a stratified sample of stored sessions both
              ways -- the uploaded events (what the index holds today) and the
              session's ATIF trajectory -- and reports whether the Lines, the
              text, the evidence spans and the extraction segment inputs are
              identical, plus timings and trajectory sizes. A long-session
              subsample is replayed batch by batch, as live ingestion runs.
              This is the gate before any tenant's SESSION_RENDER is `atif`.

    backfill  Writes `trajectory.json` for stored sessions that have none
              (sessions ended before the engine wrote it on completion). DRY
              RUN unless --write. Writes nothing else: no documents, chunks,
              units, queue rows, and no LLM call. A session recorded as
              deleted is skipped, and re-checked after the write (a deletion
              that landed during the write gets its object removed).

Both read a session exactly as the worker does: the queue row's keys -> first
readable payload -> parse_webhook_event -> fetch_supplementary. Output is ids,
counts and timings, one JSON line per session and a summary line: never any
transcript text.

Run as a throwaway Job from the live worker's pod spec (same image, same
credentials; never `kubectl exec` Python into a serving pod -- an exec'd process
shares the container's memory limit and the OOM killer takes the server):

    scripts/atif_sessions_job.sh replay --all-tenants --sample 500
    scripts/atif_sessions_job.sh backfill --all-tenants            # dry run
    scripts/atif_sessions_job.sh backfill --all-tenants --write
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from typing import Any

import orjson

from engine.ingest.atif.build import build_trajectory
from engine.ingest.atif.lines import lines_from_trajectory
from engine.ingest.atif.models import Trajectory
from engine.ingest.atif.store import trajectory_key, write_trajectory
from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared import claude_code_extraction as _ext
from engine.shared import storage
from engine.shared.constants import AGENT_SESSION_SOURCES, SourceSystem
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.exceptions import StorageNotFound
from engine.shared.models import WebhookEvent
from engine.shared.session_signals import is_cron_marker_key
from engine.shared.session_suppression import deleted_sessions
from engine.shared.tenant_status import ACTIVE_TENANTS_SQL
from engine.shared.transcript_render import lines_from_events, render_lines_indexed

# The connectors register on import; only the ingestion app and the worker
# import them at startup (see scripts/rechunk_collapsed_sessions.py).
import kb.handlers  # noqa: F401  # isort: skip

_QUEUE_SQL = """
    SELECT source_system, source_event_id, status, payload_s3_keys
    FROM ingestion_queue
    WHERE customer_id = $1 AND source_system = ANY($2::text[])
"""


class Skip(Exception):
    """A session this run cannot read; reported with its reason, never fatal."""


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, separators=(",", ":"), default=str), flush=True)


async def _tenants(customers: list[str] | None, all_tenants: bool) -> list[str]:
    if not all_tenants:
        return list(dict.fromkeys(customers or []))
    async with get_pool().acquire() as conn:
        return [r["customer_id"] for r in await conn.fetch(ACTIVE_TENANTS_SQL)]


async def _queue_rows(customer_id: str, sources: list[str]) -> list[dict[str, Any]]:
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(_QUEUE_SQL, customer_id, sources)
    return [{**dict(r), "customer_id": customer_id} for r in rows]


async def _events_by_key(store: Any, customer_id: str, keys: list[str]) -> dict[str, bytes]:
    bucket = await store.bucket_for(customer_id)
    sem = asyncio.Semaphore(8)

    async def one(key: str) -> tuple[str, bytes]:
        if is_cron_marker_key(key):
            return key, b""
        async with sem:
            try:
                return key, await store.get(bucket, key)
            except StorageNotFound:
                return key, b""

    return dict(await asyncio.gather(*(one(k) for k in keys)))


async def _read_session(
    normalizer: Normalizer, store: Any, row: dict[str, Any]
) -> list[dict[str, Any]]:
    """The worker's read: first readable payload -> parse -> fetch_supplementary."""
    customer_id, source = row["customer_id"], SourceSystem(row["source_system"])
    keys = [k for k in (row["payload_s3_keys"] or []) if k]
    if not keys:
        raise Skip("no_keys")
    bucket = await store.bucket_for(customer_id)
    first = None
    for key in keys:
        if is_cron_marker_key(key):
            continue
        try:
            body = await store.get(bucket, key)
        except StorageNotFound:
            continue
        if body:
            first = (key, orjson.loads(body))
            break
    if first is None:
        raise Skip("no_readable_payload")
    first_key, envelope = first
    payload = envelope.get("payload", envelope)
    headers = envelope.get("_headers", {})
    connector = normalizer._connector(source)
    parsed = connector.parse_webhook_event(customer_id, headers, payload)
    if parsed is None:
        raise Skip("unparseable_first_payload")
    event = WebhookEvent(
        customer_id=customer_id,
        source_system=source,
        source_event_id=parsed.source_event_id,
        received_at=parsed.received_at,
        payload_s3_key=first_key,
        payload_s3_keys=keys,
        raw_payload=payload,
        headers=headers,
    )
    hydrated = await connector.fetch_supplementary(event, None)
    return list(hydrated.get("events") or [])


def _segment_digest(lines: list[Any]) -> list[tuple[str, str, int | None, int | None]]:
    """Per extraction segment: (boundary, hash of what the model is sent, line bounds)."""
    segments, _capped = _ext._segment_session(lines)
    out = []
    for segment, boundary in segments:
        kept = [x for x in segment if not _ext._is_compact_summary(x)]
        text, spans = render_lines_indexed(_ext._as_lines(kept))
        digest = hashlib.sha256(orjson.dumps([text, spans])).hexdigest()[:16]
        out.append((boundary, digest, *_ext._line_bounds(segment)))
    return out


def _compare(events: list[dict[str, Any]], session_id: str, agent: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    legacy = lines_from_events(events)
    t1 = time.perf_counter()
    built = build_trajectory(events, session_id=session_id, agent_name=agent)
    t2 = time.perf_counter()
    try:
        atif = lines_from_trajectory(built.trajectory)
        render_error = None
    except Exception as exc:
        atif, render_error = None, type(exc).__name__
    t3 = time.perf_counter()
    same = atif == legacy
    first_diff = None
    diff_kind = None
    if atif is not None and not same:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(atif, legacy, strict=False)) if a != b),
            min(len(atif), len(legacy)),
        )
        if first_diff < min(len(atif), len(legacy)):
            a, b = atif[first_diff], legacy[first_diff]
            diff_kind = "text" if a.text != b.text else "flags_or_line_no"
        else:
            diff_kind = "length"
    text_same = spans_same = segments_same = None
    if atif is not None:
        text_same, spans_same = (
            render_lines_indexed(atif)[0] == render_lines_indexed(legacy)[0],
            render_lines_indexed(atif)[1] == render_lines_indexed(legacy)[1],
        )
        segments_same = _segment_digest(atif) == _segment_digest(legacy)
    try:
        Trajectory.model_validate(built.trajectory)
        invalid = None
    except Exception as exc:
        invalid = str(exc).splitlines()[0][:200]
    return {
        "events": len(events),
        "steps": len(built.trajectory.get("steps") or []),
        "same": same,
        "text_same": text_same,
        "spans_same": spans_same,
        "segments_same": segments_same,
        "first_diff": first_diff,
        "diff_kind": diff_kind,
        "render_error": render_error,
        "unparsed": built.unparsed,
        "invalid": invalid,
        "trajectory_bytes": len(orjson.dumps(built.trajectory)),
        "ms_legacy": round((t1 - t0) * 1000, 2),
        "ms_build": round((t2 - t1) * 1000, 2),
        "ms_atif": round((t3 - t2) * 1000, 2),
    }


def _merge_prefix(bodies: dict[str, bytes], keys: list[str]) -> list[dict[str, Any]]:
    """fetch_supplementary's merge over the first `keys` (dedupe by line_no, sort)."""
    seen: set[int] = set()
    merged: list[dict[str, Any]] = []
    for key in keys:
        body = bodies.get(key) or b""
        if not body:
            continue
        try:
            envelope = orjson.loads(body)
        except orjson.JSONDecodeError:
            continue
        payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else {}
        for obj in (payload or {}).get("events") or []:
            if not isinstance(obj, dict):
                continue
            n = obj.get("line_no")
            if n is not None:
                if n in seen:
                    continue
                seen.add(n)
            merged.append(obj)
    merged.sort(key=lambda e: (e.get("line_no") is None, e.get("line_no") or 0))
    return merged


def _batchwise(bodies: dict[str, bytes], keys: list[str], session_id: str, agent: str,
               points: int) -> dict[str, Any]:
    """Replay the session as live ingestion did: one pass per batch prefix.

    Sampled at `points` evenly spaced prefixes (a 2,485-key session would be
    millions of re-reads otherwise). Reports cumulative milliseconds for the
    legacy path alone and for legacy + build + trajectory lines, and whether
    every prefix agreed.
    """
    readable = [k for k in keys if bodies.get(k)]
    n = len(readable)
    if n == 0:
        return {"prefixes": 0}
    cuts = sorted({max(1, round(n * (i + 1) / points)) for i in range(points)})
    legacy_ms = atif_ms = 0.0
    disagreements = 0
    for cut in cuts:
        events = _merge_prefix(bodies, readable[:cut])
        r = _compare(events, session_id, agent)
        legacy_ms += r["ms_legacy"]
        atif_ms += r["ms_legacy"] + r["ms_build"] + r["ms_atif"]
        disagreements += 0 if r["same"] else 1
    return {
        "prefixes": len(cuts),
        "keys": n,
        "cumulative_ms_legacy": round(legacy_ms, 1),
        "cumulative_ms_with_atif": round(atif_ms, 1),
        "prefix_disagreements": disagreements,
    }


def _sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Stratified by (tenant, source): every stratum gets at least one, the rest
    in proportion to its share of sessions."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["customer_id"], row["source_system"])].append(row)
    total = len(rows)
    chosen: list[dict[str, Any]] = []
    for _key, members in sorted(strata.items()):
        quota = max(1, round(size * len(members) / max(total, 1)))
        chosen.extend(rng.sample(members, min(quota, len(members))))
    rng.shuffle(chosen)
    return chosen[: max(size, len(strata))]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def replay(args: argparse.Namespace) -> None:
    normalizer = Normalizer(make_default_context())
    store = storage.get_store()
    rows: list[dict[str, Any]] = []
    for customer_id in await _tenants(args.customer, args.all_tenants):
        # Read-only: a row the worker is about to re-run is as good a sample as any.
        rows.extend(await _queue_rows(customer_id, args.sources))
    sample = _sample(rows, args.sample, args.seed)
    results: list[dict[str, Any]] = []
    by_keys = sorted(sample, key=lambda r: len(r["payload_s3_keys"] or []), reverse=True)
    batchwise_ids = {(r["customer_id"], r["source_event_id"]) for r in by_keys[: args.batchwise]}
    for row in sample:
        ident = {"customer": row["customer_id"], "source": row["source_system"],
                 "session_id": row["source_event_id"],
                 "keys": len(row["payload_s3_keys"] or [])}
        try:
            events = await _read_session(normalizer, store, row)
            record = {**ident, **_compare(events, row["source_event_id"], row["source_system"])}
            if (row["customer_id"], row["source_event_id"]) in batchwise_ids:
                keys = [k for k in row["payload_s3_keys"] or [] if k]
                bodies = await _events_by_key(store, row["customer_id"], keys)
                record["batchwise"] = _batchwise(
                    bodies, keys, row["source_event_id"], row["source_system"], args.points,
                )
        except Skip as skip:
            record = {**ident, "skipped": str(skip)}
        except Exception as exc:
            record = {**ident, "error": type(exc).__name__}
        results.append(record)
        _emit({"kind": "session", **record})

    compared = [r for r in results if "same" in r]
    ratio = [(r["ms_legacy"] + r["ms_build"] + r["ms_atif"]) / r["ms_legacy"]
             for r in compared if r["ms_legacy"] > 0]
    sizes = [r["trajectory_bytes"] for r in compared]
    batch = [r["batchwise"] for r in compared if r.get("batchwise", {}).get("prefixes")]
    summary = {
        "kind": "summary",
        "candidates": len(rows),
        "sampled": len(sample),
        "compared": len(compared),
        "skipped": sum(1 for r in results if "skipped" in r),
        "errors": sum(1 for r in results if "error" in r),
        "identical": sum(1 for r in compared if r["same"]),
        "text_identical": sum(1 for r in compared if r["text_same"]),
        "spans_identical": sum(1 for r in compared if r["spans_same"]),
        "segments_identical": sum(1 for r in compared if r["segments_same"]),
        "with_unparsed": sum(1 for r in compared if r["unparsed"]),
        "invalid": sum(1 for r in compared if r["invalid"]),
        "diff_kinds": dict(sorted(
            ((k, sum(1 for r in compared if r["diff_kind"] == k))
             for k in {r["diff_kind"] for r in compared if r["diff_kind"]}),
        )),
        "by_source": {
            s: sum(1 for r in compared if r["source"] == s) for s in sorted({r["source"] for r in compared})
        },
        "render_ratio_p50": _pct(ratio, 0.5),
        "render_ratio_p95": _pct(ratio, 0.95),
        "trajectory_bytes_p50": _pct(sizes, 0.5),
        "trajectory_bytes_p95": _pct(sizes, 0.95),
        "trajectory_bytes_max": max(sizes) if sizes else None,
        "over_8mb": sum(1 for s in sizes if s > 8_000_000),
        "batchwise_sessions": len(batch),
        "batchwise_disagreements": sum(b["prefix_disagreements"] for b in batch),
        "batchwise_cumulative_ratio": (
            round(sum(b["cumulative_ms_with_atif"] for b in batch)
                  / max(sum(b["cumulative_ms_legacy"] for b in batch), 1e-9), 2)
            if batch else None
        ),
        "gate_passed": bool(compared) and all(
            r["same"] and r["text_same"] and r["spans_same"] and r["segments_same"]
            for r in compared
        ) and not any(b["prefix_disagreements"] for b in batch),
    }
    if ratio:
        summary["render_ratio_mean"] = round(statistics.fmean(ratio), 2)
    _emit(summary)


async def _deleted(customer_id: str, source: str, session_id: str) -> bool:
    async with with_tenant(customer_id) as conn:
        return bool(await deleted_sessions(conn, customer_id, source, [session_id]))


async def backfill(args: argparse.Namespace) -> None:
    normalizer = Normalizer(make_default_context())
    store = storage.get_store()
    counts: dict[str, int] = defaultdict(int)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row: dict[str, Any]) -> None:
        customer_id, source, session_id = (
            row["customer_id"], row["source_system"], row["source_event_id"]
        )
        ident = {"customer": customer_id, "source": source, "session_id": session_id}
        key = trajectory_key(source, customer_id, session_id)
        async with sem:
            try:
                bucket = await store.bucket_for(customer_id)
                if not args.force and await store.exists(bucket, key):
                    counts["already_present"] += 1
                    return
                if await _deleted(customer_id, source, session_id):
                    counts["deleted"] += 1
                    return
                events = await _read_session(normalizer, store, row)
                built = build_trajectory(events, session_id=session_id, agent_name=source)
                if not args.write:
                    counts["would_write"] += 1
                    _emit({"kind": "session", **ident, "would_write": True,
                           "steps": len(built.trajectory["steps"]), "unparsed": built.unparsed})
                    return
                size, invalid = await write_trajectory(
                    store, bucket, key, built.trajectory, max_bytes=args.max_bytes
                )
                if await _deleted(customer_id, source, session_id):
                    # A deletion finished while we wrote: its sweep has run, so
                    # nothing else will remove this object.
                    await store.delete(bucket, key)
                    counts["deleted_during_write"] += 1
                    return
                counts["written"] += 1
                counts["invalid"] += 1 if invalid else 0
                _emit({"kind": "session", **ident, "written": size, "invalid": bool(invalid),
                       "unparsed": built.unparsed})
            except Skip as skip:
                counts[f"skipped:{skip}"] += 1
            except Exception as exc:
                counts[f"error:{type(exc).__name__}"] += 1
                _emit({"kind": "session", **ident, "error": type(exc).__name__})

    for customer_id in await _tenants(args.customer, args.all_tenants):
        # A row the worker is about to run writes its own trajectory if it ends.
        rows = [r for r in await _queue_rows(customer_id, args.sources)
                if r["status"] not in ("pending", "processing")]
        await asyncio.gather(*(one(r) for r in rows))
        _emit({"kind": "tenant", "customer": customer_id, "sessions": len(rows)})
    _emit({"kind": "summary", "write": args.write, **dict(sorted(counts.items()))})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "backfill"):
        p = sub.add_parser(name)
        scope = p.add_mutually_exclusive_group(required=True)
        scope.add_argument("--customer", action="append")
        scope.add_argument("--all-tenants", action="store_true")
        p.add_argument("--sources", nargs="*",
                       default=sorted(s.value for s in AGENT_SESSION_SOURCES))
        if name == "replay":
            p.add_argument("--sample", type=int, default=500)
            p.add_argument("--seed", type=int, default=20261006)
            p.add_argument("--batchwise", type=int, default=20,
                           help="replay this many of the sample's longest sessions batch by batch")
            p.add_argument("--points", type=int, default=50,
                           help="prefixes per batchwise session")
        else:
            p.add_argument("--write", action="store_true")
            p.add_argument("--force", action="store_true",
                           help="rewrite trajectories that already exist")
            p.add_argument("--concurrency", type=int, default=4)
            p.add_argument("--max-bytes", type=int, default=64_000_000)
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from engine.shared.config import get_settings

    await init_pool(get_settings())
    try:
        await (replay(args) if args.command == "replay" else backfill(args))
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())

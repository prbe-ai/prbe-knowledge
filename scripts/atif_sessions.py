"""Agent sessions as ATIF (engine/ingest/atif): the replay gate and the backfill.

    replay    READ-ONLY. Re-renders a stratified sample of stored sessions both
              ways -- the uploaded events (what the index holds today) and the
              session's ATIF trajectory -- and reports whether the Lines, the
              text, the evidence spans and the extraction segment inputs are
              identical, plus timings and trajectory sizes. A long-session
              subsample is replayed batch by batch, as live ingestion runs.
              This is the gate before any tenant's SESSION_RENDER is `atif`.
              --compare-builders also builds each session with BOTH builders
              (engine/ingest/atif/build.py: the frozen `reference` and
              `fold(map(fragment, events))`), reports whether they agree and
              where they first differ, at every sampled batch prefix too, and
              compares both with the session's stored final trajectory.json
              when there is one. This is the gate before SESSION_ATIF_BUILDER
              is `fold` and before any client uploads fragments.

    backfill  Writes `trajectory.json` for ENDED stored sessions that have
              none (they ended before the engine wrote it on completion). DRY
              RUN unless --write. Writes nothing else: no documents, chunks,
              units, queue rows, and no LLM call. Skips a deleted session, a
              running one, one whose queue row is pending or processing, and a
              tenant that is not active. Builds with SESSION_ATIF_BUILDER,
              as the worker does. After each write it re-reads the
              queue row: if the row moved on (a new batch or end marker), is
              gone (deletion, purge) or the session or tenant was closed
              meanwhile, the object is removed -- a reader then gets "not
              built", never a stale document -- and a rerun rebuilds it.

Both read a session exactly as the worker does: the queue row's keys -> first
readable payload -> parse_webhook_event -> fetch_supplementary. Output is ids,
counts and timings, one JSON line per session and a summary line: never any
transcript text.

A protocol-3 session (its client uploaded ATIF fragments, kb/session_receipts.py)
is read as the worker reads it: Lines from its fragments, trajectory
fold(fragments). replay checks that fold's provenance renders those Lines back;
--compare-builders, when its batches carried the canary's events, also compares
the client's fragments with fragment(events) and fold(client fragments) with the
frozen builder over the events, and with neither, fold against the stored copy
(`builders_skipped`). backfill writes fold(fragments).

Run as a throwaway Job from the live worker's pod spec, pinned to the image
digest the worker runs (same code, same credentials; never `kubectl exec` Python
into a serving pod -- an exec'd process shares the container's memory limit and
the OOM killer takes the server):

    scripts/atif_sessions_job.sh replay --all-tenants --sample 500
    scripts/atif_sessions_job.sh replay --all-tenants --sample 2000 --compare-builders
    scripts/atif_sessions_job.sh backfill --all-tenants            # dry run
    scripts/atif_sessions_job.sh backfill --all-tenants --write
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from enum import StrEnum
from typing import Any

import orjson

from engine.ingest.atif.build import Builder, BuildResult, build_trajectory, builder_for
from engine.ingest.atif.compare import (
    PATH_KEYS,
    WRITE_STAMPS,
    first_difference,
    fragments_difference,
    result_data,
)
from engine.ingest.atif.fragment import fragment
from engine.ingest.atif.lines import lines_from_trajectory
from engine.ingest.atif.models import Trajectory
from engine.ingest.atif.store import (
    TrajectoryTooLarge,
    read_trajectory,
    scrub_trajectory,
    strip_provenance,
    trajectory_key,
    write_trajectory,
)
from engine.ingest.atif.uploaded import fold_fragments, fragment_lines, fragment_ordinal
from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared import claude_code_extraction as _ext
from engine.shared import storage
from engine.shared.constants import AGENT_SESSION_SOURCES, SourceSystem
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.exceptions import StorageNotFound
from engine.shared.models import WebhookEvent
from engine.shared.session_signals import is_cron_marker_key
from engine.shared.session_suppression import deleted_sessions, session_of_event_id
from engine.shared.tenant_status import ACTIVE_TENANTS_SQL, refusal_for
from engine.shared.transcript_render import lines_from_events, render_lines_indexed

# The connectors register on import; only the ingestion app and the worker
# import them at startup (see scripts/rechunk_collapsed_sessions.py).
import kb.handlers  # noqa: F401  # isort: skip

_QUEUE_SQL = """
    SELECT source_system, source_event_id, status, version, payload_s3_keys
    FROM ingestion_queue
    WHERE customer_id = $1 AND source_system = ANY($2::text[])
"""
_ROW_SQL = """
    SELECT status, version, payload_s3_keys
    FROM ingestion_queue
    WHERE customer_id = $1 AND source_system = $2 AND source_event_id = $3
"""
_BUSY = ("pending", "processing")

#: The write stamps a rebuild is compared without, and the content-free
#: difference report: engine/ingest/atif/compare.py (the ingest pass's fragment
#: shadow reports the same way). The render provenance (`extra.probe`, each
#: step's `extra.probe_parts`) is never stored (store.strip_provenance), so a
#: rebuild goes through the same strip and the same credential scrub as the
#: write before it is compared. Between the two builders nothing is ignored:
#: the whole BuildResult, provenance included, must be equal.
_WRITE_STAMPS = WRITE_STAMPS
_PATH_KEYS = PATH_KEYS
_first_difference = first_difference
_result_data = result_data


class StoredCopy(StrEnum):
    """What a session's stored trajectory.json was, for --compare-builders."""

    #: The session has not ended: only a live copy can exist, of other events.
    NOT_ENDED = "not_ended"
    ABSENT = "absent"
    #: An ended session whose final copy is not written yet.
    LIVE = "live"
    FINAL = "final"
    ERROR = "error"


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


async def _queue_row(customer_id: str, source: str, event_id: str) -> dict[str, Any] | None:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(_ROW_SQL, customer_id, source, event_id)
    return dict(row) if row is not None else None


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
) -> dict[str, Any]:
    """The worker's read: first readable payload -> parse -> fetch_supplementary.
    Returns its result: `events`, `session_id`, `session_complete`, ..."""
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
    return await connector.fetch_supplementary(event, None)


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


def _compare(
    events: list[dict[str, Any]], session_id: str, agent: str, builder: Builder = Builder.REFERENCE
) -> dict[str, Any]:
    t0 = time.perf_counter()
    legacy = lines_from_events(events)
    t1 = time.perf_counter()
    try:
        built = build_trajectory(events, session_id=session_id, agent_name=agent, builder=builder)
    except Exception as exc:  # a builder crash on a real shape fails the gate
        return {"events": len(events), "same": False, "build_error": type(exc).__name__}
    return _render_record(legacy, built, len(events), t0, t1, time.perf_counter())


def _compare_uploaded(fragments: list[Any], session_id: str, agent: str) -> dict[str, Any]:
    """`_compare` for a protocol-3 session: the Lines it serves are its
    fragments' own (engine/ingest/atif/uploaded.py), its trajectory is
    fold(fragments); the record says whether fold's provenance renders them
    back. `ms_legacy` is the time to read the fragments' Lines."""
    t0 = time.perf_counter()
    read = fragment_lines(fragments)
    t1 = time.perf_counter()
    try:
        built = fold_fragments(fragments, session_id, agent)
    except Exception as exc:  # fold never raises on content; a crash fails the gate
        return {
            "events": len(fragments),
            "same": False,
            "build_error": type(exc).__name__,
            "protocol": 3,
        }
    record = _render_record(read.lines, built, len(fragments), t0, t1, time.perf_counter())
    return {**record, "protocol": 3, "fragments_degraded": read.degraded}


def _render_record(
    legacy: list[Any], built: BuildResult, count: int, t0: float, t1: float, t2: float
) -> dict[str, Any]:
    """Whether a build's trajectory renders back to the Lines the index holds."""
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
        "events": count,
        "steps": len(built.trajectory.get("steps") or []),
        "same": same,
        "text_same": text_same,
        "spans_same": spans_same,
        "segments_same": segments_same,
        "first_diff": first_diff,
        "diff_kind": diff_kind,
        "render_error": render_error,
        "build_error": None,
        "unparsed": built.unparsed,
        "invalid": invalid,
        "trajectory_bytes": len(orjson.dumps(built.trajectory)),
        "ms_legacy": round((t1 - t0) * 1000, 2),
        "ms_build": round((t2 - t1) * 1000, 2),
        "ms_atif": round((t3 - t2) * 1000, 2),
    }


def _builders(
    events: list[dict[str, Any]], session_id: str, agent: str
) -> tuple[dict[str, Any], BuildResult | None, BuildResult | None]:
    """Both builders over the same events: (record, reference, fold)."""
    built: dict[Builder, BuildResult] = {}
    for builder in (Builder.REFERENCE, Builder.FOLD):
        try:
            built[builder] = build_trajectory(
                events, session_id=session_id, agent_name=agent, builder=builder
            )
        except Exception as exc:  # a crash on a real shape fails the gate
            return (
                {
                    "builders_same": False,
                    "builders_diff": None,
                    "builders_error": f"{builder.value}:{type(exc).__name__}",
                },
                None,
                None,
            )
    reference, folded = built[Builder.REFERENCE], built[Builder.FOLD]
    diff = _first_difference(_result_data(reference), _result_data(folded))
    return (
        {"builders_same": diff is None, "builders_diff": diff, "builders_error": None},
        reference,
        folded,
    )


def _uploaded_builders(
    fragments: list[Any], events: list[dict[str, Any]], session_id: str, agent: str
) -> tuple[dict[str, Any], BuildResult | None, BuildResult | None]:
    """A protocol-3 session whose batches carried the canary's events: the
    frozen builder over the events against fold over the CLIENT's fragments,
    and the client's fragments against this engine's fragment(events)."""
    try:
        reference = build_trajectory(
            events, session_id=session_id, agent_name=agent, builder=Builder.REFERENCE
        )
        folded = fold_fragments(fragments, session_id, agent)
        server = [fragment(e) for e in events if isinstance(e, dict)]
    except Exception as exc:  # a crash on a real shape fails the gate
        return (
            {
                "builders_same": False,
                "builders_diff": None,
                "builders_error": f"uploaded:{type(exc).__name__}",
            },
            None,
            None,
        )
    diff = _first_difference(_result_data(reference), _result_data(folded))
    fragments_diff, _differing = fragments_difference(fragments, server)
    return (
        {
            "builders_same": diff is None,
            "builders_diff": diff,
            "builders_error": None,
            "fragments_same": fragments_diff is None,
            "fragments_diff": fragments_diff,
        },
        reference,
        folded,
    )


def _without_stamps(document: dict[str, Any]) -> dict[str, Any]:
    out = dict(document)
    extra = {k: v for k, v in (out.get("extra") or {}).items() if k not in _WRITE_STAMPS}
    if extra:
        out["extra"] = extra
    else:
        out.pop("extra", None)
    return out


async def _as_stored(trajectory: dict[str, Any]) -> dict[str, Any]:
    """What the worker's final write stores for this build (store.write_trajectory:
    the final stamp, strip, scrub), less the write stamps."""
    final = {**trajectory, "extra": {**(trajectory.get("extra") or {}), "session_ended": True}}
    clean = await scrub_trajectory(strip_provenance(final))
    return _without_stamps(orjson.loads(orjson.dumps(clean)))


async def _compare_builders(
    store: Any, row: dict[str, Any], events: list[dict[str, Any]], session_id: str, complete: bool
) -> dict[str, Any]:
    """--compare-builders for one session: the two builders, then each against
    the stored final copy. Ids, counts and schema paths only."""
    source = row["source_system"]
    record, reference, folded = _builders(events, session_id, source)
    if reference is None or folded is None:
        return record
    if not complete:
        return {**record, "stored": StoredCopy.NOT_ENDED.value}
    try:
        bucket = await store.bucket_for(row["customer_id"])
        stored = await read_trajectory(
            store, bucket, trajectory_key(source, row["customer_id"], session_id)
        )
        if stored is None:
            return {**record, "stored": StoredCopy.ABSENT.value}
        if (stored.get("extra") or {}).get("session_ended") is False:
            return {**record, "stored": StoredCopy.LIVE.value}
        actual = _without_stamps(stored)
        fold_diff = _first_difference(actual, await _as_stored(folded.trajectory))
        reference_diff = (
            fold_diff
            if record["builders_same"]
            else _first_difference(actual, await _as_stored(reference.trajectory))
        )
    except Exception as exc:
        return {**record, "stored": StoredCopy.ERROR.value, "stored_error": type(exc).__name__}
    return {
        **record,
        "stored": StoredCopy.FINAL.value,
        "stored_fold_same": fold_diff is None,
        "stored_fold_diff": fold_diff,
        "stored_reference_same": reference_diff is None,
        "stored_reference_diff": reference_diff,
    }


async def _compare_builders_uploaded(
    store: Any,
    row: dict[str, Any],
    fragments: list[Any],
    events: list[dict[str, Any]],
    session_id: str,
    complete: bool,
) -> dict[str, Any]:
    """--compare-builders for a protocol-3 session. With the canary's events:
    `_uploaded_builders`, then both builds against the stored final copy.
    Without them nothing independent exists to compare with: only fold of the
    stored fragments against the stored final copy (`builders_skipped`)."""
    source = row["source_system"]
    if events:
        record, reference, folded = _uploaded_builders(fragments, events, session_id, source)
        if reference is None or folded is None:
            return record
    else:
        record = {"builders_skipped": "no_events"}
        reference, folded = None, fold_fragments(fragments, session_id, source)
    if not complete:
        return {**record, "stored": StoredCopy.NOT_ENDED.value}
    try:
        bucket = await store.bucket_for(row["customer_id"])
        stored = await read_trajectory(
            store, bucket, trajectory_key(source, row["customer_id"], session_id)
        )
        if stored is None:
            return {**record, "stored": StoredCopy.ABSENT.value}
        if (stored.get("extra") or {}).get("session_ended") is False:
            return {**record, "stored": StoredCopy.LIVE.value}
        actual = _without_stamps(stored)
        fold_diff = _first_difference(actual, await _as_stored(folded.trajectory))
        if reference is None:
            reference_diff = None
        elif record["builders_same"]:
            reference_diff = fold_diff
        else:
            reference_diff = _first_difference(actual, await _as_stored(reference.trajectory))
    except Exception as exc:
        return {**record, "stored": StoredCopy.ERROR.value, "stored_error": type(exc).__name__}
    return {
        **record,
        "stored": StoredCopy.FINAL.value,
        "stored_fold_same": fold_diff is None,
        "stored_fold_diff": fold_diff,
        "stored_reference_same": None if reference is None else reference_diff is None,
        "stored_reference_diff": reference_diff,
    }


def _merge_prefix(
    bodies: dict[str, bytes], keys: list[str], field: str = "events"
) -> list[dict[str, Any]]:
    """fetch_supplementary's merge over the first `keys` (dedupe by ordinal,
    sort). `field` "fragments" merges a protocol-3 session's fragments; on such
    a session "events" are the canary's."""
    ordinal = fragment_ordinal if field == "fragments" else (lambda obj: obj.get("line_no"))
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
        for obj in (payload or {}).get(field) or []:
            if not isinstance(obj, dict):
                continue
            n = ordinal(obj)
            if n is not None:
                if n in seen:
                    continue
                seen.add(n)
            merged.append(obj)
    merged.sort(key=lambda e: (ordinal(e) is None, ordinal(e) or 0))
    return merged


def _batchwise(
    bodies: dict[str, bytes],
    keys: list[str],
    session_id: str,
    agent: str,
    points: int,
    builder: Builder = Builder.REFERENCE,
    compare_builders: bool = False,
    uploaded: bool = False,
) -> dict[str, Any]:
    """Replay the session as live ingestion did: one pass per batch prefix.
    `uploaded`: a protocol-3 session, read from its fragments (with the
    canary's events, when its batches carry them, for `compare_builders`).

    Sampled at `points` evenly spaced prefixes (a 2,485-key session would be
    millions of re-reads otherwise). Reports cumulative milliseconds for the
    legacy path alone and for legacy + build + trajectory lines, and whether
    every prefix agreed; with `compare_builders`, whether both builders agree
    on every prefix (what a live pass folds mid-session).
    """
    readable = [k for k in keys if bodies.get(k)]
    n = len(readable)
    if n == 0:
        return {"prefixes": 0}
    cuts = sorted({max(1, round(n * (i + 1) / points)) for i in range(points)})
    legacy_ms = atif_ms = 0.0
    disagreements = 0
    builder_disagreements = 0
    for cut in cuts:
        events = _merge_prefix(bodies, readable[:cut])
        if uploaded:
            fragments = _merge_prefix(bodies, readable[:cut], "fragments")
            if (
                compare_builders
                and events
                and not _uploaded_builders(fragments, events, session_id, agent)[0]["builders_same"]
            ):
                builder_disagreements += 1
            r = _compare_uploaded(fragments, session_id, agent)
        else:
            if compare_builders and not _builders(events, session_id, agent)[0]["builders_same"]:
                builder_disagreements += 1
            r = _compare(events, session_id, agent, builder)
        if r.get("build_error"):
            disagreements += 1
            continue
        legacy_ms += r["ms_legacy"]
        atif_ms += r["ms_legacy"] + r["ms_build"] + r["ms_atif"]
        disagreements += 0 if r["same"] else 1
    out = {
        "prefixes": len(cuts),
        "keys": n,
        "cumulative_ms_legacy": round(legacy_ms, 1),
        "cumulative_ms_with_atif": round(atif_ms, 1),
        "prefix_disagreements": disagreements,
    }
    if compare_builders:
        out["prefix_builder_disagreements"] = builder_disagreements
    return out


def _strata(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["customer_id"], row["source_system"])].append(row)
    return strata


def _sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Stratified by (tenant, source): every stratum gets one first, then the
    remaining places go in proportion to each stratum's share of sessions."""
    rng = random.Random(seed)
    strata = _strata(rows)
    total = len(rows)
    chosen: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for _key, members in sorted(strata.items()):
        picked = rng.sample(members, len(members))
        chosen.append(picked[0])
        quota = max(0, round(size * len(members) / max(total, 1)) - 1)
        rest.extend(picked[1 : 1 + quota])
    rng.shuffle(rest)
    return chosen + rest[: max(0, size - len(chosen))]


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
    builder = builder_for(_settings().session_atif_builder)
    for row in sample:
        ident = {"customer": row["customer_id"], "source": row["source_system"],
                 "session_id": row["source_event_id"],
                 "keys": len(row["payload_s3_keys"] or [])}
        uploaded = False
        try:
            hydrated = await _read_session(normalizer, store, row)
            session_id = hydrated.get("session_id") or row["source_event_id"]
            complete = bool(hydrated.get("session_complete"))
            events = list(hydrated.get("events") or [])
            fragments = hydrated.get("fragments")
            uploaded = fragments is not None
            if uploaded:
                # Protocol 3: read from its fragments; any `events` are the canary's.
                fragments = list(fragments)
                events = list(hydrated.get("canary_events") or [])
            del hydrated
            if uploaded:
                record = {**ident, **_compare_uploaded(fragments, session_id, row["source_system"])}
                if args.compare_builders:
                    record.update(
                        await _compare_builders_uploaded(
                            store, row, fragments, events, session_id, complete
                        )
                    )
            else:
                record = {**ident, **_compare(events, session_id, row["source_system"], builder)}
                if args.compare_builders:
                    record.update(await _compare_builders(store, row, events, session_id, complete))
            del events, fragments
        except Skip as skip:
            record = {**ident, "skipped": str(skip)}
        except Exception as exc:
            record = {**ident, "error": type(exc).__name__}
        if "same" in record and (row["customer_id"], row["source_event_id"]) in batchwise_ids:
            try:
                keys = [k for k in row["payload_s3_keys"] or [] if k]
                bodies = await _events_by_key(store, row["customer_id"], keys)
                record["batchwise"] = _batchwise(
                    bodies, keys, row["source_event_id"], row["source_system"], args.points,
                    builder, args.compare_builders, uploaded,
                )
                del bodies
            except Exception as exc:
                record["batchwise"] = {"error": type(exc).__name__}
        results.append(record)
        _emit({"kind": "session", **record})

    build_errors = [r for r in results if r.get("build_error")]
    compared = [r for r in results if "same" in r and not r.get("build_error")]
    batch_errors = sum(1 for r in compared if r.get("batchwise", {}).get("error"))
    strata = _strata(rows)
    covered = {(r["customer"], r["source"]) for r in compared}
    sampled_strata = {(r["customer_id"], r["source_system"]) for r in sample}
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
        "build_errors": len(build_errors),
        "build_error_kinds": sorted({r["build_error"] for r in build_errors}),
        "strata": len(strata),
        "strata_sampled": len(sampled_strata),
        "strata_compared": len(covered & set(strata)),
        "identical": sum(1 for r in compared if r["same"]),
        "text_identical": sum(1 for r in compared if r["text_same"]),
        "spans_identical": sum(1 for r in compared if r["spans_same"]),
        "segments_identical": sum(1 for r in compared if r["segments_same"]),
        "with_unparsed": sum(1 for r in compared if r["unparsed"]),
        "uploaded": sum(1 for r in compared if r.get("protocol") == 3),
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
        "batchwise_errors": batch_errors,
        "batchwise_cumulative_ratio": (
            round(sum(b["cumulative_ms_with_atif"] for b in batch)
                  / max(sum(b["cumulative_ms_legacy"] for b in batch), 1e-9), 2)
            if batch else None
        ),
        # Strict: a read error, a builder crash, a batchwise failure or an
        # unsampled stratum fails it; a person reads the records. (A stratum
        # whose sample was all skipped, e.g. no keys left, is reported in
        # strata_compared, not failed.)
        "gate_passed": bool(compared) and all(
            r["same"] and r["text_same"] and r["spans_same"] and r["segments_same"]
            for r in compared
        ) and not any(b["prefix_disagreements"] for b in batch)
        and not build_errors and not batch_errors
        and not any("error" in r for r in results)
        and len(sampled_strata) == len(strata),
    }
    if ratio:
        summary["render_ratio_mean"] = round(statistics.fmean(ratio), 2)
    if args.compare_builders:
        summary.update(_builders_summary(results))
    _emit(summary)


def _builders_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """--compare-builders over the run: counts, and where differences start
    (paths with their indexes folded, e.g. `trajectory.steps[].metrics`).

    Over every session that reached the comparison, including one whose
    configured builder crashed in the render gate (with SESSION_ATIF_BUILDER
    `fold`, that is a fold crash, which `compared` above leaves out)."""
    compared = [r for r in results if "builders_same" in r]
    unread = sum(1 for r in results if "error" in r)
    batch = [r["batchwise"] for r in compared if r.get("batchwise", {}).get("prefixes")]
    batch_errors = sum(1 for r in compared if r.get("batchwise", {}).get("error"))
    errors = [r for r in compared if r.get("builders_error")]
    paths = Counter(
        re.sub(r"\[\d+\]", "[]", r["builders_diff"]["path"])
        for r in compared
        if r.get("builders_diff")
    )
    # Every session with a stored-copy check, including a protocol-3 one with
    # no canary events (`builders_skipped`), which has no builders to compare.
    with_stored = [r for r in results if r.get("stored")]
    stored = Counter(r["stored"] for r in with_stored)
    final = [r for r in with_stored if r["stored"] == StoredCopy.FINAL]
    uploaded = [r for r in compared if "fragments_same" in r]
    stored_paths = Counter(
        re.sub(r"\[\d+\]", "[]", r["stored_fold_diff"]["path"])
        for r in final
        if r.get("stored_fold_diff")
    )
    prefix_disagreements = sum(b.get("prefix_builder_disagreements", 0) for b in batch)
    # Wherever the reference reproduces the stored copy, fold must too.
    fold_lost_stored = sum(
        1 for r in final if r["stored_reference_same"] and not r["stored_fold_same"]
    )
    return {
        "builders_compared": len(compared),
        "builders_identical": sum(1 for r in compared if r.get("builders_same")),
        "builders_errors": len(errors),
        "builders_error_kinds": sorted({r["builders_error"] for r in errors}),
        "builders_diff_paths": dict(paths.most_common(20)),
        "builders_prefix_disagreements": prefix_disagreements,
        "builders_unread": unread,
        "stored": dict(sorted(stored.items())),
        "stored_fold_identical": sum(1 for r in final if r["stored_fold_same"]),
        "stored_reference_identical": sum(1 for r in final if r["stored_reference_same"]),
        "stored_fold_diff_paths": dict(stored_paths.most_common(20)),
        "stored_fold_lost": fold_lost_stored,
        "builders_skipped": sum(1 for r in results if r.get("builders_skipped")),
        "fragments_compared": len(uploaded),
        "fragments_identical": sum(1 for r in uploaded if r["fragments_same"]),
        # Strict, as gate_passed: every compared session and sampled prefix
        # agrees, nothing crashed or went unread, and fold reproduces every
        # stored copy the reference does. Stored copies the reference no longer
        # reproduces (written before a builder fix, or by an older scanner) are
        # counted, not failed.
        "builders_gate_passed": bool(compared)
        and all(r.get("builders_same") for r in compared)
        and not errors
        and not unread
        and not batch_errors
        and not prefix_disagreements
        and not fold_lost_stored
        # A protocol-3 client's fragments are what this engine makes of its events.
        and all(r["fragments_same"] for r in uploaded),
    }


def _settings() -> Any:
    from engine.shared.config import get_settings

    return get_settings()


async def _deleted(customer_id: str, source: str, session_id: str) -> bool:
    async with with_tenant(customer_id) as conn:
        return bool(await deleted_sessions(conn, customer_id, source, [session_id]))


async def backfill(args: argparse.Namespace) -> None:
    normalizer = Normalizer(make_default_context())
    store = storage.get_store()
    counts: dict[str, int] = defaultdict(int)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row: dict[str, Any]) -> None:
        customer_id, source, event_id = (
            row["customer_id"], row["source_system"], row["source_event_id"]
        )
        ident = {"customer": customer_id, "source": source, "session_id": event_id}
        async with sem:
            try:
                if session_of_event_id(event_id) != event_id:
                    # A pre-0026 `<sid>:finalize` / `<sid>:<n>` row: not a session.
                    counts["skipped:legacy_event_row"] += 1
                    return
                # The per-tenant list is a snapshot; decide on the row as it is now.
                fresh = await _queue_row(customer_id, source, event_id)
                if fresh is None:
                    counts["gone"] += 1
                    return
                if fresh["status"] in _BUSY:
                    # The worker is about to run it, and writes its own if it ends.
                    counts["busy"] += 1
                    return
                bucket = await store.bucket_for(customer_id)
                # Before the read: a rerun must not download every batch of every
                # session it already wrote.
                key = trajectory_key(source, customer_id, event_id)
                if not args.force and await store.exists(bucket, key):
                    counts["already_present"] += 1
                    return
                if await _deleted(customer_id, source, event_id):
                    counts["deleted"] += 1
                    return
                hydrated = await _read_session(normalizer, store, {**row, **fresh})
                if (hydrated.get("session_id") or event_id) != event_id:
                    counts["skipped:id_mismatch"] += 1
                    return
                session_id = event_id
                if not hydrated.get("session_complete"):
                    # A running session's copy is the worker's (live, then final):
                    # the backfill writes only final copies of ended sessions.
                    counts["not_ended"] += 1
                    return
                events = list(hydrated.get("events") or [])
                fragments = hydrated.get("fragments")
                del hydrated
                if fragments is not None:
                    # Protocol 3: the worker folds the client's fragments, whatever
                    # SESSION_ATIF_BUILDER says (that picks protocol 2's builder).
                    if not fragments:
                        counts["no_fragments"] += 1
                        return
                    built = fold_fragments(list(fragments), session_id, source)
                else:
                    if not events:
                        counts["no_events"] += 1  # normalize() writes nothing for these either
                        return
                    built = build_trajectory(
                        events,
                        session_id=session_id,
                        agent_name=source,
                        builder=_settings().session_atif_builder,
                    )
                if not args.write:
                    counts["would_write"] += 1
                    _emit({"kind": "session", **ident, "would_write": True,
                           "steps": len(built.trajectory["steps"]), "unparsed": built.unparsed})
                    return
                # Only ended sessions are backfilled: this is the final copy.
                final = {**built.trajectory,
                         "extra": {**(built.trajectory.get("extra") or {}), "session_ended": True}}
                size, invalid = await write_trajectory(
                    store, bucket, key, final,
                    max_bytes=args.max_bytes or _settings().session_trajectory_max_bytes,
                )
                # Anything that overtook the write -- a new batch or end marker
                # (version), a deletion or purge (the row goes before their R2
                # sweep), a closed tenant -- may have run its own pass or sweep
                # already, so the object is removed: "not built", never stale.
                try:
                    after = await _queue_row(customer_id, source, event_id)
                    overtaken = (
                        after is None
                        or after["version"] != fresh["version"]
                        or await _deleted(customer_id, source, event_id)
                        or await refusal_for(customer_id) is not None
                    )
                except Exception as exc:
                    # Unchecked is treated as overtaken: a rerun rebuilds it.
                    overtaken = True
                    _emit({"kind": "session", **ident, "unverified": type(exc).__name__})
                if overtaken:
                    try:
                        await store.delete(bucket, key)
                    except Exception as exc:
                        counts["delete_failed"] += 1
                        _emit({"kind": "session", **ident, "delete_failed": type(exc).__name__,
                               "key": key})
                        return
                    counts["overtaken"] += 1
                    _emit({"kind": "session", **ident, "overtaken": True})
                    return
                counts["written"] += 1
                counts["invalid"] += 1 if invalid else 0
                _emit({"kind": "session", **ident, "written": size, "invalid": bool(invalid),
                       "unparsed": built.unparsed})
            except Skip as skip:
                counts[f"skipped:{skip}"] += 1
            except TrajectoryTooLarge:
                counts["skipped:too_large"] += 1
            except Exception as exc:
                counts[f"error:{type(exc).__name__}"] += 1
                _emit({"kind": "session", **ident, "error": type(exc).__name__})

    for customer_id in await _tenants(args.customer, args.all_tenants):
        if (refusal := await refusal_for(customer_id)) is not None:
            _emit({"kind": "tenant", "customer": customer_id, "skipped": refusal["status"]})
            continue
        rows = [r for r in await _queue_rows(customer_id, args.sources)
                if r["status"] not in _BUSY]
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
            p.add_argument(
                "--compare-builders",
                action="store_true",
                help="also build with both builders (reference, fold) and compare "
                "them with each other and with the stored final trajectory",
            )
        else:
            p.add_argument("--write", action="store_true")
            p.add_argument("--force", action="store_true",
                           help="rewrite trajectories that already exist")
            p.add_argument("--concurrency", type=int, default=4)
            p.add_argument("--max-bytes", type=int, default=None,
                           help="skip larger trajectories (default: SESSION_TRAJECTORY_MAX_BYTES)")
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

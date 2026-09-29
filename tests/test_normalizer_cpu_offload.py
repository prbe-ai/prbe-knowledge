"""A large session's chunk plan must not stall the worker's event loop.

The worker runs every claim loop on one event loop. On 2026-09-29 a 5.5 MB
live session (probe tenant, 21.5k events) re-planned on every append took
~50 s, and ~26 s of that was the pure-Python credential scrub. It ran in
`asyncio.to_thread`, but it held the GIL, so the loop waited behind it: 597 ms
max heartbeat lag, 200 loopback round trips in 5.6 s instead of 5 ms, and
one-document ingests on the same pod at 11-53 s instead of ~0.7 s. The scrub
now runs in `cpu_pool`'s processes (engine/ingest/cpu_pool.py).

Two properties, both on the REAL scrubber and chunker over a synthetic session
built to take the same slow path as the real one (the whole-value fallback,
then line by line). Only the collaborators the change does not touch are
stubbed: the one live-chunks SELECT, the embedder, and the gitleaks backstop
(tests/test_redactd.py covers that one against the real daemon).

  1. The offloaded plan is the inline plan, byte for byte.
  2. While it is computed, the loop keeps answering: a heartbeat's max lag
     stays under 200 ms and small loopback round trips stay fast.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import statistics
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

import engine.ingest.normalizer as norm_mod
from engine.ingest import cpu_pool, payload_redaction, secret_redaction
from engine.ingest.handlers.base import ConnectorContext
from engine.shared.config import get_settings
from engine.shared.constants import (
    CHUNKER_VERSION,
    DocClass,
    DocType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.embeddings import EmbeddedChunk, EmbedResult
from engine.shared.models import ACLPrincipal, ACLSnapshot, Document

#: A value only the stub scanner reports (the stdlib pass does not know it), so
#: the replace-a-finding step of the gitleaks pass runs in both modes.
_PLANTED = "zq-planted-value-0042"


def _synthetic_session(lines: int) -> str:
    """Transcript-shaped text that takes the scrubber's SLOW path, as the real
    5.5 MB session did: `%XX` / `\\uXXXX` escapes make the whole-value scrub
    give up and re-scrub line by line. ~4.7 us of scrub per byte, the same rate
    measured on the real session."""
    out = []
    for i in range(lines):
        h = hashlib.sha256(str(i).encode()).hexdigest()
        k = i % 6
        if k == 0:
            out.append(f"user: check run {h[:12]} at https://example.com/runs/{h[:8]}?q=a%2Fb")
        elif k == 1:
            out.append(f'assistant: {{"tool": "Bash", "input": {{"command": "grep -rn t_{i} src/"}}}}')
        elif k == 2:
            out.append(f"tool_result: /home/dev/src/module_{i}.py:{i % 400}: def h_{i}(e):  # {h}")
        elif k == 3:
            out.append(f"assistant: timeout={i % 90}s retries={i % 5}; caf\\u00e9 C:\\\\Users\\\\dev")
        elif k == 4:
            out.append(f"tool_result: echo ${{HOME}}/cache/{h[:16]} password=<your-password>")
        else:
            out.append(f"user: merge it. blob {h[:40]}== version v{i % 9}.{i % 7}.{i % 11}")
    out.insert(lines // 2, f"tool_result: marker {_PLANTED} end")
    return "\n".join(out)


def _session_doc(body: str) -> Document:
    now = datetime(2026, 9, 29, tzinfo=UTC)
    return Document(
        doc_id="claude_code:c1:synthetic-session",
        customer_id="c1",
        source_system=SourceSystem.CLAUDE_CODE,
        source_id="synthetic-session",
        source_url="https://prbe.ai/dashboard/agent-sessions/synthetic-session",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.CLAUDE_CODE_SESSION,
        content_type="application/json",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        title="Claude Code session synthet",
        body_preview=body[:200],
        body_size_bytes=len(body.encode()),
        body_token_count=0,
        author_id="emp-1",
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        metadata={"agent": "claude_code", "cwd": "/home/dev", "event_count": 1},
        body=body,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id="c1",
                    permission=Permission.READ,
                )
            ],
            captured_at=now,
        ),
    )


class _ZeroEmbedder:
    """The embedder's real result shape, no network."""

    async def embed_documents(self, items):
        return EmbedResult(
            embedded=[EmbeddedChunk(chunk_index=i, embedding=[0.0] * 4) for i in range(len(items))],
            failed=[],
        )


@pytest.fixture
async def planner(monkeypatch):
    """A real Normalizer whose only fakes are the live-chunks SELECT, the
    embedder and the gitleaks backstop. `live_rows` is what the SELECT returns."""
    live_rows: list[dict[str, Any]] = []

    class _Conn:
        async def fetch(self, *_args, **_kwargs):
            return list(live_rows)

    @contextlib.asynccontextmanager
    async def _with_tenant(_customer_id):
        yield _Conn()

    def _find_secrets(text: str) -> list[tuple[str, str, int]]:
        return [("synthetic-rule", _PLANTED, 1)] if _PLANTED in text else []

    monkeypatch.setattr(norm_mod, "with_tenant", _with_tenant)
    monkeypatch.setattr(secret_redaction, "find_secrets", _find_secrets)
    async with httpx.AsyncClient() as http:
        ctx = ConnectorContext(settings=get_settings(), http=http)
        normalizer = norm_mod.Normalizer(ctx, store=object(), embedder=_ZeroEmbedder())
        try:
            yield normalizer, live_rows
        finally:
            cpu_pool.shutdown()


def _pool(monkeypatch, *, workers: int, min_chars: int) -> None:
    cpu_pool.shutdown()
    monkeypatch.setattr(get_settings(), "ingest_cpu_pool_workers", workers)
    monkeypatch.setattr(get_settings(), "ingest_cpu_pool_min_chars", min_chars)


def _snapshot(doc: Document, plan: Any) -> dict[str, Any]:
    return {
        "body": doc.body,
        "title": doc.title,
        "body_preview": doc.body_preview,
        "metadata": doc.metadata,
        "reused": sorted(plan.reused_content_hashes),
        "reused_metadata": plan.reused_metadata_hash,
        "removed": sorted(plan.removed_hashes),
        "added": [
            (p.chunk_index, p.content, p.token_count, kind, vec)
            for p, vec, kind in plan.added_pieces
        ],
        "failed": plan.failed_pieces,
        "counts": (plan.added_count, plan.reused_count, plan.removed_count,
                   plan.failed_count, plan.live_count),
    }


async def test_offloaded_plan_is_identical_to_the_inline_plan(planner, monkeypatch) -> None:
    normalizer, live_rows = planner
    body = _synthetic_session(1500)

    # A prior version whose live chunks cover every other chunk of this one, plus
    # one that no longer exists: the plan then reuses, adds AND removes.
    _pool(monkeypatch, workers=0, min_chars=0)
    seed = await normalizer._plan_chunks("c1", _session_doc(body))
    live_rows.extend(
        {"content_hash": norm_mod._chunk_hash(p.content), "chunker_version": CHUNKER_VERSION,
         "chunk_index": p.chunk_index, "kind": "content"}
        for p, _, kind in seed.added_pieces[::2] if kind == "content"
    )
    live_rows.append({"content_hash": "0" * 64, "chunker_version": CHUNKER_VERSION,
                      "chunk_index": 9999, "kind": "content"})

    inline_doc = _session_doc(body)
    inline = _snapshot(inline_doc, await normalizer._plan_chunks("c1", inline_doc))

    _pool(monkeypatch, workers=2, min_chars=0)
    pooled_doc = _session_doc(body)
    pooled = _snapshot(pooled_doc, await normalizer._plan_chunks("c1", pooled_doc))

    # The pool really ran it: a process was started for the scrub.
    assert cpu_pool._pool is not None and cpu_pool._pool._processes
    assert pooled == inline
    # And the plan is the interesting one, not an empty or all-reused one.
    assert inline["reused"] and inline["added"] and inline["removed"]
    # Both passes ran: the stdlib scrub (the slow, offloaded one) and the
    # gitleaks backstop's replacement after it.
    assert "password=<redacted>" in inline["body"]
    assert _PLANTED not in inline["body"]
    assert "<redacted:synthetic-rule>" in inline["body"]
    assert not any(_PLANTED in content for _, content, *_ in inline["added"])


async def _heartbeat(stop: asyncio.Event, lags: list[float]) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        start = loop.time()
        await asyncio.sleep(0.01)
        lags.append(loop.time() - start - 0.01)


async def _round_trips(stop: asyncio.Event, batches: list[float], rounds: int = 50) -> None:
    """Batches of tiny loopback round trips: the shape of a small ingest's
    asyncpg traffic, which the GIL hand-off hurts far more than one sleeper."""

    async def echo(reader, writer):
        while data := await reader.read(64):
            writer.write(data)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", server.sockets[0].getsockname()[1]
    )
    try:
        while not stop.is_set():
            start = time.perf_counter()
            for _ in range(rounds):
                writer.write(b"x")
                await writer.drain()
                await reader.read(64)
            batches.append(time.perf_counter() - start)
            await asyncio.sleep(0.02)
    finally:
        writer.close()
        server.close()


async def _plan_under_watch(normalizer) -> tuple[float, list[float], list[float]]:
    stop = asyncio.Event()
    lags: list[float] = []
    batches: list[float] = []
    watchers = [
        asyncio.create_task(_heartbeat(stop, lags)),
        asyncio.create_task(_round_trips(stop, batches)),
    ]
    await asyncio.sleep(0.1)
    start = time.perf_counter()
    try:
        await normalizer._plan_chunks("c1", _session_doc(_synthetic_session(4000)))
    finally:
        wall = time.perf_counter() - start
        stop.set()
        await asyncio.gather(*watchers)
    return wall, lags, batches


async def test_event_loop_stays_responsive_while_a_large_session_is_planned(
    planner, monkeypatch
) -> None:
    normalizer, _ = planner
    # Production defaults: this body (~440 KB) is far over the pool threshold.
    _pool(monkeypatch, workers=2, min_chars=32_768)
    await cpu_pool.run_cpu(payload_redaction._scrub_free_texts, ["warm"], size=1 << 20)

    wall, lags, batches = await _plan_under_watch(normalizer)

    assert wall > 1.0, f"plan took {wall:.2f}s: too small to prove anything"
    assert len(batches) >= 5, "the round trips never ran while the plan did"
    assert max(lags) < 0.2, f"event loop stalled {max(lags) * 1000:.0f} ms"
    # The guard that fails without the pool. The heartbeat above barely moves
    # at this size (16 ms with the scrub on a thread: its longest single regex
    # call is short); what the GIL costs is every hand-off, and a burst of
    # round trips pays one per wait -- 715 ms for 50 with the scrub on a
    # thread, ~5 ms in the pool, ~1 ms idle.
    assert max(batches) < 0.2, (
        f"50 loopback round trips took {max(batches) * 1000:.0f} ms at worst "
        f"(median {statistics.median(batches) * 1000:.0f} ms)"
    )

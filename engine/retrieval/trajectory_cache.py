"""Per-process cache of stored session trajectories, for paged GET /trajectory.

WHY: readers page a trajectory (the dashboard's Steps tab, research-os's intent
reader and `probe session export` walk every page), and each page used to GET
the whole `trajectory.json` from R2 (up to SESSION_TRAJECTORY_MAX_BYTES, 8 MB),
parse it, strip it, size every step of the page with stdlib json and hand the
page to FastAPI's encoder -- to serve one slice. Walking a session cost
O(pages x session size): the 8 pages of a 7.6 MB session moved 61 MB out of
the store and spent ~45 ms of CPU each, measured locally.

WHAT IS HELD: the document as the reader gets it (render provenance stripped),
already serialized: every step's compact JSON in one `bytes`, with the offset
where each step starts, plus the members around `steps`. A page is one slice
of it, so a cached page parses and re-encodes nothing, and an entry's memory
is the bytes it holds (7.66 MB for a 7.63 MB object, measured with
tracemalloc), which is what lets the bound be a byte bound. A parsed dict
would hold ~2x the object and could only be guessed at; keeping orjson's
per-step outputs would hold ~6x (its output buffers are over-allocated), so
nothing it returns is kept as is.

FRESHNESS: every request revalidates with a conditional GET carrying the
cached ETag (ObjectStore.get_if_changed). R2 answers 304 with no body when the
object is unchanged, or the new body when the worker rewrote it, or 404 when
it was deleted (a resume, a failed write, a session deletion, a tenant
purge). A live session's trajectory is rewritten in place about once a
minute, so no time-based rule could be right; the object's own ETag is the
only signal that changes exactly when the bytes do. A stale page is never
served -- the cost is one round trip per page, without the body.

TENANCY: the key is (customer_id, source, session_id), from the caller's
authenticated tenant and the document row read under that tenant's RLS. The
customer is in the key even though the object path also names it, as in
reassembly_cache.py: a cross-tenant hit would be an RLS bypass through a cache.

BOUND: TRAJECTORY_CACHE_MAX_BYTES, default 16 MiB PER PROCESS. The retrieval
pod runs 4 uvicorn processes (the image's `RETRIEVAL_WORKERS:-4`; the chart
sets none), each with its own cache, so the pod-level cost is 64 MiB. That pod
has a 2 GiB limit and ran ~1.1 GiB steady (2026-08-18); the reassembly cache
may add 128 MiB per process, and /retrieve bursts once OOMKilled it at 1 GiB.
16 MiB holds 2 of the largest trajectories a pass may write (8 MB) or ~50 at
the measured p95 (326 KB); a walk needs only the session it is paging. A
trajectory larger than the whole budget is served uncached, and 0 turns the
cache off. Separate processes (and pods) do not share entries: each one's
first read of a version is a miss.
"""

from __future__ import annotations

import asyncio
import sys
from array import array
from dataclasses import dataclass
from typing import Any

import orjson

from engine.ingest.atif.store import strip_provenance
from engine.retrieval.byte_lru import ByteBudgetLRU
from engine.shared.constants import _env_int
from engine.shared.exceptions import StorageNotFound
from engine.shared.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MAX_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_ENTRIES = 256

TRAJECTORY_CACHE_MAX_BYTES = max(0, _env_int("TRAJECTORY_CACHE_MAX_BYTES", _DEFAULT_MAX_BYTES))
TRAJECTORY_CACHE_MAX_ENTRIES = max(
    1, _env_int("TRAJECTORY_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES)
)

#: (customer_id, source_system, session_id)
CacheKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class TrajectoryPages:
    """One stored trajectory, provenance stripped, serialized once.

    `steps` is every step's compact JSON joined by commas -- the inside of the
    document's `steps` array -- and `bounds[i]` is where step i starts in it
    (`bounds[-1]` is one past the end plus the comma that is not there), so a
    page is ONE slice of `steps`. `head` and `tail` are the members before and
    after `steps`, serialized without their braces, so a page keeps the
    document's member order.
    """

    etag: str | None
    head: bytes
    tail: bytes
    steps: bytes
    bounds: array[int]

    @classmethod
    def from_body(cls, body: bytes, etag: str | None) -> TrajectoryPages | None:
        """None when the body is not a JSON object (the reader's `not_built`)."""
        try:
            value = orjson.loads(body)
        except orjson.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        # Documents stored before provenance was stripped at write time still carry it.
        public = strip_provenance(value)
        names = list(public)
        at = names.index("steps")
        # Slicing copies: orjson's own output buffer is over-allocated (16 KiB
        # held for a 2 KB step, measured), so nothing it returns is kept as is.
        head = orjson.dumps({k: public[k] for k in names[:at]})[1:-1]
        tail = orjson.dumps({k: public[k] for k in names[at + 1 :]})[1:-1]
        joined = bytearray()
        bounds = array("q")
        for i, step in enumerate(public["steps"]):
            if i:
                joined += b","
            bounds.append(len(joined))
            joined += orjson.dumps(step)
        bounds.append(len(joined) + 1)
        return cls(etag=etag, head=head, tail=tail, steps=bytes(joined), bounds=bounds)

    @property
    def total_steps(self) -> int:
        return len(self.bounds) - 1

    @property
    def nbytes(self) -> int:
        """What the entry holds in memory (`sys.getsizeof`: exact for bytes and arrays)."""
        return (
            sys.getsizeof(self.head)
            + sys.getsizeof(self.tail)
            + sys.getsizeof(self.steps)
            + sys.getsizeof(self.bounds)
            + len(self.etag or "")
        )

    def page(self, step_from: int, step_limit: int, max_bytes: int) -> tuple[bytes, int]:
        """The trajectory JSON holding steps `step_from`.. (1-based), at most
        `step_limit` of them, stopping before the step whose compact JSON would
        pass `max_bytes` (a page always holds at least one step if any remain).
        Returns (the document's bytes, how many steps it holds)."""
        first = min(step_from - 1, self.total_steps)
        last = first
        budget = max_bytes
        for i in range(first, min(first + step_limit, self.total_steps)):
            size = self.bounds[i + 1] - self.bounds[i] - 1
            if last > first and size > budget:
                break
            last = i + 1
            budget -= size
        array_body = self.steps[self.bounds[first] : self.bounds[last] - 1] if last > first else b""
        members = [m for m in (self.head, b'"steps":[' + array_body + b"]", self.tail) if m]
        return b"{" + b",".join(members) + b"}", last - first


trajectory_cache: ByteBudgetLRU[CacheKey, TrajectoryPages] = ByteBudgetLRU(
    TRAJECTORY_CACHE_MAX_BYTES, TRAJECTORY_CACHE_MAX_ENTRIES
)


async def load_trajectory_pages(
    store: Any,
    bucket: str,
    key: str,
    cache_key: CacheKey,
    cache: ByteBudgetLRU[CacheKey, TrajectoryPages] = trajectory_cache,
) -> tuple[TrajectoryPages | None, bool]:
    """The session's current stored trajectory, from cache when R2 says the
    cached copy is still the object (304). Returns (pages or None when there
    is none or it is unreadable, whether the cache served it).

    Raises what the store raises other than StorageNotFound (an R2 outage is
    the caller's 5xx, as it was before the cache).
    """
    cached = cache.get(cache_key)
    try:
        read = await store.get_if_changed(bucket, key, cached.etag if cached else None)
    except StorageNotFound:
        cache.pop(cache_key)
        return None, False
    if read.body is None:
        if cached is not None:
            return cached, True
        # A 304 for a copy we do not hold cannot happen (no ETag was sent);
        # never serve on it.
        cache.pop(cache_key)
        return None, False
    pages = await asyncio.to_thread(TrajectoryPages.from_body, read.body, read.etag)
    if pages is None:
        log.warning("trajectory.unreadable", key=key, bytes=len(read.body))
        cache.pop(cache_key)
        return None, False
    if pages.etag:
        # No ETag, no way to revalidate: such a copy is served, never cached.
        cache.put(cache_key, pages, pages.nbytes)
    else:
        cache.pop(cache_key)
    return pages, False

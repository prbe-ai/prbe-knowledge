"""trajectory_cache: a page is never older than the stored object.

The cache holds a session's stored trajectory, serialized per step, and
revalidates it on every read with a conditional GET. These tests pin what that
promises: a repeat read transfers no body; a rewritten object (a live session
growing) or a removed one is never served from cache; the byte bound evicts;
and one tenant's entry never answers another tenant's read.
"""

from __future__ import annotations

import gc
import hashlib
import tracemalloc
from typing import Any

import orjson
import pytest

from engine.ingest.atif.build import build_trajectory
from engine.ingest.atif.store import strip_provenance
from engine.retrieval.byte_lru import ByteBudgetLRU
from engine.retrieval.trajectory_cache import (
    TrajectoryPages,
    load_trajectory_pages,
)
from engine.shared.storage import ObjectRead, StorageNotFound

SESSION = "11111111-2222-3333-4444-555555555555"
KEY = f"raw/claude_code/cust-a/{SESSION}/trajectory.json"


class FakeStore:
    """An object store with S3's conditional GET: the ETag is the body's MD5."""

    def __init__(self, *, etags: bool = True) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.etags = etags
        #: (key, the ETag the caller sent, body bytes transferred)
        self.reads: list[tuple[str, str | None, int]] = []

    async def get_if_changed(self, bucket: str, key: str, etag: str | None) -> ObjectRead:
        try:
            body = self.objects[(bucket, key)]
        except KeyError:
            self.reads.append((key, etag, 0))
            raise StorageNotFound(key) from None
        current = f'"{hashlib.md5(body).hexdigest()}"' if self.etags else None
        if etag is not None and etag == current:
            self.reads.append((key, etag, 0))
            return ObjectRead(body=None, etag=current)
        self.reads.append((key, etag, len(body)))
        return ObjectRead(body=body, etag=current)


def _trajectory(n_users: int, text: str = "turn") -> dict[str, Any]:
    events = [{"line_no": i, "raw": {"type": "user", "message": {"content": f"{text} {i}"}}}
              for i in range(n_users)]
    return build_trajectory(events, session_id=SESSION, agent_name="claude_code").trajectory


def _with_provenance(n_users: int, text: str = "turn") -> bytes:
    """As stored before provenance was stripped at write time: the cache strips it."""
    doc = _trajectory(n_users, text)
    doc.setdefault("extra", {})["probe"] = {"render": "x"}
    doc["steps"][0].setdefault("extra", {})["probe_parts"] = [1]
    return orjson.dumps(doc)


def _page(pages: TrajectoryPages, step_from: int = 1, limit: int = 1000,
          max_bytes: int = 10**9) -> dict[str, Any]:
    body, _ = pages.page(step_from, limit, max_bytes)
    return orjson.loads(body)


@pytest.fixture
def cache() -> ByteBudgetLRU:
    return ByteBudgetLRU(max_bytes=10**9, max_entries=100)


# --- what a page holds -----------------------------------------------------


def test_a_page_is_the_stripped_document_sliced_in_member_order() -> None:
    body = _with_provenance(5)
    pages = TrajectoryPages.from_body(body, '"e"')
    assert pages is not None
    expected = strip_provenance(orjson.loads(body))
    expected["steps"] = expected["steps"][1:3]
    got = _page(pages, step_from=2, limit=2)
    assert got == expected
    assert list(got) == list(expected), "the document keeps its member order"
    assert "probe" not in (got.get("extra") or {})
    assert all("probe_parts" not in (s.get("extra") or {}) for s in got["steps"])


def test_a_page_stops_before_the_step_that_passes_the_byte_budget() -> None:
    doc = _trajectory(4)
    pages = TrajectoryPages.from_body(orjson.dumps(doc), '"e"')
    assert pages is not None
    # The budget counts each step's compact UTF-8 JSON, as before the cache.
    one = len(orjson.dumps(strip_provenance(doc)["steps"][0]))
    _, shown = pages.page(1, 4, one * 2 + 1)
    assert shown == 2
    # A page always carries at least one step, however large.
    _, shown = pages.page(1, 4, 1)
    assert shown == 1
    # Past the end: an empty page, still a valid document.
    body, shown = pages.page(99, 4, 10**9)
    assert shown == 0 and orjson.loads(body)["steps"] == []


@pytest.mark.parametrize("body", [b"{oops", b"[1, 2]", b'"text"'])
def test_an_unreadable_body_has_no_pages(body: bytes) -> None:
    assert TrajectoryPages.from_body(body, '"e"') is None


# --- hit, miss, freshness --------------------------------------------------


async def test_a_repeat_read_is_a_304_that_transfers_no_body(cache) -> None:
    store = FakeStore()
    store.objects[("b", KEY)] = orjson.dumps(_trajectory(5))
    first, hit1 = await load_trajectory_pages(store, "b", KEY, ("cust-a", "claude_code", SESSION),
                                              cache)
    second, hit2 = await load_trajectory_pages(store, "b", KEY,
                                               ("cust-a", "claude_code", SESSION), cache)
    assert (hit1, hit2) == (False, True)
    assert second is first
    assert store.reads[0][1:] == (None, len(store.objects[("b", KEY)]))
    assert store.reads[1] == (KEY, first.etag, 0), "revalidated, nothing transferred"


async def test_a_growing_live_session_is_never_served_stale(cache) -> None:
    """The worker rewrites a running session's trajectory in place about once
    a minute; the read after a rewrite must see the new document."""
    store = FakeStore()
    ck = ("cust-a", "claude_code", SESSION)
    store.objects[("b", KEY)] = orjson.dumps(_trajectory(3))
    old, _ = await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert old.total_steps == 3

    store.objects[("b", KEY)] = orjson.dumps(_trajectory(5))  # two more turns
    new, hit = await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert hit is False
    assert new.total_steps == 5
    assert [s["message"] for s in _page(new, step_from=4)["steps"]] == ["turn 3", "turn 4"]
    # And the next read is a hit on the NEW version.
    again, hit = await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert hit is True and again is new

    # Same step count, different content (a rewrite, not an append) misses too.
    store.objects[("b", KEY)] = orjson.dumps(_trajectory(5, text="edited"))
    edited, hit = await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert hit is False and _page(edited)["steps"][0]["message"] == "edited 0"


@pytest.mark.parametrize("replacement", [None, b"{oops"])
async def test_a_removed_or_unreadable_object_drops_the_entry(cache, replacement) -> None:
    """A resume, a failed write or a deletion removes trajectory.json; the
    reader must get `not_built`, never the copy it read before."""
    store = FakeStore()
    ck = ("cust-a", "claude_code", SESSION)
    store.objects[("b", KEY)] = orjson.dumps(_trajectory(3))
    await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert len(cache) == 1
    if replacement is None:
        del store.objects[("b", KEY)]
    else:
        store.objects[("b", KEY)] = replacement
    pages, hit = await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert (pages, hit) == (None, False)
    assert len(cache) == 0 and cache.total_bytes == 0


async def test_an_object_without_an_etag_is_served_but_never_cached(cache) -> None:
    store = FakeStore(etags=False)
    store.objects[("b", KEY)] = orjson.dumps(_trajectory(3))
    ck = ("cust-a", "claude_code", SESSION)
    first, _ = await load_trajectory_pages(store, "b", KEY, ck, cache)
    second, hit = await load_trajectory_pages(store, "b", KEY, ck, cache)
    assert first is not None and second is not None and hit is False
    assert len(cache) == 0
    assert [r[1] for r in store.reads] == [None, None]


# --- the byte bound ---------------------------------------------------------


async def test_the_byte_bound_evicts_the_least_recently_read() -> None:
    store = FakeStore()
    sessions = [f"s{i}" for i in range(3)]
    for s in sessions:
        store.objects[("b", s)] = orjson.dumps(_trajectory(20))
    size = TrajectoryPages.from_body(store.objects[("b", "s0")], '"e"').nbytes
    cache = ByteBudgetLRU(max_bytes=size * 2 + size // 2, max_entries=100)
    for s in sessions[:2]:
        await load_trajectory_pages(store, "b", s, ("c", "claude_code", s), cache)
    await load_trajectory_pages(store, "b", "s0", ("c", "claude_code", "s0"), cache)  # touch s0
    await load_trajectory_pages(store, "b", "s2", ("c", "claude_code", "s2"), cache)  # evicts s1
    assert len(cache) == 2 and cache.total_bytes <= size * 2 + size // 2
    _, hit = await load_trajectory_pages(store, "b", "s1", ("c", "claude_code", "s1"), cache)
    assert hit is False, "s1 was evicted"


async def test_a_trajectory_larger_than_the_budget_is_served_uncached() -> None:
    store = FakeStore()
    store.objects[("b", "small")] = orjson.dumps(_trajectory(1))
    store.objects[("b", "big")] = orjson.dumps(_trajectory(200))
    small = TrajectoryPages.from_body(store.objects[("b", "small")], '"e"').nbytes
    cache = ByteBudgetLRU(max_bytes=small * 3, max_entries=100)
    await load_trajectory_pages(store, "b", "small", ("c", "x", "small"), cache)
    big, _ = await load_trajectory_pages(store, "b", "big", ("c", "x", "big"), cache)
    assert big is not None and big.total_steps == 200
    assert len(cache) == 1, "the small entry survives; the big one is not held"
    _, hit = await load_trajectory_pages(store, "b", "small", ("c", "x", "small"), cache)
    assert hit is True


def test_nbytes_is_the_memory_the_entry_holds() -> None:
    """The bound is only a bound if nbytes is what an entry really costs.
    orjson's outputs are over-allocated (16 KiB for a 2 KB step), so an entry
    that kept them would hold several times what it counted."""
    body = orjson.dumps(_trajectory(300, text="x" * 2000))
    gc.collect()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        pages = TrajectoryPages.from_body(body, '"e"')
        gc.collect()
        held, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pages is not None
    assert abs((held - before) - pages.nbytes) < 0.05 * pages.nbytes
    assert pages.nbytes < 1.1 * len(body)


# --- tenancy ---------------------------------------------------------------


async def test_tenant_b_never_gets_tenant_a_entry(cache) -> None:
    """Same source, same session id, two tenants. Tenant A's cached copy must
    not answer tenant B's read: not when B has no trajectory, not when B's
    differs, and B's read must not even revalidate against A's ETag."""
    store = FakeStore()
    key_a = "raw/claude_code/tenant-a/s/trajectory.json"
    key_b = "raw/claude_code/tenant-b/s/trajectory.json"
    store.objects[("bucket-a", key_a)] = orjson.dumps(_trajectory(3, text="tenant a"))
    a, _ = await load_trajectory_pages(store, "bucket-a", key_a,
                                       ("tenant-a", "claude_code", SESSION), cache)

    b, hit = await load_trajectory_pages(store, "bucket-b", key_b,
                                         ("tenant-b", "claude_code", SESSION), cache)
    assert (b, hit) == (None, False), "B has no trajectory: not_built, never A's"
    assert store.reads[-1] == (key_b, None, 0), "B's read carried no ETag of A's"

    store.objects[("bucket-b", key_b)] = orjson.dumps(_trajectory(3, text="tenant b"))
    b, hit = await load_trajectory_pages(store, "bucket-b", key_b,
                                         ("tenant-b", "claude_code", SESSION), cache)
    assert hit is False
    assert _page(b)["steps"][0]["message"] == "tenant b 0"
    assert store.reads[-1][1] is None

    a_again, hit = await load_trajectory_pages(store, "bucket-a", key_a,
                                               ("tenant-a", "claude_code", SESSION), cache)
    assert hit is True and a_again is a
    assert _page(a_again)["steps"][0]["message"] == "tenant a 0"

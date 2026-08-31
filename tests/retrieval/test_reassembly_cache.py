"""ReassemblyCache: the fingerprint IS the correctness argument.

Versions are not immutable here (live sessions mutate membership in place),
so nothing may be served from cache unless the loaded chunk set is
byte-for-byte the set the cached text was stitched from. These tests pin
that: membership changes miss, order changes miss, tenants never share, and
the byte bound actually evicts.
"""

from __future__ import annotations

from engine.retrieval.reassembly_cache import (
    ReassemblyCache,
    chunk_set_fingerprint,
)


def _rows(*pairs: tuple[int, str]) -> list[dict]:
    return [{"chunk_index": i, "content_hash": cid} for i, cid in pairs]


def test_same_set_hits_and_changed_membership_misses() -> None:
    cache = ReassemblyCache(max_bytes=1_000_000)
    fp1 = chunk_set_fingerprint(_rows((0, "a"), (1, "b")))
    cache.put("cust", "doc", fp1, "stitched", [])
    assert cache.get("cust", "doc", fp1) == ("stitched", [])

    # An appended chunk (the live-session resync) changes the fingerprint.
    fp2 = chunk_set_fingerprint(_rows((0, "a"), (1, "b"), (2, "c")))
    assert fp2 != fp1
    assert cache.get("cust", "doc", fp2) is None
    # A removed-and-reordered set does too -- order feeds the stitcher.
    fp3 = chunk_set_fingerprint(_rows((0, "b"), (1, "a")))
    assert fp3 != fp1
    assert cache.get("cust", "doc", fp3) is None


def test_tenants_never_share_an_entry() -> None:
    """Same doc_id string, same chunk set, different tenant: a hit here
    would be an RLS bypass through a cache."""
    cache = ReassemblyCache(max_bytes=1_000_000)
    fp = chunk_set_fingerprint(_rows((0, "a")))
    cache.put("tenant-one", "slack:T1:C1:x", fp, "tenant one text", [])
    assert cache.get("tenant-two", "slack:T1:C1:x", fp) is None


def test_byte_bound_evicts_least_recent() -> None:
    cache = ReassemblyCache(max_bytes=10, max_entries=100)
    fp_a = chunk_set_fingerprint(_rows((0, "a")))
    fp_b = chunk_set_fingerprint(_rows((0, "b")))
    fp_c = chunk_set_fingerprint(_rows((0, "c")))
    cache.put("c", "doc-a", fp_a, "aaaa", [])  # 4 bytes
    cache.put("c", "doc-b", fp_b, "bbbb", [])  # 8 total
    cache.get("c", "doc-a", fp_a)  # touch a -> b is now least recent
    cache.put("c", "doc-c", fp_c, "cccc", [])  # 12 -> evict b
    assert cache.get("c", "doc-b", fp_b) is None
    assert cache.get("c", "doc-a", fp_a) is not None
    assert cache.get("c", "doc-c", fp_c) is not None
    assert cache.total_bytes <= 10


def test_oversized_document_is_served_uncached() -> None:
    """One doc bigger than the whole budget must not flush the cache to
    store itself."""
    cache = ReassemblyCache(max_bytes=10)
    fp_small = chunk_set_fingerprint(_rows((0, "s")))
    cache.put("c", "small", fp_small, "tiny", [])
    fp_big = chunk_set_fingerprint(_rows((0, "big")))
    cache.put("c", "big", fp_big, "x" * 1000, [])
    assert cache.get("c", "big", fp_big) is None
    assert cache.get("c", "small", fp_small) is not None


def test_reput_same_key_replaces_without_leaking_bytes() -> None:
    cache = ReassemblyCache(max_bytes=100)
    fp = chunk_set_fingerprint(_rows((0, "a")))
    cache.put("c", "doc", fp, "12345", [])
    cache.put("c", "doc", fp, "1234567890", [])
    assert cache.total_bytes == 10
    assert len(cache) == 1

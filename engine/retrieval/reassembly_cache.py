"""Byte-bounded cache for reassembled source-view text.

WHY: the first measurement `SourceViewResponse.timing_ms` ever took (2026-08-31)
found the get_source cost center in one shot -- a 479KB / 304-chunk session
transcript spent 422.5ms of a 447.9ms request in the threaded overlap-aware
reassembly, rebuilding all 8,003 lines to serve a 15-line tail. get_source runs
at ~13x /retrieve's call volume, and its dominant pattern is an agent paging
through ONE document with cursors -- every page re-stitching identical text.

WHY THE KEY IS A CHUNK-SET FINGERPRINT AND NOT (doc, version): versions are
NOT immutable here. Incomplete documents -- live agent sessions, the biggest
and hottest sources -- add and remove chunks at the SAME version on every
resync (the in-place branch in ingest/normalizer.py). A (doc, version) key
would serve stale text for exactly the documents that matter. The fingerprint
hashes the ordered (chunk_index, chunk_id) sequence of the rows the request
actually loaded: chunk CONTENT is immutable per chunk_id (rows are
content_hash-addressed; resurrects and edits change membership, never a
chunk's body), so same fingerprint == same stitched text, and any append,
removal, resurrection, or visibility flip changes membership and misses
honestly. Correctness does not depend on invalidation arriving from anywhere.

customer_id is IN the key even though doc_ids are namespaced per source:
nothing structural guarantees two tenants cannot carry the same doc_id string,
and a cross-tenant text hit would be an RLS bypass through a cache. Belt over
cleverness.

BOUNDS: LRU by total cached BYTES (the values are whole reassembled documents,
so entry counts mean nothing) with an entry cap as a backstop. The budget is
deliberately modest -- this cache rides inside the retrieval pod next to the
gatherer's working memory; evicting a transcript costs one 400ms restitch,
OOMing the pod costs every in-flight search.

CONCURRENCY: get/put run on the event loop thread only (the stitch itself runs
in a worker thread, but callers await it before putting), so plain dict
mutation is safe without locks. Two replicas each warm their own cache; that
is fine -- the win is repeat reads within a conversation, which pin to a pod
rarely but re-stitch cheaply on the other one at worst.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Any

from engine.shared.logging import get_logger

log = get_logger(__name__)

#: Total bytes of reassembled text the cache may hold per process.
#: Env-tunable for incident response; malformed values fall back (same
#: rationale as constants._env_int -- a typo'd kubectl set env must not
#: crashloop the fleet at import).
_DEFAULT_MAX_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_ENTRIES = 512


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


REASSEMBLY_CACHE_MAX_BYTES = max(
    0, _env_int("SOURCE_REASSEMBLY_CACHE_MAX_BYTES", _DEFAULT_MAX_BYTES)
)
REASSEMBLY_CACHE_MAX_ENTRIES = max(
    1, _env_int("SOURCE_REASSEMBLY_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES)
)


def chunk_set_fingerprint(chunk_rows: list[Any]) -> str:
    """Hash the ordered chunk membership of one loaded document.

    (chunk_index, content_hash) pairs, in load order. content_hash IS the
    identity of what the stitcher consumes -- chunk bodies are
    content-hash-addressed and never mutate under a hash -- and chunk_index
    is included because ORDER matters: two sets with the same members in
    different positions must not collide. Any append, removal, resurrection
    or visibility flip changes this sequence and misses honestly.
    """
    h = hashlib.sha256()
    for row in chunk_rows:
        h.update(f"{row['chunk_index']}\x00{row['content_hash']}\x1f".encode())
    return h.hexdigest()


class ReassemblyCache:
    def __init__(
        self,
        max_bytes: int = REASSEMBLY_CACHE_MAX_BYTES,
        max_entries: int = REASSEMBLY_CACHE_MAX_ENTRIES,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        # spans are ChunkLineSpan objects, read-only downstream
        # (_chunk_line_offsets only zips over them), so sharing one list
        # across requests is safe.
        self._entries: OrderedDict[
            tuple[str, str, str], tuple[str, list[Any], int]
        ] = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(
        self, customer_id: str, doc_id: str, fingerprint: str
    ) -> tuple[str, list[Any]] | None:
        key = (customer_id, doc_id, fingerprint)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        content, spans, _ = entry
        return content, spans

    def put(
        self,
        customer_id: str,
        doc_id: str,
        fingerprint: str,
        content: str,
        spans: list[Any],
    ) -> None:
        nbytes = len(content)
        if nbytes > self._max_bytes:
            # A single document larger than the whole budget: caching it
            # would evict everything to hold one entry. Serve it uncached.
            return
        key = (customer_id, doc_id, fingerprint)
        old = self._entries.pop(key, None)
        if old is not None:
            self._total_bytes -= old[2]
        self._entries[key] = (content, spans, nbytes)
        self._total_bytes += nbytes
        while self._entries and (
            self._total_bytes > self._max_bytes
            or len(self._entries) > self._max_entries
        ):
            _, (_, _, evicted_bytes) = self._entries.popitem(last=False)
            self._total_bytes -= evicted_bytes

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._entries)


#: Process-wide instance, mirroring the module-level posture of the ANN
#: semaphore in retrievers/vector.py: the repeat-read pattern is
#: cross-request, so a per-request cache would cache nothing.
reassembly_cache = ReassemblyCache()

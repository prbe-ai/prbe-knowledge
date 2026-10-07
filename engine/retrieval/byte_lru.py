"""An LRU bounded by the total size of what it holds, not by how many entries.

The retrieval pod's per-process caches hold whole documents (a session
transcript's stitched text, a session's trajectory), so an entry count says
nothing about memory: one entry can be 10 bytes or 8 MB. Each `put` declares
its value's size, and the least recently used entries go until the total fits
the budget again. An entry cap stays as a backstop.

CONCURRENCY: callers use it from the event loop thread only (they await any
threaded work before `put`), so plain dict mutation needs no lock.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable


class ByteBudgetLRU[K: Hashable, V]:
    def __init__(self, max_bytes: int, max_entries: int) -> None:
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> V | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key: K, value: V, nbytes: int) -> bool:
        """Hold `value`; False when it alone is larger than the whole budget.

        Such a value is not stored: holding it would evict everything else to
        keep one entry. The caller serves it uncached.
        """
        self.pop(key)
        if nbytes > self._max_bytes:
            return False
        self._entries[key] = (value, nbytes)
        self._total_bytes += nbytes
        while self._entries and (
            self._total_bytes > self._max_bytes
            or len(self._entries) > self._max_entries
        ):
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self._total_bytes -= evicted_bytes
        return True

    def pop(self, key: K) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._total_bytes -= old[1]

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._entries)

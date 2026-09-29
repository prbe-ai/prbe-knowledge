"""A process pool for the ingestion worker's pure-Python CPU work.

WHY A PROCESS AND NOT A THREAD
------------------------------
The worker runs every claim loop on ONE event loop. The credential scrub
(`payload_redaction._scrub_free_texts`) is pure-Python regex over the whole
document: for a 5.5 MB live session (probe tenant, 2026-09-29, 21.5k events)
that is ~26 s of CPU, and a growing session repeats it on every append. It
already ran in `asyncio.to_thread`, which moved it off the loop in name only:
the thread holds the GIL, and one `re.sub` over a multi-MB string holds it for
the whole call without yielding. Measured on the real session: 597 ms max
event-loop lag, and 200 loopback round trips (the shape of a small asyncpg
ingest) taking 5.6 s instead of 5 ms. In the pod, unrelated one-document
ingests went from ~0.7 s to 11-53 s. A separate process has its own GIL.

`chunk_text` does NOT come here: its cost is tiktoken's encode, which releases
the GIL, so a thread is enough (and cheaper than pickling the pieces back).

WHAT STAYS OUT
--------------
- Small inputs (`ingest_cpu_pool_min_chars`): milliseconds of GIL on a thread,
  and routing them here would queue them behind a large session's scrub.
- The redactd scan: it waits on a socket with the GIL released, and it needs
  the parent's one supervised daemon, not one daemon per pool process.

FAILURE
-------
A process that dies mid-task (an OOM kill) breaks the whole executor, and every
later submit would fail too. The broken pool is dropped, the next call builds a
fresh one, and this call raises `CpuPoolUnavailable` (transient): the row
retries, it never proceeds without the scrub it asked for.

`ingest_cpu_pool_workers=0` turns the pool off: everything runs on a thread,
exactly as before this module existed.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from engine.shared.config import get_settings
from engine.shared.exceptions import CpuPoolUnavailable
from engine.shared.logging import get_logger

log = get_logger(__name__)

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _die_with_parent() -> None:
    """PR_SET_PDEATHSIG in each pool process: the worker leaves through
    `os._exit` on a fatal error (kb.worker._abort), which runs no executor
    shutdown, and a pool process blocked on its task queue would outlive it."""
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL


def _executor() -> ProcessPoolExecutor | None:
    global _pool
    workers = get_settings().ingest_cpu_pool_workers
    if workers <= 0:
        return None
    with _pool_lock:
        if _pool is None:
            # spawn, not fork: the worker has threads (the default executor,
            # redactd's supervisor), and a forked child inherits their locks
            # in whatever state they were in at the fork.
            _pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_die_with_parent,
            )
        return _pool


def _discard(pool: ProcessPoolExecutor) -> None:
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


async def run_cpu[T](fn: Callable[..., T], *args: Any, size: int) -> T:
    """`fn(*args)` off the event loop: in the pool when `size` (the input's
    character count) reaches `ingest_cpu_pool_min_chars`, on a thread
    otherwise. `fn` and its arguments must pickle (a module-level function
    over plain data)."""
    pool = _executor()
    if pool is None or size < get_settings().ingest_cpu_pool_min_chars:
        return await asyncio.to_thread(fn, *args)
    try:
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)
    except BrokenProcessPool as exc:
        _discard(pool)
        log.warning("cpu_pool.broken", fn=getattr(fn, "__qualname__", repr(fn)), size=size)
        raise CpuPoolUnavailable("a CPU pool process died mid-task", size=size) from exc


def shutdown() -> None:
    """Stop the pool, if one was started. Queued work is cancelled; a task
    already running in a process finishes there and its result is dropped."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)

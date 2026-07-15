"""Per-source poller ABC + registry.

Each source poller is a subclass of ``BasePoller`` that knows how to:

  1. Ask the upstream API "give me everything since ``cursor``".
  2. Extract documents from the raw response into a normalized shape
     the ingestion pipeline can consume.
  3. Compute the next cursor value to store.

The scheduler doesn't know anything source-specific — it walks the
``ingestion_cursors`` table, looks up the poller for each row's
``source`` value in this module's registry, calls it, and persists
the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from engine.shared.constants import SourceSystem


@dataclass
class PollResult:
    """What a poller hands back to the scheduler after one tick.

    ``documents`` — the records to enqueue for normalize + index. Each
    item is a dict in the same shape an inbound webhook would produce
    (so the downstream ingestion queue / normalizer doesn't need to
    branch on origin).

    ``next_cursor`` — the cursor value to persist for the next tick.
    ``None`` means "no change" (e.g. the upstream returned an empty
    page with no new sentinel); the scheduler keeps the existing
    cursor.

    ``error`` — set on a soft failure (e.g. a 429 we should back off
    on, or a 403 from an expired source token). The scheduler stamps
    it onto the cursor row and keeps walking; a NULL error clears the
    previous one.
    """

    documents: list[dict[str, Any]]
    next_cursor: str | None = None
    error: str | None = None


class BasePoller(ABC):
    """Per-source poller contract.

    Subclasses must implement ``poll`` (the one-tick fetch + parse). The
    scheduler calls poll() with the current cursor and applies the
    returned ``PollResult``.

    Subclasses MUST be safe to instantiate without arguments — the
    scheduler does ``PollerCls()`` on every tick. Move per-call state
    (auth tokens, HTTP client) into the ``poll`` method's parameters
    or load it from a (per-customer) Settings object inside poll().
    """

    # The SourceSystem this poller handles. Set on the subclass.
    source: SourceSystem

    @abstractmethod
    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        """One-tick fetch. The scheduler has already wrapped this call
        in ``with_tenant(customer_id)`` so any DB read this method does
        is RLS-gated to the right tenant.

        ``cursor`` is whatever the previous tick's ``PollResult.next_cursor``
        was, or ``None`` for the first tick on a fresh resource.
        Source-specific interpretation — an ISO timestamp for some
        sources, an ETag string for others.

        Implementations should be idempotent under repeated cursors —
        the scheduler may retry the same cursor on a transient
        scheduler crash before the cursor is persisted.
        """


# --- Registry ------------------------------------------------------------
#
# Per-source poller classes register themselves at module import time
# (see services/ingestion/polling/github.py etc. in PRs E1-E5). The
# scheduler resolves source -> class via get_poller().

_REGISTRY: dict[SourceSystem, type[BasePoller]] = {}


def register_poller(source: SourceSystem, poller_cls: type[BasePoller]) -> None:
    """Register a poller class for a SourceSystem. Idempotent — re-registering
    the same source with the same class is a no-op; re-registering with a
    different class raises (catches accidental double-registration)."""
    existing = _REGISTRY.get(source)
    if existing is poller_cls:
        return
    if existing is not None:
        raise RuntimeError(
            f"polling: source {source} already registered to {existing!r}; "
            f"refusing to re-register with {poller_cls!r}"
        )
    _REGISTRY[source] = poller_cls


def get_poller(source: SourceSystem) -> type[BasePoller] | None:
    """Look up a poller class. Returns None if no poller is registered for
    this source — the scheduler treats that as "skip this row" (some
    sources never poll, e.g. MANUAL_UPLOAD)."""
    return _REGISTRY.get(source)


def registered_sources() -> set[SourceSystem]:
    """Snapshot of sources with a registered poller. Used by the
    scheduler's startup log and by tests."""
    return set(_REGISTRY)

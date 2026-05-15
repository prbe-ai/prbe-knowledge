"""Pre-enqueue gate: is this (customer, source) still connected?

Closes the race where a webhook / backfill drain / worker callback inserts
into `ingestion_queue` between the moment a user clicks "disconnect" and the
moment all in-flight tasks notice the integration is gone. Without the gate,
those stragglers materialise documents for a source the user has already
disconnected — exactly the probe-founders/github incident on 2026-05-15.
"""

from __future__ import annotations

from shared.constants import IntegrationStatus, SourceSystem
from shared.db import raw_conn

# Sources whose lifecycle is keyed on an `integration_tokens` row. For these,
# disconnect = token row gone (or status flipped from 'active'), and enqueue
# must reject after that point.
#
# Other SourceSystem values use different lifecycles and are intentionally
# not gated here:
#   - CLAUDE_CODE / CODEX: agent sessions, no OAuth token
#   - MANUAL_UPLOAD / CUSTOM_INGEST: BYO upload paths, separate token tables
#   - WIKI: authored programmatically, no upstream
#   - CODE_GRAPH: derived from github content; its own enqueue path
#                 short-circuits when the upstream github source disappears
_OAUTH_SOURCES: frozenset[SourceSystem] = frozenset(
    {
        SourceSystem.SLACK,
        SourceSystem.LINEAR,
        SourceSystem.GITHUB,
        SourceSystem.NOTION,
        SourceSystem.SENTRY,
        SourceSystem.GRANOLA,
    }
)

# Sources that intentionally bypass the gate. Listed explicitly so adding
# a new SourceSystem forces a gate decision via the coverage test
# (`tests/test_enqueue_disconnect_gate.py::test_all_sources_classified`).
_UNGATED_SOURCES: frozenset[SourceSystem] = frozenset(
    {
        SourceSystem.CLAUDE_CODE,
        SourceSystem.CODEX,
        SourceSystem.MANUAL_UPLOAD,
        SourceSystem.CUSTOM_INGEST,
        SourceSystem.WIKI,
        SourceSystem.CODE_GRAPH,
    }
)


async def is_source_connected(customer_id: str, source: SourceSystem) -> bool:
    """Return True iff this (customer, source) is currently connected.

    For OAuth-backed sources, "connected" means an `integration_tokens` row
    exists with `status='active'`. For all other source types, the gate is
    a no-op (returns True) — see module docstring.

    Cheap one-row SELECT keyed on the unique index
    `(customer_id, source_system)`; safe to call on every enqueue.

    `raw_conn()` opens a fresh connection at READ COMMITTED outside any
    enclosing transaction, so the gate always sees the latest committed
    state — once disconnect_integration's tx commits, every subsequent
    enqueue sees the token row gone. The sub-millisecond window between
    gate-SELECT and INSERT is tolerable: ingestion_queue's UNIQUE
    constraint dedupes the rare straggler and the worker pipeline can
    handle a singleton orphan.
    """
    if source not in _OAUTH_SOURCES:
        return True
    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system=$2",
            customer_id,
            source.value,
        )
    return status == IntegrationStatus.ACTIVE.value

"""Bridge: enqueues synthetic CODE_GRAPH events from other source connectors.

Code-graph has no public webhook surface. Source connectors that already
verified inbound traffic (handlers/github.py today, handlers/gitlab.py
later) call into this module to enqueue work for the downstream
CodeGraphConnector. The bridge writes a structured payload to R2 and
inserts an ingestion_queue row with `source_system='code_graph'`.

Three event kinds:

    initial_backfill — one-time full clone + extraction for a newly-connected
                       repo. Triggered from the source connector's OAuth
                       install-complete hook.
    incremental      — push-driven delta extraction. Triggered from a
                       verified push webhook in the source connector. Per
                       spec critical-gap fix, file count is hard-capped per
                       event; spillover converts to a partial backfill.
    disconnect       — soft-delete code-graph data for a set of repos.
                       Triggered when a customer un-installs the GitHub App
                       or removes specific repos from the install.

source_event_id format mirrors what the downstream connector recomputes in
`parse_webhook_event` so the (customer, source, event_id) UNIQUE constraint
dedupes redeliveries on the same logical event.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import orjson

from shared.constants import (
    DEFAULT_INGESTION_PRIORITY,
    SOURCE_INGESTION_PRIORITY,
    SourceSystem,
)
from shared.db import get_pool
from shared.logging import get_logger
from shared.storage import get_store

log = get_logger(__name__)

# Bump if the payload schema changes incompatibly so the downstream connector
# can refuse old rows loudly instead of silently mis-parsing.
BRIDGE_PAYLOAD_SCHEMA_VERSION = 1

# Spec §10 critical gap #1: cap total changed-path count per incremental
# event to bound R2 payload size and downstream Phase B txn duration.
# Spillover triggers a partial backfill that walks the full tree (slower
# but correct via content_hash diff cache).
_MAX_PATHS_PER_INCREMENTAL = 500


# ---- public API -----------------------------------------------------------


async def enqueue_initial_backfill(
    customer_id: str,
    repo: str,
    head_sha: str,
    integration_token_id: int | None,
    originating_source: SourceSystem,
) -> bool:
    """Trigger a full backfill for `repo@head_sha` for `customer_id`.

    Idempotent on (customer_id, repo, head_sha): re-firing for the same SHA
    is a no-op via the ingestion_queue UNIQUE constraint.
    """
    payload = {
        "schema_version": BRIDGE_PAYLOAD_SCHEMA_VERSION,
        "kind": "initial_backfill",
        "repo": repo,
        "sha": head_sha,
        "integration_token_id": integration_token_id,
        "originating_source": originating_source.value,
        "enqueued_at": datetime.now(UTC).isoformat(),
    }
    source_event_id = f"code_graph:backfill:{repo}:{head_sha}"
    return await _put_and_enqueue(customer_id, source_event_id, payload)


async def enqueue_incremental(
    customer_id: str,
    repo: str,
    sha: str,
    files_added: list[str],
    files_modified: list[str],
    files_removed: list[str],
    integration_token_id: int | None,
    originating_source: SourceSystem,
) -> bool:
    """Forward a verified push event for `repo@sha` to the code-graph queue.

    When the changed-path count exceeds the per-event cap, drops added +
    modified (re-discovered by a partial backfill) but keeps removed (cheap,
    deterministic — they trigger soft-deletes only). The spillover backfill
    is enqueued as a separate row keyed off `<sha>:spillover` so it dedupes
    independently from the main incremental event.
    """
    capped_added, capped_modified, capped_removed, spillover = _cap_changed_paths(
        files_added, files_modified, files_removed,
    )

    payload = {
        "schema_version": BRIDGE_PAYLOAD_SCHEMA_VERSION,
        "kind": "incremental",
        "repo": repo,
        "sha": sha,
        "files_added": capped_added,
        "files_modified": capped_modified,
        "files_removed": capped_removed,
        "integration_token_id": integration_token_id,
        "originating_source": originating_source.value,
        "enqueued_at": datetime.now(UTC).isoformat(),
    }
    source_event_id = f"code_graph:incremental:{repo}:{sha}"
    enqueued = await _put_and_enqueue(customer_id, source_event_id, payload)

    if spillover:
        await enqueue_initial_backfill(
            customer_id=customer_id,
            repo=repo,
            head_sha=f"{sha}:spillover",
            integration_token_id=integration_token_id,
            originating_source=originating_source,
        )
        log.warning(
            "code_graph.bridge.spillover",
            customer=customer_id,
            repo=repo,
            sha=sha,
            total_paths=len(files_added) + len(files_modified) + len(files_removed),
        )
    return enqueued


async def enqueue_disconnect(
    customer_id: str,
    repos: list[str],
    originating_source: SourceSystem,
) -> bool:
    """Soft-delete code-graph data for the given repos.

    The downstream connector cascades: marks code.symbol Documents
    deleted, drops code_repo_state rows, closes graph_node_provenance
    for code_graph on the affected nodes (existing disconnect cleanup
    soft-deletes nodes whose last provenance source disappears).

    Each disconnect call gets a unique source_event_id (timestamp-suffixed)
    so reconnect-then-disconnect cycles don't collide.
    """
    ts = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": BRIDGE_PAYLOAD_SCHEMA_VERSION,
        "kind": "disconnect",
        "repos": repos,
        "originating_source": originating_source.value,
        "enqueued_at": ts,
    }
    repos_label = "+".join(sorted(repos))[:200]
    source_event_id = f"code_graph:disconnect:{repos_label}:{ts}"
    return await _put_and_enqueue(customer_id, source_event_id, payload)


# ---- internals ------------------------------------------------------------


def _cap_changed_paths(
    added: list[str],
    modified: list[str],
    removed: list[str],
) -> tuple[list[str], list[str], list[str], bool]:
    total = len(added) + len(modified) + len(removed)
    if total <= _MAX_PATHS_PER_INCREMENTAL:
        return added, modified, removed, False
    # Removals stay (cheap soft-deletes); added/modified get spilled to a
    # partial backfill so they don't silently disappear.
    return [], [], removed, True


async def _put_and_enqueue(
    customer_id: str,
    source_event_id: str,
    payload: dict[str, Any],
) -> bool:
    """R2 write + ingestion_queue insert. Mirrors main._enqueue's
    non-coalesced path: ON CONFLICT DO NOTHING dedupes redeliveries."""
    store = get_store()
    bucket = await store.bucket_for(customer_id)
    key = (
        f"raw/{SourceSystem.CODE_GRAPH.value}/{customer_id}/"
        f"{uuid.uuid4()}.json"
    )
    body = orjson.dumps(payload)
    await store.put(bucket, key, body)

    priority = SOURCE_INGESTION_PRIORITY.get(
        SourceSystem.CODE_GRAPH, DEFAULT_INGESTION_PRIORITY
    )
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, priority)
            VALUES ($1, $2, $3, $4, ARRAY[$4], $5)
            ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING
            RETURNING queue_id
            """,
            customer_id,
            SourceSystem.CODE_GRAPH.value,
            source_event_id,
            key,
            priority,
        )
    enqueued = row is not None
    log.info(
        "code_graph.bridge.enqueued",
        customer=customer_id,
        kind=payload.get("kind"),
        source_event_id=source_event_id,
        new=enqueued,
    )
    return enqueued


__all__ = [
    "BRIDGE_PAYLOAD_SCHEMA_VERSION",
    "enqueue_disconnect",
    "enqueue_incremental",
    "enqueue_initial_backfill",
]

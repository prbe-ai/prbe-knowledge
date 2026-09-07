"""Installation-scoped purge, fenced with the same row as projection writes.

Only connector-issued bindings confer deletion ownership. Matching a source name
or repository title cannot grant ownership of an unrelated/legacy document.
"""

from __future__ import annotations

from fastapi import HTTPException

from engine.shared.db import with_tenant
from kb.github_control import get_installation


async def _owned_doc_ids(conn, customer_id, installation_id):
    rows = await conn.fetch(
        """SELECT b.doc_id FROM github_document_bindings b
        WHERE b.customer_id=$1 AND b.installation_id=$2 AND NOT EXISTS (
          SELECT 1 FROM github_document_bindings other WHERE other.customer_id=$1
          AND other.doc_id=b.doc_id AND other.installation_id<>$2)""",
        customer_id,
        installation_id,
    )
    return [r["doc_id"] for r in rows]


async def preview(customer_id: str, installation_id: str) -> dict:
    async with with_tenant(customer_id) as conn:
        await get_installation(conn, customer_id, installation_id)
        ids = await _owned_doc_ids(conn, customer_id, installation_id)
        chunks = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE customer_id=$1 AND doc_id=ANY($2::text[])",
            customer_id,
            ids,
        )
        jobs = await conn.fetchval(
            """SELECT count(*) FROM github_backfill_jobs WHERE customer_id=$1 AND installation_id=$2
            AND state IN ('queued','running','cancel_requested')""",
            customer_id,
            installation_id,
        )
    return {
        "documents": len(ids),
        "chunks": chunks,
        "active_jobs": jobs,
        "warnings": ["Shared graph entities and unbound legacy documents are preserved."],
    }


async def purge(customer_id: str, installation_id: str, *, allow_legacy: bool = False) -> dict:
    async with with_tenant(customer_id) as conn:
        # Retain the revoked binding row: late webhook deliveries and in-flight
        # hydration cannot accidentally fall through to the legacy singleton lane.
        row = await conn.fetchrow(
            "SELECT * FROM github_installations WHERE customer_id=$1 AND installation_id=$2 FOR UPDATE",
            customer_id,
            installation_id,
        )
        if row is None:
            return {"verified": True, "documents": 0, "chunks": 0, "active_jobs": 0}
        if (
            not row["managed"]
            and not allow_legacy
            and await conn.fetchval(
                """SELECT
            EXISTS(SELECT 1 FROM ingestion_queue WHERE customer_id=$1 AND source_system='github'
                AND github_installation_id IS NULL AND status IN ('pending','processing'))
            OR EXISTS(SELECT 1 FROM backfill_state WHERE customer_id=$1 AND source_system='github'
                AND status IN ('pending','running'))""",
                customer_id,
            )
        ):
            raise HTTPException(
                409, "Existing GitHub work must finish before an installation-scoped disconnect"
            )
        await conn.execute(
            """UPDATE github_installations SET active=FALSE,sync_enabled=FALSE,managed=TRUE,
            generation=generation+1,revision=revision+1,updated_at=now() WHERE customer_id=$1 AND installation_id=$2""",
            customer_id,
            installation_id,
        )
        await conn.execute(
            """UPDATE github_backfill_jobs SET state='canceled',finished_at=now(),lease_id=NULL
            WHERE customer_id=$1 AND installation_id=$2 AND state IN ('queued','running','cancel_requested')""",
            customer_id,
            installation_id,
        )
        await conn.execute(
            """UPDATE ingestion_queue SET status='v2_canceled',github_payload=NULL,
            github_lease_id=NULL,completed_at=now() WHERE customer_id=$1 AND github_installation_id=$2
            AND status IN ('v2_pending','v2_processing')""",
            customer_id,
            installation_id,
        )
        ids = await _owned_doc_ids(conn, customer_id, installation_id)
        chunks = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE customer_id=$1 AND doc_id=ANY($2::text[])",
            customer_id,
            ids,
        )
        for table, column in (
            ("inferred_edges_queue", "anchor_doc_id"),
            ("chunks", "doc_id"),
            ("failed_chunks", "doc_id"),
            ("documents", "doc_id"),
        ):
            await conn.execute(
                f"DELETE FROM {table} WHERE customer_id=$1 AND {column}=ANY($2::text[])",
                customer_id,
                ids,
            )
        # Edges incident to document nodes disappear by FK cascade. Shared people
        # and repository entities survive; never source-wide-delete another App.
        await conn.execute(
            """DELETE FROM graph_nodes n WHERE customer_id=$1 AND canonical_id=ANY($2::text[])
            AND NOT EXISTS(SELECT 1 FROM graph_node_provenance p WHERE p.customer_id=$1
              AND p.node_id=n.node_id AND p.source_system<>'github')""",
            customer_id,
            ids,
        )
        await conn.execute(
            "DELETE FROM github_document_bindings WHERE customer_id=$1 AND installation_id=$2",
            customer_id,
            installation_id,
        )
        await conn.execute(
            """DELETE FROM integration_tokens WHERE customer_id=$1 AND source_system='github'
            AND scope=$2 AND device_id IS NULL""",
            customer_id,
            f"installation:{installation_id}",
        )
        await conn.execute(
            "DELETE FROM customer_source_mapping WHERE customer_id=$1 AND source_system='github' AND external_id=$2",
            customer_id,
            installation_id,
        )
    return {
        "verified": True,
        "documents": len(ids),
        "chunks": chunks,
        "active_jobs": 0,
        "warnings": ["Shared graph entities and unbound legacy documents were preserved."],
    }

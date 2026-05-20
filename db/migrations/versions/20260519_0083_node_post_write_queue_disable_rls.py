"""node_post_write_queue: disable RLS so the cross-tenant worker can drain it.

Migration 0082 created ``node_post_write_queue`` with ENABLE (no FORCE) RLS
and a `tenant_isolation` policy. Intent was to mirror inferred_edges_queue
post-migration 0068. But live state of inferred_edges_queue is actually
``relrowsecurity = f`` (RLS fully disabled) — 0068's intermediate "ENABLE
no-FORCE" state was further relaxed somewhere downstream so the
PostgreSQL `probe_app` role (which lacks ``BYPASSRLS``) can drain the
queue cross-tenant without setting ``app.current_customer_id`` at claim
time (the GUC can only be set per-customer, but the worker doesn't know
the customer until it has the row).

This migration drops the policy and disables RLS on
``node_post_write_queue`` to match inferred_edges_queue's actual current
state. Tenant scoping still happens elsewhere:
  * graph_writer.upsert_nodes (the only INSERT path) already runs inside
    ``with_tenant(customer_id)`` so it cannot insert a row with the wrong
    customer_id.
  * The worker reads (customer_id, node_id) from the queue row and then
    calls ``with_tenant(customer_id)`` before loading the node — so all
    DOWNSTREAM access stays RLS-enforced.

Revision ID: 0083_node_post_write_queue_disable_rls
Revises: 0082_node_post_write_pipeline
Create Date: 2026-05-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0083_node_post_write_queue_disable_rls"
down_revision: str | Sequence[str] | None = "0082_node_post_write_pipeline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS node_post_write_queue_tenant_isolation "
        "ON node_post_write_queue"
    )
    op.execute("ALTER TABLE node_post_write_queue DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE node_post_write_queue ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY node_post_write_queue_tenant_isolation
            ON node_post_write_queue
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )

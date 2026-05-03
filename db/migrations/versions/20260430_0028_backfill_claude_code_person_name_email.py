"""backfill claude_code Person nodes with name + email from neon_auth.user

Revision ID: 0028_cc_person_email_bkfl
Revises: 0027_mcp_oauth_sessions
Create Date: 2026-04-30

One-shot backfill that hydrates existing Claude Code Person graph nodes
with `name` and `email` properties pulled from neon_auth."user".

WHY
---
Lane A (prbe-backend PR #67) made the gateway inject employee_name +
employee_email into the /webhooks/claude_code body. Lane B (the handler
change shipping with this migration) writes those fields onto the Person
GraphNodeSpec.properties so name-keyed graph filters resolve via
idx_graph_nodes_lower_props_name. The handler change covers nodes
created from now on; this migration covers nodes created before Lane A
went live.

This becomes obsolete once the cross-connector alias-resolution layer
lands, at which point Person node properties will be derived from a
canonical identity store rather than per-connector enrichment. Until
then, a one-shot backfill is the cheapest unblock for retrieval.

WHAT
----
For every graph_nodes row that:
  * has label = 'Person'
  * has a graph_node_provenance row asserting source_system = 'claude_code'
  * has canonical_id matching a UUID (the claude_code employee_id format,
    which mirrors neon_auth."user".id), and
  * does not already carry both name + email,
merge name + email from neon_auth."user" into properties. Null fields in
neon_auth are skipped via jsonb_strip_nulls so we never write null
properties (the LOWER(properties->>'name') index would otherwise hold
useless empty entries).

IDEMPOTENT
----------
Running twice produces the same result. The WHERE clause excludes rows
that already have both keys, and the merge skips nulls. Rows whose
neon_auth.user has both name and email NULL are unchanged on every run.

DOWNGRADE
---------
No-op. The properties merge is non-destructive and there's no clean way
to reverse it (we'd have to know which keys came from this backfill vs
the live handler).
"""

from __future__ import annotations

from alembic import op

revision = "0028_cc_person_email_bkfl"
down_revision = "0027_mcp_oauth_sessions"
branch_labels = None
depends_on = None


# UUID regex matches the canonical_id format Claude Code uses (the verified
# employee_id from neon_auth."user".id). Keeping this guard means even if a
# future connector also stamps Person nodes with source_system='claude_code',
# we won't try to JOIN their non-UUID canonical_ids against neon_auth.user.id.
_UUID_REGEX = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def upgrade() -> None:
    # graph_nodes has FORCE ROW LEVEL SECURITY (see migration 0002), so the
    # tenant_isolation policy applies even to the table owner this migration
    # runs as. Without app.current_customer_id set, the policy reduces to
    # customer_id = '' and the UPDATE would silently touch zero rows. We
    # briefly disable FORCE for the duration of this migration's transaction
    # and restore it before commit. Alembic wraps migrations in a single
    # transaction, so any failure rolls back the NO FORCE along with the
    # UPDATE -- RLS posture is identical pre- and post-migration.
    op.execute("ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        UPDATE graph_nodes g
        SET properties = g.properties || jsonb_strip_nulls(
                jsonb_build_object('name', u.name, 'email', u.email)
            ),
            updated_at = NOW()
        FROM graph_node_provenance p, neon_auth."user" u
        WHERE p.node_id = g.node_id
          AND p.customer_id = g.customer_id
          AND p.source_system = 'claude_code'
          AND g.label = 'Person'
          AND g.canonical_id ~* '{_UUID_REGEX}'
          AND u.id::text = g.canonical_id
          AND (u.name IS NOT NULL OR u.email IS NOT NULL)
          AND NOT (g.properties ? 'name' AND g.properties ? 'email')
        """
    )
    op.execute("ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # No-op. Property merges aren't cleanly reversible: a present key may
    # have come from this backfill, the live handler, or a future write.
    pass

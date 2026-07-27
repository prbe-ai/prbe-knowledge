"""backfill AgentSession node names so existing sessions become groundable

AgentSession nodes written before this carry a name grounding cannot match:
either the session document's title (an email address dilutes pg_trgm
similarity below the 0.3 threshold, and the id is truncated to 8 characters) or
research-os's old short form. Those sessions are unreachable as graph anchors,
so a run and its transcript never become neighbours.

New ingests get the corrected name from `agent_session_display_name`, but a
COMPLETED session receives no further batches, so `normalize()` never re-runs
and the name is never rewritten. Without this backfill the fix applies only to
sessions captured after deploy.

Rebuilt in SQL rather than by re-ingesting: the inputs are already on the node
(`agent`, `session_id`) and on the session document (the author's display
name), so nothing needs re-reading from R2.

Mirrors 0040/0042 (the claude_code / codex doc-title backfills): drop FORCE RLS
for the statement because it rewrites every tenant's rows, then restore it.

Revision ID: 0098_agent_session_names
Revises: 0097_purge_runs
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0098_agent_session_names"
down_revision: str | Sequence[str] | None = "0097_purge_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY")
    # The person comes from the session document's author, resolved through the
    # Person node that the same handler writes. LEFT JOIN: identity is optional,
    # and an identity-less session still becomes id-groundable, which is the
    # half that was completely broken.
    op.execute(
        """
        WITH person AS (
            SELECT p.customer_id,
                   p.canonical_id AS employee_id,
                   NULLIF(p.properties->>'name', '') AS display_name
            FROM graph_nodes p
            WHERE p.label = 'Person'
        )
        UPDATE graph_nodes g
        SET properties = g.properties || jsonb_build_object(
                'name',
                CASE
                    WHEN d.author_id IS NOT NULL AND person.display_name IS NOT NULL
                        THEN person.display_name || ' '
                             || replace(g.properties->>'agent', '_', ' ')
                             || ' session ' || (g.properties->>'session_id')
                    ELSE replace(g.properties->>'agent', '_', ' ')
                         || ' session ' || (g.properties->>'session_id')
                END
            ),
            updated_at = NOW()
        FROM documents d
        LEFT JOIN person
               ON person.customer_id = d.customer_id
              AND person.employee_id = d.author_id
        WHERE g.label = 'AgentSession'
          AND g.properties ? 'agent'
          AND g.properties ? 'session_id'
          AND d.customer_id = g.customer_id
          AND d.source_id   = g.properties->>'session_id'
          AND d.source_system = g.properties->>'agent'
        """
    )
    # Sessions whose document is missing (purged, or never ingested because the
    # node came from the run side before research-os stopped asserting it) still
    # get the id-groundable form.
    op.execute(
        """
        UPDATE graph_nodes g
        SET properties = g.properties || jsonb_build_object(
                'name',
                replace(g.properties->>'agent', '_', ' ')
                || ' session ' || (g.properties->>'session_id')
            ),
            updated_at = NOW()
        WHERE g.label = 'AgentSession'
          AND g.properties ? 'agent'
          AND g.properties ? 'session_id'
          AND (g.properties->>'name') NOT LIKE '%' || (g.properties->>'session_id')
        """
    )
    op.execute("ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # Names are derived, not authored: the previous value carried no information
    # this cannot recompute, and restoring an unreachable name is not useful.
    pass

"""backfill codex session Document titles + identity metadata.

Revision ID: 0042_backfill_codex_doc_titles
Revises: 0041_wiki_v4_agent_loop
Create Date: 2026-05-05

WHY
---
Claude Code session identity backfill landed in migration 0040. Codex uses
the same downstream handler shape and doc_type, but historical Codex sessions
were still titled like "Codex session 82861aa0" and lacked identity metadata
because the Codex gateway path did not yet forward employee_name,
employee_email, or employee_hostname.

This migration gives old live Codex session docs the same retrieval surface
as new sessions: title + metadata on documents, and name/email/hostname on
Codex Person graph nodes.

Re-embedding of metadata chunks is handled by:
    scripts/backfill_cc_metadata_chunks.py --source codex

DOWNGRADE
---------
No-op. These merges are non-destructive and cannot be cleanly separated from
live handler writes after deployment.
"""

from __future__ import annotations

from alembic import op

revision = "0042_backfill_codex_doc_titles"
down_revision = "0041_wiki_v4_agent_loop"
branch_labels = None
depends_on = None


# UUID regex matches the canonical_id format coding-agent Person nodes use:
# the verified employee_id from neon_auth."user".id.
_UUID_REGEX = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def upgrade() -> None:
    # ----- (1) documents: title + identity metadata --------------------------
    #
    # documents has neither RLS nor FORCE RLS, so a global UPDATE under the
    # migration role reaches all tenants. The update is idempotent via
    # IS DISTINCT FROM checks against the computed title and metadata values.
    op.execute(
        """
        WITH enrichment AS (
            SELECT
                d.customer_id,
                d.doc_id,
                d.version,
                u.name  AS u_name,
                u.email AS u_email,
                (
                    SELECT NULLIF(it.device_metadata->>'hostname', '')
                    FROM integration_tokens it
                    WHERE it.customer_id = d.customer_id
                      AND it.device_id   = d.metadata->>'device_id'
                      AND NULLIF(it.device_metadata->>'hostname', '') IS NOT NULL
                    ORDER BY it.updated_at DESC NULLS LAST
                    LIMIT 1
                ) AS u_hostname
            FROM documents d
            LEFT JOIN neon_auth."user" u ON u.id::text = d.author_id
            WHERE d.source_system = 'codex'
              AND d.doc_type      = 'claude_code.session'
              AND d.valid_to IS NULL
        ),
        formatted AS (
            SELECT
                e.*,
                CONCAT(
                    CASE WHEN e.u_name IS NOT NULL THEN e.u_name || '''s ' ELSE '' END,
                    CASE WHEN e.u_email IS NOT NULL THEN '(' || e.u_email || ') ' ELSE '' END,
                    'Codex session ', LEFT(d.source_id, 8),
                    CASE WHEN e.u_hostname IS NOT NULL THEN ' (' || e.u_hostname || ')' ELSE '' END
                ) AS new_title
            FROM enrichment e
            JOIN documents d
              ON d.customer_id = e.customer_id
             AND d.doc_id      = e.doc_id
             AND d.version     = e.version
        )
        UPDATE documents d
        SET title = f.new_title,
            metadata = d.metadata || jsonb_strip_nulls(
                jsonb_build_object(
                    'employee_name',     f.u_name,
                    'employee_email',    f.u_email,
                    'employee_hostname', f.u_hostname
                )
            ),
            updated_at = NOW()
        FROM formatted f
        WHERE f.customer_id = d.customer_id
          AND f.doc_id      = d.doc_id
          AND f.version     = d.version
          AND (
              d.title IS DISTINCT FROM f.new_title
              OR d.metadata->>'employee_name'     IS DISTINCT FROM f.u_name
              OR d.metadata->>'employee_email'    IS DISTINCT FROM f.u_email
              OR d.metadata->>'employee_hostname' IS DISTINCT FROM f.u_hostname
          )
        """
    )

    # ----- (2) graph_nodes: identity onto Codex Person nodes -----------------
    #
    # graph_nodes has FORCE ROW LEVEL SECURITY, so bracket the update with
    # NO FORCE / FORCE as in migrations 0028 and 0040.
    op.execute("ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        WITH user_hostname AS (
            SELECT DISTINCT ON (it.customer_id, it.device_metadata->>'employee_id')
                it.customer_id,
                (it.device_metadata->>'employee_id')         AS employee_id,
                NULLIF(it.device_metadata->>'hostname', '')  AS hostname
            FROM integration_tokens it
            WHERE it.device_metadata ? 'hostname'
              AND it.device_metadata ? 'employee_id'
              AND NULLIF(it.device_metadata->>'hostname', '')    IS NOT NULL
              AND NULLIF(it.device_metadata->>'employee_id', '') IS NOT NULL
            ORDER BY
                it.customer_id,
                it.device_metadata->>'employee_id',
                it.updated_at DESC NULLS LAST
        ),
        enrichment AS (
            SELECT
                g.customer_id,
                g.node_id,
                u.name AS u_name,
                u.email AS u_email,
                uh.hostname AS u_hostname
            FROM graph_nodes g
            JOIN graph_node_provenance p
              ON p.node_id = g.node_id
             AND p.customer_id = g.customer_id
            LEFT JOIN neon_auth."user" u
              ON u.id::text = g.canonical_id
            LEFT JOIN user_hostname uh
              ON uh.customer_id = g.customer_id
             AND uh.employee_id = g.canonical_id
            WHERE p.source_system = 'codex'
              AND g.label = 'Person'
              AND g.canonical_id ~* '{_UUID_REGEX}'
        )
        UPDATE graph_nodes g
        SET properties = g.properties || jsonb_strip_nulls(
                jsonb_build_object(
                    'name',     e.u_name,
                    'email',    e.u_email,
                    'hostname', e.u_hostname
                )
            ),
            updated_at = NOW()
        FROM enrichment e
        WHERE e.customer_id = g.customer_id
          AND e.node_id = g.node_id
          AND (
              e.u_name IS NOT NULL
              OR e.u_email IS NOT NULL
              OR e.u_hostname IS NOT NULL
          )
          AND (
              (e.u_name IS NOT NULL AND g.properties->>'name' IS DISTINCT FROM e.u_name)
              OR (e.u_email IS NOT NULL AND g.properties->>'email' IS DISTINCT FROM e.u_email)
              OR (
                  e.u_hostname IS NOT NULL
                  AND g.properties->>'hostname' IS DISTINCT FROM e.u_hostname
              )
          )
        """
    )
    op.execute("ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # No-op. See module docstring.
    pass

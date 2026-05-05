"""backfill claude_code session Document titles + identity metadata.

Revision ID: 0040_backfill_cc_doc_titles
Revises: 0039_wiki_reclaim_and_stage
Create Date: 2026-05-04

NOTE: revision id kept under 32 chars because alembic_version.version_num
is varchar(32) by default; longer ids overflow the column and break the
deploy on first attempt (see migration 0028 for the gotcha).

WHY
---
Lane B (handler change shipping with this branch) writes identity-bearing
titles, employee_name/email/hostname into documents.metadata, and adds
hostname onto Person graph_nodes for future Claude Code session ingests.
This migration covers the docs/nodes that were created before Lane B
went live so historical sessions also surface under name+email+hostname
search.

Title format mirrors the handler's _format_session_title:
    "{name}'s ({email}) Claude Code session {short_id} ({hostname})"
with each clause omitted when its underlying field is NULL. With nothing
present it degrades to "Claude Code session {short_id}", matching the
pre-Lane-B status quo.

Re-embedding of the metadata chunks (so retrieval picks up the new
metadata text) is handled by the companion script
scripts/backfill_cc_metadata_chunks.py — atomic close-old / insert-new
per doc, no observable gap. This migration intentionally does NOT touch
the chunks table.

WHAT
----
Two UPDATEs:

(1) documents:
      For every live (valid_to IS NULL) claude_code session document,
      look up the author's name + email from neon_auth."user" via
      author_id, plus the hostname recorded for the session's device in
      integration_tokens.device_metadata. Format the title from those
      three (any may be NULL), and merge employee_name / employee_email /
      employee_hostname into the metadata jsonb (jsonb_strip_nulls so
      NULLs never land as keys).

(2) graph_nodes:
      For every Person node tagged with provenance source_system =
      'claude_code', merge a hostname property pulled from the latest
      device the user owns (MAX(updated_at) tiebreak). Skip when no
      hostname is recorded on any of their devices. Extends migration
      0028's name+email backfill.

The documents table has neither RLS nor FORCE RLS (verified against
0002_force_rls + 0036_strip_metadata_body), so the documents UPDATE
runs against the migration role without any FORCE bracket. graph_nodes
DOES have FORCE RLS (migration 0002), so the Person node UPDATE is
wrapped in NO FORCE / FORCE around the statement (same pattern as
migration 0028).

IDEMPOTENT
----------
Running twice produces the same result.
  * documents: skipped when metadata already carries 'employee_name'
    (the marker key written here). The first run sets it; subsequent
    runs filter it out via NOT (metadata ? 'employee_name').
  * graph_nodes: skipped when properties already carries 'hostname'.

DOWNGRADE
---------
No-op. The merges are non-destructive and there's no clean way to know
which keys this migration introduced vs the live handler / migration
0028 / a future write.
"""

from __future__ import annotations

from alembic import op

revision = "0040_backfill_cc_doc_titles"
down_revision = "0039_wiki_reclaim_and_stage"
branch_labels = None
depends_on = None


# ASCII-only Python comments throughout this file: ruff RUF003 fails CI on
# letter-confusable Unicode like multiplication sign / fancy quotes inside
# Python comments. SQL string literals are exempt, but keep both ASCII to
# minimise drift risk.

# UUID regex matches the canonical_id format Claude Code uses (the verified
# employee_id from neon_auth."user".id). Same guard as migration 0028.
_UUID_REGEX = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def upgrade() -> None:
    # ----- (1) documents: title + identity metadata --------------------------
    #
    # documents has neither RLS nor FORCE RLS, so a global UPDATE under
    # the migration's role hits every row (see 0036_strip_metadata_body).
    # No FORCE bracket needed here.
    #
    # The CTE pre-resolves identity per (customer_id, doc_id, version) so the
    # outer UPDATE keys cleanly on the documents per-tenant primary key.
    # author_id holds the employee UUID (matches neon_auth."user".id), and
    # device_id (from documents.metadata) joins integration_tokens to find
    # the hostname captured at device-register time.
    #
    # Title CASE WHEN clauses mirror the handler's _format_session_title
    # rules. Doubled '' inside the SQL string handles the genitive
    # apostrophe ('s ).
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
                    -- NULLIF coerces empty-string hostnames to NULL so
                    -- jsonb_strip_nulls drops them and the title's
                    -- CASE WHEN ... IS NOT NULL skips the trailing " ()".
                    -- Mirrors the Python handler's `if employee_hostname:`
                    -- truthy check.
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
            WHERE d.source_system = 'claude_code'
              AND d.doc_type      = 'claude_code.session'
              AND d.valid_to IS NULL
              AND NOT (d.metadata ? 'employee_name')
        )
        UPDATE documents d
        SET title = CONCAT(
                CASE WHEN e.u_name IS NOT NULL THEN e.u_name || '''s ' ELSE '' END,
                CASE WHEN e.u_email IS NOT NULL THEN '(' || e.u_email || ') ' ELSE '' END,
                'Claude Code session ', LEFT(d.source_id, 8),
                CASE WHEN e.u_hostname IS NOT NULL THEN ' (' || e.u_hostname || ')' ELSE '' END
            ),
            metadata = d.metadata || jsonb_strip_nulls(
                jsonb_build_object(
                    'employee_name',     e.u_name,
                    'employee_email',    e.u_email,
                    'employee_hostname', e.u_hostname
                )
            ),
            updated_at = NOW()
        FROM enrichment e
        WHERE e.customer_id = d.customer_id
          AND e.doc_id      = d.doc_id
          AND e.version     = d.version
        """
    )

    # ----- (2) graph_nodes: hostname onto Person nodes -----------------------
    #
    # graph_nodes has FORCE ROW LEVEL SECURITY (see migration 0002), so we
    # briefly disable FORCE for the duration of this migration's transaction
    # and restore it before commit. Same pattern as migration 0028.
    op.execute("ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        -- NOTE: no `it.source_system = 'claude_code'` filter here. The
        -- verify_device_token endpoint auto-reconciles claude_code rows
        -- to codex on first non-CC hit and never demotes; a user with one
        -- laptop used for both CC and Codex ends up on a single
        -- source_system=codex row, and filtering would give them no
        -- hostname on their CC Person at all. Following migration 0028's
        -- (name/email Lane B) pattern of source-agnostic lookup.
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
        )
        UPDATE graph_nodes g
        SET properties = g.properties || jsonb_build_object('hostname', uh.hostname),
            updated_at = NOW()
        FROM graph_node_provenance p, user_hostname uh
        WHERE p.node_id        = g.node_id
          AND p.customer_id    = g.customer_id
          AND p.source_system  = 'claude_code'
          AND g.label          = 'Person'
          AND g.canonical_id ~* '{_UUID_REGEX}'
          AND uh.customer_id   = g.customer_id
          AND uh.employee_id   = g.canonical_id
          AND uh.hostname IS NOT NULL
          AND NOT (g.properties ? 'hostname')
        """
    )
    op.execute("ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # No-op. See module docstring.
    pass

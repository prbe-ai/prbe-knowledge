"""wiki bootstrap: link graph, timeline, raw-data sidecar + run tracking

Revision ID: 0044_wiki_bootstrap_schema
Revises: 0043_agent_turn_traces
Create Date: 2026-05-06

Schema foundation for the wiki bootstrap pipeline (per-source crawler
agents). v4 daily-replay loop is untouched. Six discrete changes:

1. `wiki_links` — typed link graph between wiki pages. Populated by a
   pure parser at every update_page / create_page commit (no LLM call).
   GBrain v0.25 column shape: src/dst (wiki_type, slug), link_type
   (relation verb), context (~80 chars surrounding), link_source
   (markdown vs frontmatter vs manual). One UNIQUE constraint with
   NULLS NOT DISTINCT collapses duplicate edges.

2. `wiki_timeline_entries` — structured chronological audit trail for
   each wiki page. Crawlers append one row per source event they
   absorbed; dashboard renders this as a collapsible "audit" section
   below the page body. Dedup via UNIQUE on
   (customer_id, wiki_type, slug, entry_date, summary).

3. `wiki_raw_data` — sidecar for the original API response that
   produced (or contributed to) each wiki page. Lets us answer
   "why does the wiki say X?" by tracing back to the exact Slack
   thread or GitHub PR. UNIQUE on (customer_id, wiki_type, slug,
   source, source_ref) so re-bootstrap doesn't double-fill.

4. `wiki_synthesis_runs.kind` CHECK extension — adds 'bootstrap' to
   the accepted set ('onboarding','wake','scheduled') so per-crawler
   runs can record their own row alongside daily replay runs.

5. `wiki_synthesis_runs.source` — nullable column. Daily replay leaves
   it NULL; bootstrap crawlers set it to their source name ('slack',
   'github', 'linear', ...) so the dashboard can show per-source
   bootstrap progress.

6. RLS posture — all three new tables follow the wiki_synthesis_queue
   precedent (migration 0034): no RLS. Tenant scoping is application-
   enforced via explicit WHERE customer_id = $1 and the existing
   with_tenant context. Forcing RLS here would require setting
   app.current_customer_id around every cross-customer admin query,
   which the bootstrap orchestrator does not need.

Downgrade is full: drop the three tables, revert the CHECK constraint,
drop the source column. wiki_synthesis_runs rows with kind='bootstrap'
are remapped to 'onboarding' (semantically the closest v3 kind) so the
constraint passes.
"""

from __future__ import annotations

from alembic import op

revision = "0044_wiki_bootstrap_schema"
down_revision = "0043_agent_turn_traces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. wiki_links — typed link graph
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE wiki_links (
            id              BIGSERIAL PRIMARY KEY,
            customer_id     TEXT NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
            src_wiki_type   TEXT NOT NULL,
            src_slug        TEXT NOT NULL,
            dst_wiki_type   TEXT NOT NULL,
            dst_slug        TEXT NOT NULL,
            link_type       TEXT NOT NULL DEFAULT '',
            context         TEXT NOT NULL DEFAULT '',
            link_source     TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT ck_wiki_links_source
                CHECK (link_source IN ('markdown','frontmatter','manual')),
            CONSTRAINT uq_wiki_links UNIQUE NULLS NOT DISTINCT
                (customer_id, src_wiki_type, src_slug,
                 dst_wiki_type, dst_slug, link_type, link_source)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_wiki_links_from ON wiki_links (customer_id, src_wiki_type, src_slug)"
    )
    op.execute("CREATE INDEX ix_wiki_links_to ON wiki_links (customer_id, dst_wiki_type, dst_slug)")

    # ------------------------------------------------------------------
    # 2. wiki_timeline_entries — chronological audit per page
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE wiki_timeline_entries (
            id              BIGSERIAL PRIMARY KEY,
            customer_id     TEXT NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
            wiki_type       TEXT NOT NULL,
            slug            TEXT NOT NULL,
            entry_date      DATE NOT NULL,
            source          TEXT NOT NULL,
            summary         TEXT NOT NULL,
            detail          TEXT NOT NULL DEFAULT '',
            source_ref      TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_wiki_timeline_dedup UNIQUE
                (customer_id, wiki_type, slug, entry_date, summary)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_wiki_timeline_page "
        "ON wiki_timeline_entries (customer_id, wiki_type, slug, entry_date DESC)"
    )
    op.execute(
        "CREATE INDEX ix_wiki_timeline_date ON wiki_timeline_entries (customer_id, entry_date DESC)"
    )

    # ------------------------------------------------------------------
    # 3. wiki_raw_data — sidecar for original API responses
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE wiki_raw_data (
            id              BIGSERIAL PRIMARY KEY,
            customer_id     TEXT NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
            wiki_type       TEXT NOT NULL,
            slug            TEXT NOT NULL,
            source          TEXT NOT NULL,
            source_ref      TEXT NOT NULL,
            data            JSONB NOT NULL,
            fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_wiki_raw_data UNIQUE
                (customer_id, wiki_type, slug, source, source_ref)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_wiki_raw_data_page "
        "ON wiki_raw_data (customer_id, wiki_type, slug, fetched_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_wiki_raw_data_source ON wiki_raw_data (customer_id, source, source_ref)"
    )

    # ------------------------------------------------------------------
    # 4. wiki_synthesis_runs.kind — extend CHECK with 'bootstrap'
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_kind")
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs ADD CONSTRAINT ck_wsr_kind CHECK (
            kind IN ('onboarding','wake','scheduled','bootstrap')
        )
        """
    )

    # ------------------------------------------------------------------
    # 5. wiki_synthesis_runs.source — nullable per-crawler discriminator
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE wiki_synthesis_runs ADD COLUMN IF NOT EXISTS source TEXT")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wsr_kind_source "
        "ON wiki_synthesis_runs (customer_id, kind, source, started_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_wsr_kind_source")
    op.execute("ALTER TABLE wiki_synthesis_runs DROP COLUMN IF EXISTS source")
    # Remap any lingering bootstrap rows so the v3 CHECK passes.
    op.execute("UPDATE wiki_synthesis_runs SET kind = 'onboarding' WHERE kind = 'bootstrap'")
    op.execute("ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_kind")
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs ADD CONSTRAINT ck_wsr_kind CHECK (
            kind IN ('onboarding','wake','scheduled')
        )
        """
    )
    op.execute("DROP TABLE IF EXISTS wiki_raw_data")
    op.execute("DROP TABLE IF EXISTS wiki_timeline_entries")
    op.execute("DROP TABLE IF EXISTS wiki_links")

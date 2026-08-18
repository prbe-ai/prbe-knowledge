"""wiki_append_idempotency: make POST .../append safe to retry

Revision ID: 0106_wiki_append_idem
Revises: 0105_documents_parent_live_index
Create Date: 2026-08-18

WHY A RETRY IS THE NORMAL PATH, NOT AN EDGE CASE
------------------------------------------------
`POST /api/wiki/pages/{type}/{slug}/append` exists so an agent can log one
decision in one call. Agents retry on timeout, and a timeout is precisely the
case where the caller cannot know whether the write landed. Without a record of
what has already been applied, the retry appends the paragraph a second time and
the page now shows one decision as two.

That is worse than the read-modify-write it replaces. The 409 that pattern
produced was at least visible; a duplicated decision log is silently wrong, and
nobody is watching -- the whole premise of the append route is unattended
writing.

WHY ITS OWN TABLE
-----------------
The same reasoning as `wiki_page_settings` (0103), for the same reason:
`documents` is a version chain whose rows are built wholly from the incoming
payload by `Normalizer._persist`, so a column there would be a fact about one
VERSION that every other writer would have to remember to carry forward. An
idempotency record is a fact about an OPERATION, not about a revision of the
text, and it must survive the next version being written.

KEYED ON (customer_id, wiki_type, slug, idempotency_key)
--------------------------------------------------------
`wiki_type`/`slug` rather than `doc_id`, matching `wiki_page_settings`,
`wiki_links` and `wiki_timeline_entries`: that is the identity the routes and
the agent both carry, and there is no FK to `documents` available in any shape
because its PK includes `version`.

Scoping the key to the PAGE rather than to the tenant is deliberate. A key is
minted per append call by the SDK, so a collision across two different pages is
not a retry of anything -- treating it as one would silently drop a real write.

RETENTION IS OPPORTUNISTIC, NOT A JOB
-------------------------------------
The route deletes this page's own rows older than the retention window inside
the same transaction, under the page lock it already holds. Bounded work
(one page's keys), no new worker, no cron, and it cannot fall behind silently
the way an external sweeper can. `created_at` is indexed to keep that delete
an index scan rather than a scan of the tenant's keys.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# 21 chars, inside alembic_version's 32-char column.
revision = "0106_wiki_append_idem"
# The REVISION STRING of 0105, which for that file matches its filename stem.
down_revision = "0105_documents_parent_live_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wiki_append_idempotency",
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey("customers.customer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("wiki_type", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        # The version the append PRODUCED. Returned verbatim on replay, so a
        # retry gets the same answer as the call it is retrying rather than
        # whatever the page has drifted to since.
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint(
            "customer_id", "wiki_type", "slug", "idempotency_key"
        ),
    )

    # Supports the opportunistic prune described in the docstring: delete this
    # page's expired keys without scanning the tenant's.
    op.create_index(
        "wiki_append_idempotency_prune_idx",
        "wiki_append_idempotency",
        ["customer_id", "wiki_type", "slug", "created_at"],
    )

    # Same tenant guard as every other tenant table here. FORCE so the policy
    # applies to the table owner too -- the ingestion role owns these tables,
    # and without FORCE it would read across tenants.
    op.execute("ALTER TABLE wiki_append_idempotency ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_append_idempotency FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON wiki_append_idempotency
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    # Dropping the table makes every append unconditionally apply again. An
    # in-flight retry that spans the downgrade can therefore duplicate a
    # paragraph. Stated rather than guarded against: the alternative is
    # retaining idempotency records for a route that no longer reads them.
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON wiki_append_idempotency")
    op.drop_index(
        "wiki_append_idempotency_prune_idx", table_name="wiki_append_idempotency"
    )
    op.drop_table("wiki_append_idempotency")

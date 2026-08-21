"""wfmem: actually backfill the misc bucket — 0118's INSERTs were eaten by RLS

Revision ID: 0119_wfmem_misc_backfill_rls
Revises: 0118_wfmem_situation_fallback
Create Date: 2026-08-21

0118 SHIPPED, REPORTED SUCCESS, AND INSERTED NOTHING. Its `ALTER TABLE ADD
COLUMN` landed (DDL is not row-filtered) and both of its data statements were
silently filtered to zero rows. In production, alembic sat at 0118 with
`situations.classifiable` present and not one `misc` row in the database.

WHY. `situations`, `clauses` and `clause_situation_edges` all carry FORCE ROW
LEVEL SECURITY, and migrations run as `app` -- the table OWNER, but not a
superuser and without BYPASSRLS. FORCE is precisely the flag that subjects the
owner to the policy too. With no `app.current_customer_id` set, the policy's
`customer_id = current_setting('app.current_customer_id', true)` compares against
NULL, every row is invisible, and

    INSERT INTO situations (...) SELECT DISTINCT s.customer_id ... FROM situations s

reads zero rows and inserts zero rows. No error. No warning. A migration whose
whole purpose was to stop a silent data loss was itself silently lost.

WHY THE TESTS DID NOT CATCH IT. The local and CI role (`prbe`) is a SUPERUSER and
bypasses RLS entirely, so under test the SELECT saw every tenant and the backfill
worked perfectly. This module's own suite warns about exactly that trap in three
separate docstrings, and the drift guard compares SCHEMA rather than data, so it
had nothing to say either. `tests/test_workflow_memory_isolation.py` now runs this
backfill under the non-superuser `prbe_rls_test` role, which fails against 0118's
version and passes against this one.

THE RULE THIS ESTABLISHES: a data migration that touches an RLS-FORCED table must
bind the tenant GUC per tenant. There is no ambient "run as admin" here -- the
role that runs migrations cannot bypass the policy, so the migration has to
cooperate with it. `customers` is deliberately NOT row-secured, which is what
makes the tenant list readable to drive the loop.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0119_wfmem_misc_backfill_rls"
down_revision = "0118_wfmem_situation_fallback"
branch_labels = None
depends_on = None

#: Kept byte-identical to `engine.shared.wfmem.situations.FALLBACK_SITUATION`;
#: a test compares them. Duplicated rather than imported because a migration that
#: imports app code runs whatever that code says TODAY, not what it said when the
#: migration was written.
_SLUG = "misc"
_LABEL = "Anything else"
_DESCRIPTION = (
    "A rule that does not belong to any of the situations above. This is a "
    "holding bucket, not a label: nothing is ever classified INTO it, and rules "
    "land here only because no situation fit them."
)

#: EVERY tenant, unfiltered -- and the filtering happens later, per tenant, with
#: the GUC bound. The obvious query here is
#:
#:     SELECT c.customer_id FROM customers c
#:      WHERE EXISTS (SELECT 1 FROM situations s WHERE s.customer_id = c.customer_id)
#:
#: and it is WRONG IN EXACTLY THE WAY 0118 WAS WRONG: the EXISTS reads
#: `situations`, which is row-secured, so with no GUC bound it is false for every
#: tenant and the loop body never runs. The first draft of this repair shipped
#: that query and the new test under `prbe_rls_test` caught it. Any read of a
#: row-secured table before the GUC is bound is the same bug wearing a different
#: hat.
_ALL_TENANTS = sa.text("SELECT customer_id FROM customers ORDER BY 1")

#: Asked ONCE THE GUC IS BOUND, where `situations` is finally visible. A tenant
#: with zero situations has never had the capability enabled, and a lone `misc`
#: row would put them in a state no code path produces -- enabled-looking, with a
#: vocabulary that cannot classify anything -- while corrupting what
#: `enabled_tenants_missing_situations` reports.
_HAS_VOCABULARY = sa.text(
    """
    SELECT EXISTS (
        SELECT 1 FROM situations
         WHERE customer_id = :customer_id AND slug <> :slug
    )
    """
)

_INSERT_BUCKET = sa.text(
    """
    INSERT INTO situations (customer_id, slug, label, description, classifiable)
    VALUES (:customer_id, :slug, :label, :description, false)
    ON CONFLICT (customer_id, slug) DO UPDATE SET classifiable = false
    """
)

_ADOPT_ORPHANS = sa.text(
    """
    INSERT INTO clause_situation_edges (customer_id, clause_id, situation_id, classification)
    SELECT c.customer_id, c.id, s.id, '{"method": "fallback_backfill_0119"}'::jsonb
      FROM clauses c
      JOIN situations s
        ON s.customer_id = c.customer_id AND s.slug = :slug
     WHERE c.customer_id = :customer_id
       AND NOT EXISTS (
               SELECT 1 FROM clause_situation_edges e WHERE e.clause_id = c.id
           )
    ON CONFLICT (clause_id, situation_id) DO NOTHING
    """
)

#: `set_config(..., is_local => true)` rather than `SET LOCAL`, because SET does
#: not take bind parameters and a tenant id interpolated into DDL-ish SQL is the
#: kind of thing that is fine until one contains a quote.
_BIND_TENANT = sa.text("SELECT set_config('app.current_customer_id', :customer_id, true)")


def upgrade() -> None:
    conn = op.get_bind()
    # `customers` is NOT row-secured, which is the only reason a data migration
    # can discover who to bind to. Reading the tenant list from `situations` --
    # the obvious source, and what 0118 did -- returns nothing.
    tenants = [row[0] for row in conn.execute(_ALL_TENANTS)]

    for customer_id in tenants:
        conn.execute(_BIND_TENANT, {"customer_id": customer_id})
        # Only NOW is `situations` readable for this tenant.
        if not conn.execute(_HAS_VOCABULARY, {"customer_id": customer_id, "slug": _SLUG}).scalar():
            continue
        conn.execute(
            _INSERT_BUCKET,
            {
                "customer_id": customer_id,
                "slug": _SLUG,
                "label": _LABEL,
                "description": _DESCRIPTION,
            },
        )
        conn.execute(_ADOPT_ORPHANS, {"customer_id": customer_id, "slug": _SLUG})

    # Leave no tenant bound to the connection the rest of the chain will use.
    conn.execute(sa.text("SELECT set_config('app.current_customer_id', '', true)"))


def downgrade() -> None:
    conn = op.get_bind()
    tenants = [row[0] for row in conn.execute(_ALL_TENANTS)]
    for customer_id in tenants:
        conn.execute(_BIND_TENANT, {"customer_id": customer_id})
        conn.execute(
            sa.text(
                """
                DELETE FROM clause_situation_edges e
                 USING situations s
                 WHERE e.situation_id = s.id
                   AND e.customer_id = :customer_id
                   AND s.slug = :slug
                """
            ),
            {"customer_id": customer_id, "slug": _SLUG},
        )
        conn.execute(
            sa.text("DELETE FROM situations WHERE customer_id = :customer_id AND slug = :slug"),
            {"customer_id": customer_id, "slug": _SLUG},
        )
    conn.execute(sa.text("SELECT set_config('app.current_customer_id', '', true)"))

"""drop workflow memory (team rules): its five tables, their rows, its two functions

Revision ID: 0142_drop_workflow_memory
Revises: 0141_tombstone_purge_side_idx
Create Date: 2026-09-26

IRREVERSIBLE, BY OWNER DECISION (Richard, 2026-09-26: "Drop it now"). The
feature's code and endpoints went in #595 (engine) and research-os #1943, which
kept the tables and rows so that dropping them could be decided on its own. This
is that decision. Nothing is exported first: the rows are rules one tenant typed
in by hand while the feature was being built, and nothing reads them any more.

WHAT IS DELETED, counted on 2026-09-26 (read-only, as superuser, on the primary):

  * research plane, database `kb`, tenant `probe` -- the only tenant with rows:
    13 situations, 2 clauses, 2 clause_situation_edges, 2 clause_evidence,
    71 serve_ledger.
  * managed plane: 0 rows in all five tables.
  * the six `wfmem_*` capability keys in `customers.preferences` that 0115 wrote:
    on research 3 of 14 tenants carried them (`probe` with two set to true, two
    others all false), on managed 5 of 5 (all false). Their only reader,
    engine/shared/wfmem/capabilities.py, went in #595.

WHAT IS DROPPED -- everything 0114-0119 created, and nothing else:

  * tables situations, clauses, clause_situation_edges, clause_evidence,
    serve_ledger. Each takes its own indexes, constraints (the composite FKs,
    the CHECKs), RLS policies, triggers, row type and, for serve_ledger, the
    BIGSERIAL sequence with it.
  * functions wfmem_touch_updated_at() and wfmem_clear_stale_clause_embedding().
    Both are trigger functions whose only triggers sit on the dropped tables.

No CASCADE. The tables are dropped children first, so the only dependents
each DROP meets are the ones it is meant to remove. Something unexpected
depending on these tables (a view someone made by hand) fails the migration by
name instead of being dropped with it without anyone noticing. On 2026-09-26
nothing outside the five depended on them on either plane.

FORCE RLS DOES NOT APPLY HERE, which is worth saying because it bit 0107 and
0118: those were DATA statements on row-secured tables, run as `app` with no
tenant GUC bound, and matched zero rows. DDL is not row-filtered. The one data
statement, the preferences UPDATE, touches `customers`, which is deliberately
not row-secured (0119 depends on the same fact), so a single statement reaches
every tenant.

DROP NEEDS OWNERSHIP. On research the owner of all five tables and both
functions is `app` (checked 2026-09-26); on managed it is `probe`. Each plane's
migrator created them, so each owns them by construction.

LOCKS. Each DROP takes ACCESS EXCLUSIVE on its own table and, because all five
have an FK to `customers`, ACCESS EXCLUSIVE on `customers` too (removing the FK
triggers there; seen in pg_locks on pg16). That blocks every tenant lookup
until the migration commits, which is milliseconds once granted. The wait to
get it is what is bounded: a long transaction holding `customers` makes this
fail after 5 s rather than queue every tenant lookup behind it.
"""

from __future__ import annotations

from alembic import op

revision = "0142_drop_workflow_memory"
down_revision = "0141_tombstone_purge_side_idx"
branch_labels = None
depends_on = None

#: Children before parents: clause_evidence and clause_situation_edges carry
#: composite FKs into clauses and situations. serve_ledger's only FK is
#: customer_id.
_TABLES = (
    "serve_ledger",
    "clause_evidence",
    "clause_situation_edges",
    "clauses",
    "situations",
)

#: Trigger functions. Dropped after the tables, which take the triggers.
_FUNCTIONS = (
    "wfmem_touch_updated_at()",
    "wfmem_clear_stale_clause_embedding()",
)

#: The keys 0115 wrote into customers.preferences. Hardcoded, not imported:
#: a migration keeps doing what it did on the day it ran.
_CAPABILITY_KEYS = (
    "wfmem_input_declared",
    "wfmem_input_imported",
    "wfmem_input_mined",
    "wfmem_output_compiled",
    "wfmem_output_midsession",
    "wfmem_output_retrieval",
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")

    # `jsonb_typeof = 'object'` because `-` on a scalar or array raises and
    # would abort the whole migration for one tenant's malformed blob (0115's
    # docstring). `?|` keeps rows without the keys untouched.
    keys = ", ".join(f"'{key}'" for key in _CAPABILITY_KEYS)
    op.execute(
        f"""
        UPDATE customers
           SET preferences = preferences - ARRAY[{keys}]::text[]
         WHERE jsonb_typeof(preferences) = 'object'
           AND preferences ?| ARRAY[{keys}]::text[]
        """
    )

    for table in _TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
    for function in _FUNCTIONS:
        op.execute(f"DROP FUNCTION IF EXISTS {function}")


def downgrade() -> None:
    """Refuses. The rows were deleted, not archived, so nothing here can bring them back.

    Recreating the five tables empty would give the pre-0142 code nothing it
    uses: that code no longer exists (#595). The exact DDL, with its FORCE RLS
    and tenant policies, is in git -- `git show <this-revision>^:db/schema.sql`.
    """
    raise RuntimeError(
        "0142_drop_workflow_memory is not reversible. The workflow-memory rows "
        "were deleted, not archived (owner decision 2026-09-26). The tables' "
        "exact DDL, including FORCE RLS and the tenant policies, is in git: "
        "`git show <this-revision>^:db/schema.sql`. Rows can only come from a "
        "database backup."
    )

"""backfill workflow-memory capability flags to false (opt-in)

Revision ID: 0111_wfmem_capability_prefs
Revises: 0110_workflow_memory_store
Create Date: 2026-08-18

The six workflow-memory capability cells live in `customers.preferences`, the
JSONB column migration 0023 added; `engine.shared.wfmem.capabilities` is the reader and
it is fail-closed, so a missing key already resolves to `false` for every
tenant. This migration writes the keys out explicitly wherever they are absent.

Behaviour change: NONE. The reader's answer is identical before and after. What
changes is observability: the dashboard's preferences GET also defaults missing
keys to false, so today an unconfigured tenant renders "off" without the row
ever having recorded it -- and "off" that comes from a fallback is
indistinguishable from "off" that a customer chose. After this, the dashboard
shows `false` because it IS false in the database.

Only ABSENT keys are touched. A tenant who has already opted in (`true`), or
who explicitly opted out (`false`), is left exactly as they are -- the WHERE
clause filters on key absence, not on value. Same idiom as 0038, which did this
for the per-source enrichment toggles.

TWO GUARDS, not one, and the second is easy to miss. `preferences` is typed
JSONB, not "JSONB object": a scalar or array is representable and the column has
no CHECK forbidding it. `jsonb_set` on a scalar does not skip the row, it raises
`cannot set path in scalar` and ABORTS THE WHOLE MIGRATION -- so one tenant with
a junk blob blocks the backfill for every other tenant. Hence
`jsonb_typeof(...) = 'object'` on both the upgrade and the downgrade (`- key` on
a scalar fails the same way). Those rows are left alone deliberately: the reader
already resolves a non-object blob to false, and quietly rewriting somebody's
malformed preferences from inside a backfill is not this migration's business.

NO DDL. `customers.preferences` already exists (schema.sql:37, `JSONB NOT NULL
DEFAULT '{}'`), so `db/schema.sql` is unchanged by this revision.

The key list is HARDCODED here rather than imported from
`engine.shared.wfmem.capabilities`. A migration is a frozen historical record: it must
keep doing what it did on the day it ran, and importing a live constant means a
future rename silently rewrites history. The cost is two lists that can drift,
which tests/test_workflow_memory_capabilities.py compares.
"""

from __future__ import annotations

from alembic import op

revision = "0111_wfmem_capability_prefs"
down_revision = "0110_workflow_memory_store"
branch_labels = None
depends_on = None

#: The six cells: three input paths, three output surfaces. Public (no leading
#: underscore) because the drift guard in the capability tests reads it.
WFMEM_CAPABILITY_KEYS_BACKFILLED = (
    "wfmem_input_declared",
    "wfmem_input_imported",
    "wfmem_input_mined",
    "wfmem_output_compiled",
    "wfmem_output_midsession",
    "wfmem_output_retrieval",
)


def upgrade() -> None:
    for key in WFMEM_CAPABILITY_KEYS_BACKFILLED:
        op.execute(
            f"""
            UPDATE customers
               SET preferences = jsonb_set(
                       COALESCE(preferences, '{{}}'::jsonb),
                       '{{{key}}}',
                       'false'::jsonb,
                       true
                   )
             WHERE NOT (COALESCE(preferences, '{{}}'::jsonb) ? '{key}')
               AND jsonb_typeof(COALESCE(preferences, '{{}}'::jsonb)) = 'object'
            """
        )


def downgrade() -> None:
    # Best-effort undo, same shape as 0038: nothing records which keys THIS
    # migration wrote, so all six go.
    #
    # For a `false` cell that is behaviourally a no-op -- the reader's
    # missing-key default is false too, and only the explicitness the upgrade
    # added is lost. For a cell some tenant had set to `true` it is NOT a
    # no-op: dropping the key turns their capability off. That is the right
    # direction for a rollback (off is the safe state, and a downgrade past
    # 0111 means the feature is being withdrawn), but it is a real state change
    # and re-upgrading does not bring the opt-in back -- the upgrade writes
    # `false` into the now-absent key.
    for key in WFMEM_CAPABILITY_KEYS_BACKFILLED:
        op.execute(
            f"""
            UPDATE customers
               SET preferences = preferences - '{key}'
             WHERE jsonb_typeof(preferences) = 'object'
            """
        )

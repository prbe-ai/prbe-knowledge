"""drop agent_runs + agent_turn_traces pivot tables (orchestrator / apps decommission)

The ticket-enrichment / incident-investigation pivot (prbe-orchestrator, later the
apps plane) owned `agent_runs` and `agent_turn_traces`. Those services have been
decommissioned and both tables were dropped directly from prod (CASCADE, with a
pg_dump backup) during teardown.

This closes the migration-chain side of the resulting drift: migration 0043 creates
`agent_turn_traces`, and migration 0090 defensively re-creates BOTH tables
("create-if-missing"), so any environment bootstrapped by running the alembic chain
from zero would resurrect them. `db/schema.sql` never included either table (see the
0090 docstring), so the from-schema.sql bootstrap path is already clean -- this makes
the run-the-chain path match it. The KB engine never reads these tables.

Idempotent: DROP TABLE IF EXISTS, so this is a no-op on prod (already dropped) and on
schema.sql-stamped envs (never created), and a real drop only on the alembic-from-zero
path. agent_turn_traces is a child of agent_runs (ON DELETE CASCADE) with no incoming
FKs from KB-core tables; CASCADE clears their indexes + any dependents (mirrors the
proven prod teardown drop).

NOTE: revision id "0093_drop_agent_run_tables" is 26 chars (<=32) per the
alembic_version.version_num cap (feedback_alembic_version_32char_cap).
"""

from __future__ import annotations

from alembic import op

revision = "0093_drop_agent_run_tables"
down_revision = "0092_drop_incident_pivot_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Child first, then parent. CASCADE makes the order irrelevant but it is
    # clearer, and it clears the FK + indexes + RLS policies in one shot.
    op.execute("DROP TABLE IF EXISTS agent_turn_traces CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_runs CASCADE")


def downgrade() -> None:
    # Irreversible: the pivot feature and its code are gone. Restore from the
    # pg_dump backup taken before the decommission drop if ever needed.
    raise NotImplementedError(
        "drop_agent_run_tables is irreversible; restore from the pg_dump backup "
        "taken before the decommission drop."
    )

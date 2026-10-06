"""Session streams accept protocol 3 (ATIF fragments) beside protocol 2.

WHY. Clients will upload a session as ATIF fragments, one per event
(protocol 3, ~/plans/research-os-atif-upload/PLAN.md R7). `session_streams`
pins each session's protocol with `CHECK (protocol_version = 2)` (0128), so
the door could not record a protocol-3 stream. This widens the check; the
door code that writes 3 ships separately and is only reached once the
receipts response advertises protocol 3, so this migration must be live
before any advertisement.

HOW. The 0128 check is an inline column constraint, so its name is
Postgres's default. It is found by definition rather than by name, then
replaced by one named check. Widening cannot fail on existing rows (all are
2), and session_streams holds one row per session (~12k in prod), so the
validating scan is milliseconds; `lock_timeout` keeps the ACCESS EXCLUSIVE
request from queueing behind a long transaction and stalling uploads.
"""

from alembic import op

revision = "0148_session_streams_protocol_3"
down_revision = "0147_kb_match_entities"
branch_labels = None
depends_on = None

CONSTRAINT = "session_streams_protocol_version_check"

UPGRADE_SQL = f"""
SET LOCAL lock_timeout = '10s';
DO $$
DECLARE c text;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'session_streams'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%protocol_version%'
  LOOP
    EXECUTE format('ALTER TABLE session_streams DROP CONSTRAINT %I', c);
  END LOOP;
END $$;
ALTER TABLE session_streams
  ADD CONSTRAINT {CONSTRAINT} CHECK (protocol_version IN (2, 3));
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    raise RuntimeError(
        "Protocol-3 streams may exist; narrowing the check back to 2 would fail on them"
    )

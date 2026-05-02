"""Backfill source_system='codex' for mislabeled device rows; index webhook_secret.

WHAT
====
Two related changes shipped together:

1. **Backfill** — Before the pairing flow learned to carry a source claim,
   every paired device landed in `integration_tokens` with
   `source_system='claude_code'`, regardless of which CLI plugin actually
   paired. This backfill promotes a row to `'codex'` when the dominant
   `source_system` of documents emitted under that `device_id` is `'codex'`.

   Devices that haven't emitted any documents stay as-is — the auto-reconcile
   path in `verify_device_token` corrects them on first webhook hit.

2. **Index** — Add a partial index on `integration_tokens.webhook_secret`
   so `verify_device_token`'s SELECT (now source-agnostic since
   `5a99fff fix(devices): verify_device_token lookup is source-agnostic`)
   uses an index lookup rather than a seq scan on the hot path.

IDEMPOTENT
==========
- The backfill UPDATE only matches rows where `source_system = 'claude_code'`,
  and after promotion they're `'codex'`. Re-running is a no-op.
- The index uses `CREATE INDEX IF NOT EXISTS`, so re-running is a no-op.

DOWNGRADE
=========
- The backfill is one-way. Source labels are derived from emitted documents,
  which the downgrade has no way to re-derive without information already
  lost in any hypothetical demotion. If you really need to revert a specific
  row, do it manually with a targeted UPDATE keyed on `device_id`.
- The index is dropped in `downgrade()`.

Revision ID: 0031_codex_device_source_bkfl
Revises: 0030_query_traces
Create Date: 2026-05-02
"""
from alembic import op


# revision identifiers, used by Alembic. Keep IDs <= 32 chars.
revision = "0031_codex_device_source_bkfl"
down_revision = "0030_query_traces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Backfill mislabeled Codex devices.
    op.execute(
        """
        UPDATE integration_tokens t
        SET source_system = 'codex',
            updated_at = NOW()
        WHERE t.source_system = 'claude_code'
          AND t.device_id IS NOT NULL
          AND t.device_id IN (
            SELECT metadata->>'device_id'
            FROM documents
            WHERE metadata ? 'device_id'
              AND source_system IN ('claude_code', 'codex')
            GROUP BY metadata->>'device_id'
            HAVING SUM(CASE WHEN source_system = 'codex' THEN 1 ELSE 0 END)
                 > SUM(CASE WHEN source_system = 'claude_code' THEN 1 ELSE 0 END)
          );
        """
    )

    # 2. Index webhook_secret for the verify_device_token hot path.
    # CREATE INDEX (not CREATE INDEX CONCURRENTLY) so this runs inside the
    # alembic transaction. Table is small (one row per device per customer)
    # so a brief lock is acceptable.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integration_tokens_webhook_secret
        ON integration_tokens (webhook_secret)
        WHERE webhook_secret IS NOT NULL AND device_id IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_integration_tokens_webhook_secret;")
    # Backfill is one-way — see module docstring.

"""backfill per-source enrichment toggles to false (opt-in)

Revision ID: 0038_backfill_prefs_off
Revises: 0037_cleanup_cc_superseded
Create Date: 2026-05-03

The orchestrator's prefs gate (introduced ~0502) is fail-closed: a
missing per-source key reads as `false`, so an unset tenant gets no
runs. That keeps behavior correct on the orchestrator side, but the
dashboard's GET /workspace/preferences also defaults missing keys to
false — meaning the UI silently falls back to "off" without the row
ever recording it.

Backfilling the keys explicitly (only where absent) makes the state
visible end-to-end: the dashboard sees `false` because it IS false in
the DB, not because of an implicit fallback. Tenants that have already
opted in (existing `true` value, or even an explicit `false`) are
untouched — the WHERE clause filters on key absence.

Behavior change: none. The orchestrator already treats missing keys as
opt-out; this migration just makes that state observable.
"""

from __future__ import annotations

from alembic import op

revision = "0038_backfill_prefs_off"
down_revision = "0037_cleanup_cc_superseded"
branch_labels = None
depends_on = None

# Per-(agent_kind, source) toggle keys read by the orchestrator's
# `prefs.is_enrichment_enabled`. Order doesn't matter (each UPDATE is
# scoped to its own key); listed alphabetically for stable diffs.
_PER_SOURCE_KEYS = (
    "dev_enrichment_github_enabled",
    "ticket_enrichment_github_enabled",
    "ticket_enrichment_linear_enabled",
)


def upgrade() -> None:
    for key in _PER_SOURCE_KEYS:
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
            """
        )


def downgrade() -> None:
    # Best-effort undo: drop the keys we may have set. We can't tell
    # apart "we wrote this" from "the user explicitly set it to false
    # at the same time", but the orchestrator's missing-key default is
    # also false, so this no-ops behaviorally.
    for key in _PER_SOURCE_KEYS:
        op.execute(f"UPDATE customers SET preferences = preferences - '{key}'")

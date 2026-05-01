"""mcp_oauth_sessions: session as the persistent identity for MCP OAuth

Revision ID: 0027_mcp_oauth_sessions
Revises: 0026_queue_payload_keys
Create Date: 2026-04-29

Refactors MCP OAuth so the session — not any individual refresh token —
is the persistent identity. Refresh tokens become rotating tickets within
a session.

WHY
---
With strict refresh-token rotation, two parallel Claude Code sessions
sharing a macOS keychain race on every refresh: whoever rotates second
gets `invalid_grant` and forces the user through the OAuth dance again.
A 30s replay grace window covers the simultaneous-refresh case but not
the late-arriver case (one CC session was dormant for minutes-to-days
while a sibling rotated past it).

Sessions fix the late-arriver case at O(1) cost. Any RT presented can
be validated against its session's liveness instead of its own
revocation state. Bonus: revocation becomes a single UPDATE on the
session row, killing every RT in the chain at once. The dashboard
"Connected apps" list also gets sane semantics — one row per session,
not one row per current RT.

WHAT
----
- New table `mcp_oauth_sessions`. One row per (client_id, user_id,
  customer_id, scope) auth grant. Holds `last_active_at` (updated on
  every refresh) and `revoked_at` (set on logout/theft).
- New column `mcp_oauth_refresh_tokens.session_id` (FK to sessions).
- Backfill groups existing RTs into one session per
  (client_id, user_id, customer_id, scope) tuple. created_at takes the
  earliest issued_at, last_active_at takes the latest. revoked_at is
  left NULL for every backfilled session — even groups whose RTs are
  all revoked, since those users have no valid RT in memory anyway and
  can't authenticate via the new path. Explicit-logout semantics are
  preserved by the existing per-RT revoked_at on the refresh-token row.
- Adds a partial index `(session_id) WHERE revoked_at IS NULL` on
  refresh tokens to keep the "find active head" query cheap.

Code in `prbe-backend/app/services/mcp_oauth/` ships in a separate PR
AFTER this migration is verified in production. This migration alone is
backward-compatible: the `session_id` column is populated but unused by
the current code path. PR #2 then refactors `consume_refresh` to use
sessions and drops the now-redundant 7-day replay grace window.

DEPLOY ORDER
------------
1. This migration lands and runs (alembic transaction). Verify in prod
   that `mcp_oauth_sessions` row count == COUNT(DISTINCT (client_id,
   user_id, customer_id, scope)) FROM mcp_oauth_refresh_tokens before
   shipping PR #2.
2. PR #2 in prbe-backend wires the code.
"""

from __future__ import annotations

from alembic import op

revision = "0027_mcp_oauth_sessions"
down_revision = "0026_queue_payload_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Sessions table. revoked_at is NULL for alive sessions; setting it
    # via UPDATE is the new logout/theft response.
    #
    # last_active_at is updated on every successful refresh by the
    # consume_refresh code path (PR #2). It's the field the inactivity
    # threshold (30 days) is checked against.
    #
    # FK to mcp_oauth_clients with ON DELETE CASCADE: if a client
    # registration is deleted (rare; mostly admin-side), every session
    # tied to it dies too. FK to customers with the same.
    op.execute(
        """
        CREATE TABLE mcp_oauth_sessions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            client_id       TEXT NOT NULL
                            REFERENCES mcp_oauth_clients(client_id) ON DELETE CASCADE,
            user_id         TEXT NOT NULL,
            customer_id     TEXT NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
            scope           TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_active_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            revoked_at      TIMESTAMPTZ
        );

        -- The hot lookup at refresh time: find the session a presented
        -- RT belongs to (via mcp_oauth_refresh_tokens.session_id),
        -- then check revoked_at + last_active_at on this row. PK on id
        -- covers it; no extra index needed yet.

        -- Listing connections per user: dashboard hits this. Partial
        -- index keeps it cheap as the table grows. Stays in sync with
        -- the existing pattern on mcp_oauth_refresh_tokens.
        CREATE INDEX mcp_oauth_sessions_user
            ON mcp_oauth_sessions(user_id, customer_id)
            WHERE revoked_at IS NULL;
        """
    )

    # Add the FK column on refresh_tokens. Nullable initially so the
    # backfill can populate it before the NOT NULL constraint goes on.
    # ON DELETE CASCADE: if a session is deleted (rare; revocation is
    # via revoked_at, not DELETE), its RTs go with it. Sane default.
    op.execute(
        """
        ALTER TABLE mcp_oauth_refresh_tokens
        ADD COLUMN session_id UUID
        REFERENCES mcp_oauth_sessions(id) ON DELETE CASCADE;
        """
    )

    # Backfill. One session per (client_id, user_id, customer_id, scope)
    # tuple. created_at = earliest issued_at in the group;
    # last_active_at = latest issued_at in the group. The CTE writes the
    # session rows, then the UPDATE links every RT to its session by
    # matching the same tuple.
    op.execute(
        """
        WITH new_sessions AS (
            INSERT INTO mcp_oauth_sessions (
                id, client_id, user_id, customer_id, scope,
                created_at, last_active_at
            )
            SELECT
                gen_random_uuid(),
                client_id,
                user_id,
                customer_id,
                scope,
                MIN(issued_at),
                MAX(issued_at)
            FROM mcp_oauth_refresh_tokens
            GROUP BY client_id, user_id, customer_id, scope
            RETURNING id, client_id, user_id, customer_id, scope
        )
        UPDATE mcp_oauth_refresh_tokens t
        SET session_id = s.id
        FROM new_sessions s
        WHERE t.client_id = s.client_id
          AND t.user_id = s.user_id
          AND t.customer_id = s.customer_id
          AND t.scope = s.scope;
        """
    )

    # Now that every existing RT has a session_id, lock it in.
    op.execute(
        """
        ALTER TABLE mcp_oauth_refresh_tokens
        ALTER COLUMN session_id SET NOT NULL;
        """
    )

    # Index for the "find active head" query in PR #2's consume_refresh:
    #   SELECT ... FROM mcp_oauth_refresh_tokens
    #   WHERE session_id = $1 AND revoked_at IS NULL
    #   ORDER BY issued_at DESC LIMIT 1
    # Partial index keeps the bytes small (only unrevoked RTs) and
    # matches the WHERE clause exactly.
    op.execute(
        """
        CREATE INDEX mcp_oauth_refresh_tokens_session_active
            ON mcp_oauth_refresh_tokens(session_id, issued_at DESC)
            WHERE revoked_at IS NULL;
        """
    )


def downgrade() -> None:
    # Drop the index first, then the FK column, then the sessions table.
    # ON DELETE CASCADE on the FK means dropping mcp_oauth_sessions
    # would also wipe RTs — we want the column drop to be a no-op on
    # token data, so we drop the column first.
    op.execute(
        """
        DROP INDEX IF EXISTS mcp_oauth_refresh_tokens_session_active;
        ALTER TABLE mcp_oauth_refresh_tokens DROP COLUMN IF EXISTS session_id;
        DROP TABLE IF EXISTS mcp_oauth_sessions CASCADE;
        """
    )

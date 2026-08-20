"""repair the two things a stamp-head bootstrap left behind on managed

Revision ID: 0112_repair_bootstrap_drift
Revises: 0111_ingestion_stats_indexes
Create Date: 2026-08-20

Found by auditing 24h of Postgres logs on both planes. Two of the four loudest
error classes on managed-shared were the same root cause wearing two hats.

WHY A PLANE CAN BE MISSING THINGS ALEMBIC THINKS IT HAS
--------------------------------------------------------
`scripts/migrate.py` bootstraps a FRESH database by applying `db/schema.sql` and
then `alembic stamp head` -- it deliberately never replays the chain, because
migrations 0007+ duplicate state schema.sql already creates. That is correct, and
it has a consequence nobody wrote down: **anything added to schema.sql AFTER a
plane was bootstrapped is missing on that plane forever.** The stamp says the
migration ran. It never did, and it never will.

`system_settings` is the case that bit. Migration 0025 created it; commit 251bdc9
(2026-07-15) folded it into schema.sql "so a fresh DB carries it" -- but managed
had already been bootstrapped, so it got neither. Measured 2026-08-20:

    SELECT value FROM system_settings WHERE key = $1
    -> ERROR: relation "system_settings" does not exist   x4,271 in 24h

The reader (`engine/system_settings/store.py`) catches that and FAILS OPEN by
design, so the damage is quiet and specific: **the global ingestion killswitch --
the master switch for halting all plugin ingestion -- has been inoperative for
every managed customer.** Flipping it would have done nothing.

Scope of the drift was measured, not assumed. Comparing every table and index
schema.sql declares against both live planes:

    managed   35 tables declared, 1 missing (system_settings), 0 indexes missing
    research  35 tables declared, 0 missing,                   1 index missing*

    * idx_chunks_fts_content -- dropped from the DB by the BM25 cleanup but still
      declared in schema.sql. Stale FILE, not a stale database. Left alone here.

So this migration repairs exactly one table. `scripts/check_schema_drift.py`
(added alongside) makes the next instance loud instead of silent.

THE SECOND HAT: neon_auth
-------------------------
    SELECT name, email FROM neon_auth."user" WHERE id = $1
    -> ERROR: permission denied for schema neon_auth        x1,514 in 24h

`scripts/migrate.py` creates the `neon_auth` shim (schema + two tables) so
`customers.organization_id`'s FK has a target. It grants nothing. On research the
app role happens to have USAGE; on managed it does not, so the gateway's person
lookup has failed on every call.

This grant is SAFE and was verified before writing it, because "grant an app role
SELECT on a table of names and emails" deserves more than a shrug: on managed
`neon_auth."user"` is that same 4-column shim, `relpages = 0` -- literally zero
rows, no customer PII, not Neon Auth's real data.

It is also NOT SUFFICIENT, and that is the honest part: the table is empty and
all five active managed customers have `organization_id IS NULL`, so person
name/email enrichment cannot work regardless of this grant. The grant stops the
error; it does not deliver the feature. See TODOS.md.

WHY GRANTS ARE DISCOVERED, NOT HARDCODED
----------------------------------------
The app role differs by deployment: `app` on research, `probe_app` on managed,
and a self-host install may have neither. `probe` already has ALTER DEFAULT
PRIVILEGES granting new tables to probe_app/probe_admin, so the CREATE TABLE
below would be covered on managed anyway -- but defaults are per-grantor and
per-schema, and relying on one silently is how the killswitch got here. The DO
blocks below grant explicitly to whichever of the known app roles actually
exists, and no-op otherwise.
"""

from __future__ import annotations

from alembic import op

revision = "0112_repair_bootstrap_drift"
down_revision = "0111_ingestion_stats_indexes"
branch_labels = None
depends_on = None

#: Deployment-specific app roles. `app` = research plane, `probe_app` /
#: `probe_admin` = managed-shared. A role that does not exist is skipped.
_APP_ROLES = "'app', 'probe_app', 'probe_admin'"


def upgrade() -> None:
    # Copied verbatim from db/schema.sql (the `system_settings` block). Kept
    # character-identical on purpose: a hand-retyped variant is how the two
    # construction paths drift apart in the first place.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS system_settings (
            key         TEXT PRIMARY KEY,
            value       JSONB NOT NULL,
            description TEXT,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_by  TEXT
        )
        """
    )
    # The row matters as much as the table. store.py treats a missing row the
    # same as a missing table: warn, fail open. Seeding it enabled=true keeps
    # today's effective behaviour (ingestion on) while making the switch real.
    op.execute(
        """
        INSERT INTO system_settings (key, value, description)
        VALUES (
            'ingestion_killswitch',
            '{"enabled": true, "reason": null}'::jsonb,
            'Master switch for all plugin ingestion. Set value->>enabled to false to halt webhooks globally.'
        )
        ON CONFLICT (key) DO NOTHING
        """
    )
    op.execute(
        f"""
        DO $$
        DECLARE role_name text;
        BEGIN
            FOR role_name IN
                SELECT rolname FROM pg_roles WHERE rolname IN ({_APP_ROLES})
            LOOP
                EXECUTE format(
                    'GRANT SELECT, INSERT, UPDATE, DELETE ON system_settings TO %I',
                    role_name
                );
            END LOOP;
        END
        $$;
        """
    )

    # neon_auth may legitimately be absent (a self-host install that never ran
    # the shim, or CI before its fixture). Guard on the schema, not on the
    # deployment mode -- the mode is not visible from inside a migration.
    op.execute(
        f"""
        DO $$
        DECLARE role_name text;
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'neon_auth') THEN
                RETURN;
            END IF;
            FOR role_name IN
                SELECT rolname FROM pg_roles WHERE rolname IN ({_APP_ROLES})
            LOOP
                EXECUTE format('GRANT USAGE ON SCHEMA neon_auth TO %I', role_name);
                EXECUTE format(
                    'GRANT SELECT ON ALL TABLES IN SCHEMA neon_auth TO %I', role_name
                );
            END LOOP;
        END
        $$;
        """
    )


def downgrade() -> None:
    # The grants are revoked; the table is NOT dropped.
    #
    # Dropping it would re-open the exact hole this migration closes, on a plane
    # where the operator's only signal is a warning log they have already been
    # ignoring for weeks. An empty extra table costs nothing; a silently
    # inoperative killswitch cost this system its master ingestion switch.
    op.execute(
        f"""
        DO $$
        DECLARE role_name text;
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'neon_auth') THEN
                RETURN;
            END IF;
            FOR role_name IN
                SELECT rolname FROM pg_roles WHERE rolname IN ({_APP_ROLES})
            LOOP
                EXECUTE format(
                    'REVOKE SELECT ON ALL TABLES IN SCHEMA neon_auth FROM %I', role_name
                );
                EXECUTE format('REVOKE USAGE ON SCHEMA neon_auth FROM %I', role_name);
            END LOOP;
        END
        $$;
        """
    )

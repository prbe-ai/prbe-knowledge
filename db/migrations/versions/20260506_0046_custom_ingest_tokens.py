"""custom_ingest_tokens: self-serve bearer tokens for the Custom Ingest API

Revision ID: 0046_custom_ingest_tokens
Revises: 0045_wiki_field_length_caps
Create Date: 2026-05-06

Storage layer for self-serve Custom Ingest bearer tokens. A customer mints
a token from the dashboard; that token authenticates writes to the Custom
Ingest endpoint without dragging the user through full OAuth.

Three pieces:

1. `custom_ingest_tokens` table. We store sha256(token) as `token_hash`
   (UNIQUE) plus a short `token_prefix` for human-readable identification
   in the dashboard. The cleartext token is shown to the user exactly
   once at mint time and never persisted.

2. RLS enabled with WITH CHECK policy; not FORCE'd so the SECURITY
   DEFINER verifier path bypasses cleanly via owner privileges.
   Matches the `integration_tokens` convention: ENABLE + policy as
   defense-in-depth for non-owner runtime roles, while owner-role
   callers (the SECURITY DEFINER verifier) can read rows without a
   tenant GUC set.

3. `verify_and_touch_custom_ingest_token(text)` SECURITY DEFINER
   function. The verifier path can't have a tenant GUC set yet (it's
   trying to figure out which tenant the bearer belongs to), so a
   plain SELECT under RLS would return zero rows for non-owner roles.
   The SECURITY DEFINER function executes as its OWNER (the migration
   role), which by Postgres default is exempt from RLS unless FORCE
   is set -- callers can't read rows directly, only via this single
   narrow surface that takes a token_hash and returns
   (token_id, customer_id).

   It also touches `last_used_at` atomically with the lookup, but
   throttled to one update per 5 minutes to keep verification cheap on
   hot paths. The IF NOT FOUND fallback handles the throttle case --
   if the token exists but was last used <5min ago, the UPDATE skips
   it, and we still need to return its identity so the caller can
   accept the request.

Downgrade drops the function, the policy, and the table (the index
goes with the table).
"""

from __future__ import annotations

from alembic import op

revision = "0046_custom_ingest_tokens"
down_revision = "0045_wiki_field_length_caps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE custom_ingest_tokens (
            token_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            name                TEXT NOT NULL,
            token_hash          TEXT NOT NULL UNIQUE,
            token_prefix        TEXT NOT NULL,
            created_by_user_id  TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used_at        TIMESTAMPTZ,
            revoked_at          TIMESTAMPTZ
        )
        """
    )

    op.execute(
        """
        CREATE INDEX ix_custom_ingest_tokens_customer_active
            ON custom_ingest_tokens (customer_id, revoked_at)
        """
    )

    # RLS enabled with WITH CHECK policy; not FORCE'd so the SECURITY DEFINER
    # verifier path bypasses cleanly via owner privileges. (Matches the
    # integration_tokens convention.)
    op.execute("ALTER TABLE custom_ingest_tokens ENABLE ROW LEVEL SECURITY")
    # FOR ALL + WITH CHECK is required: USING-only is read-side, so a
    # tenant-context INSERT/UPDATE could otherwise write a row with some
    # other tenant's customer_id. WITH CHECK enforces the same predicate
    # on the post-image of writes.
    op.execute(
        """
        CREATE POLICY custom_ingest_tokens_tenant_isolation ON custom_ingest_tokens
            FOR ALL
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )

    # SECURITY DEFINER lookup-and-touch. Runs as the function OWNER (the
    # migration role). Because the table is ENABLE'd but not FORCE'd, the
    # owner is naturally exempt from RLS -- which is the whole point: the
    # verifier path can't know the tenant until *after* the lookup. Callers
    # can only see (token_id, customer_id) for the row matching the hash
    # they supplied; no cross-tenant rows are ever exposed.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION verify_and_touch_custom_ingest_token(p_token_hash text)
        RETURNS TABLE(token_id uuid, customer_id text)
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
        BEGIN
            RETURN QUERY
            UPDATE custom_ingest_tokens
               SET last_used_at = now()
             WHERE token_hash = p_token_hash
               AND revoked_at IS NULL
               AND (last_used_at IS NULL OR last_used_at < now() - interval '5 minutes')
             RETURNING custom_ingest_tokens.token_id, custom_ingest_tokens.customer_id;
            IF NOT FOUND THEN
                RETURN QUERY
                    SELECT t.token_id, t.customer_id
                      FROM custom_ingest_tokens t
                     WHERE t.token_hash = p_token_hash
                       AND t.revoked_at IS NULL;
            END IF;
        END $$
        """
    )

    # SECURITY DEFINER functions inherit EXECUTE to PUBLIC by default, which
    # would let any role bypass RLS via this surface. Lock that down.
    # Note: PUBLIC has been revoked; default-public-grant elsewhere in the
    # codebase implies the app role inherits via group membership. If a
    # future migration introduces an explicit app role, add GRANT EXECUTE
    # here.
    op.execute(
        "REVOKE ALL ON FUNCTION verify_and_touch_custom_ingest_token(text) FROM PUBLIC"
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS verify_and_touch_custom_ingest_token(text)")
    op.execute(
        "DROP POLICY IF EXISTS custom_ingest_tokens_tenant_isolation ON custom_ingest_tokens"
    )
    op.execute("DROP TABLE IF EXISTS custom_ingest_tokens")

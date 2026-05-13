"""Fix verify_and_touch_custom_ingest_token search_path (ag_catalog hijack).

Background — the AGE-extension search_path hijack:
  prbe-knowledge's tables actually live in the ``ag_catalog`` schema, not
  ``public``. The Apache-AGE extension installs into ``ag_catalog`` and
  prepends it to the session's ``search_path`` during ``CREATE EXTENSION
  age``, so every ``CREATE TABLE`` / ``CREATE FUNCTION`` that ran while
  the migrate role had AGE loaded landed in ``ag_catalog`` — including
  ``custom_ingest_tokens`` (migration 0046).

  Migration 0046 defined ``verify_and_touch_custom_ingest_token(text)``
  with ``LANGUAGE plpgsql SECURITY DEFINER SET search_path = public``,
  which hardcodes the function's per-call search_path to a schema that
  doesn't contain the table:

      ``UPDATE custom_ingest_tokens SET last_used_at = now() ...``
      -> resolves against ``public.custom_ingest_tokens`` (doesn't exist)
      -> raises ``relation "custom_ingest_tokens" does not exist``
      -> the data plane's verifier surfaces it as
         ``HTTP 503 "custom ingest token verifier unavailable"``,
      -> every custom-ingest POST fails on a fresh tenant.

  This was caught when we drove the first real end-to-end custom-ingest
  smoke against a managed tenant; the per-tenant ``probe`` role was
  separately fixed in the prbe-backend ``apps/data_plane`` chain
  (``0005_probe_role_search_path``), but the per-function default still
  points at the wrong schema.

Fix — ``ALTER FUNCTION ... SET search_path = ag_catalog, "$user", public``.
The SECURITY DEFINER semantics are unchanged (still runs as the function
owner, still RLS-exempt); just the lookup namespace gets the AGE-schema
prefix the rest of the codebase has.

Idempotent. The original migration 0046 is left alone (it was already
applied on every existing DB — replaying it would no-op anyway since
``CREATE OR REPLACE FUNCTION`` would re-set the wrong search_path).
``db/schema.sql`` is updated separately so fresh DBs (which apply
schema.sql then ``alembic stamp head``) end up with the right value.

Revision ID: 0066_cit_search_path_fix
Revises: 0065_usage_events_outbox
Create Date: 2026-05-13
"""
from __future__ import annotations

from alembic import op

revision = "0066_cit_search_path_fix"
down_revision = "0065_usage_events_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite tests have no SECURITY DEFINER functions; no-op.
        return
    # The function was created in 0046; alembic sees it by name + arg type.
    # ALTER FUNCTION ... SET search_path overwrites the per-function config.
    op.execute(
        'ALTER FUNCTION verify_and_touch_custom_ingest_token(text) '
        'SET search_path = ag_catalog, "$user", public'
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    # Restore the (buggy) original setting from 0046 so the down chain
    # leaves the function exactly as 0046 left it.
    op.execute(
        "ALTER FUNCTION verify_and_touch_custom_ingest_token(text) "
        "SET search_path = public"
    )

"""Smoke test for migration 0028_backfill_cc_person_name_email.

We don't run the migration end-to-end here (Alembic + neon_auth fixtures
are heavy). Instead we load the revision file by path and assert the
upgrade SQL contains every load-bearing clause: provenance JOIN,
claude_code filter, neon_auth user JOIN, jsonb merge, null-skip, UUID
guard, and the idempotency check on existing properties keys.

If any of these substrings drift (e.g. someone "simplifies" the WHERE
clause and accidentally drops the UUID guard), this test catches it
before the migration runs against prod.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "db"
    / "migrations"
    / "versions"
    / "20260430_0028_backfill_claude_code_person_name_email.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_mig_0028", str(_MIGRATION_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_file_exists() -> None:
    assert _MIGRATION_PATH.is_file(), f"migration file missing at {_MIGRATION_PATH}"


def test_revision_chain() -> None:
    m = _load_migration()
    assert m.revision == "0028_backfill_cc_person_name_email"
    assert m.down_revision == "0027_mcp_oauth_sessions"


def test_downgrade_is_noop() -> None:
    m = _load_migration()
    # Should not raise and should not require a DB connection.
    assert m.downgrade() is None


def test_upgrade_sql_has_required_clauses() -> None:
    src = _MIGRATION_PATH.read_text()

    # Provenance scoping: we MUST filter by source_system to avoid
    # touching Person nodes other connectors created.
    assert "graph_node_provenance" in src
    assert "'claude_code'" in src

    # Read-only neon_auth JOIN (we never WRITE neon_auth).
    assert 'neon_auth."user"' in src

    # Property merge: jsonb_strip_nulls so null fields don't pollute the
    # graph_nodes index. jsonb_build_object so we only emit name/email
    # keys, not the whole user row.
    assert "jsonb_strip_nulls" in src
    assert "jsonb_build_object" in src
    assert "'name'" in src
    assert "'email'" in src

    # Skip rows where both fields would be null (no-op write avoidance).
    assert "IS NOT NULL" in src

    # UUID guard on canonical_id — ensures we don't try to cast
    # non-UUID strings to neon_auth."user".id.
    assert "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" in src

    # Idempotency: don't rewrite rows that already have both keys.
    assert "NOT (g.properties ? 'name' AND g.properties ? 'email')" in src

    # Person label only.
    assert "'Person'" in src

    # RLS toggle: graph_nodes has FORCE ROW LEVEL SECURITY, so the migration
    # must temporarily disable it for the UPDATE to actually touch rows
    # (without app.current_customer_id set, the tenant_isolation policy
    # would zero-match) and restore FORCE before the txn commits.
    assert "NO FORCE ROW LEVEL SECURITY" in src
    assert src.count("FORCE ROW LEVEL SECURITY") >= 2  # NO FORCE + restore FORCE

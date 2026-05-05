"""Smoke test for migration 0039_backfill_cc_doc_titles.

Like its sibling test_backfill_cc_person_name_email_migration.py, this
test loads the migration file by path and asserts the upgrade SQL has
every load-bearing clause: documents UPDATE with title CASE WHENs and
identity-metadata merge, the integration_tokens hostname subquery,
graph_nodes Person hostname extension, FORCE-RLS bracketing on
graph_nodes, and the idempotency guards.

If any of these substrings drift (e.g. someone removes the
NOT (metadata ? 'employee_name') guard and the migration starts
double-writing on second run), this test catches it before the migration
runs against prod.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "db"
    / "migrations"
    / "versions"
    / "20260504_0039_backfill_cc_doc_titles.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_mig_0039", str(_MIGRATION_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_file_exists() -> None:
    assert _MIGRATION_PATH.is_file(), f"migration file missing at {_MIGRATION_PATH}"


def test_revision_chain() -> None:
    m = _load_migration()
    assert m.revision == "0039_backfill_cc_doc_titles"
    assert m.down_revision == "0038_backfill_prefs_off"


def test_revision_id_fits_alembic_version_column() -> None:
    """alembic_version.version_num is varchar(32) by default. Lock in a
    guard so future revisions don't repeat migration 0028's overflow."""
    m = _load_migration()
    assert len(m.revision) <= 32, (
        f"revision id {m.revision!r} is {len(m.revision)} chars; "
        "alembic_version.version_num is varchar(32)"
    )


def test_downgrade_is_noop() -> None:
    m = _load_migration()
    assert m.downgrade() is None


def test_upgrade_sql_documents_clauses() -> None:
    src = _MIGRATION_PATH.read_text()

    # Documents update scoped to live CC sessions only.
    assert "source_system = 'claude_code'" in src
    assert "doc_type      = 'claude_code.session'" in src
    assert "valid_to IS NULL" in src

    # neon_auth user JOIN on author_id.
    assert 'neon_auth."user"' in src
    assert "u.id::text = d.author_id" in src

    # Hostname subquery via integration_tokens.device_metadata.
    assert "integration_tokens" in src
    assert "device_metadata->>'hostname'" in src
    assert "d.metadata->>'device_id'" in src

    # Title CASE WHENs (genitive 's, paren wrappers).
    assert "Claude Code session" in src
    # Genitive apostrophe in SQL: doubled-single-quote inside the literal.
    assert "'''s '" in src
    # Paren wrappers for email + hostname.
    assert "'(' || e.u_email || ') '" in src
    assert "' (' || e.u_hostname || ')'" in src

    # Identity merge into metadata using jsonb_strip_nulls so NULL keys
    # don't land in JSONB.
    assert "jsonb_strip_nulls" in src
    assert "'employee_name'" in src
    assert "'employee_email'" in src
    assert "'employee_hostname'" in src

    # Idempotency guard on documents — skip rows already enriched.
    assert "NOT (d.metadata ? 'employee_name')" in src


def test_upgrade_sql_graph_nodes_clauses() -> None:
    src = _MIGRATION_PATH.read_text()

    # Person + claude_code provenance scoping.
    assert "graph_node_provenance" in src
    assert "'Person'" in src
    assert "'claude_code'" in src

    # UUID guard on canonical_id.
    assert "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" in src

    # Hostname property merge via jsonb_build_object.
    assert "jsonb_build_object('hostname'" in src

    # Idempotency guard on graph_nodes — skip nodes already carrying
    # hostname.
    assert "NOT (g.properties ? 'hostname')" in src

    # FORCE RLS bracket: graph_nodes UPDATE must be wrapped in NO FORCE /
    # FORCE so the tenant_isolation policy doesn't zero-match.
    assert "ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY" in src
    assert "ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY" in src


def test_no_unicode_confusables_in_python_comments() -> None:
    """ruff RUF003 fails CI on letter-confusable Unicode in Python
    comments. Allow Unicode inside SQL string literals (between triple
    quotes) but flag bare-comment Unicode."""
    src = _MIGRATION_PATH.read_text()
    in_triple = False
    for line in src.splitlines():
        # Toggle on/off whenever a triple-double-quote appears.
        toggles = line.count('"""')
        if toggles % 2:
            in_triple = not in_triple
        if in_triple:
            continue
        # Bare-comment lines (everything after a # outside a string).
        # Strip non-ASCII; flag if any chars get dropped.
        ascii_only = line.encode("ascii", "ignore").decode("ascii")
        # NOTE: this is a coarse guard — a string literal on a non-comment
        # line is not currently checked. Migrations should keep code
        # comments ASCII regardless.
        if "#" in line and ascii_only != line:
            raise AssertionError(
                f"non-ASCII char in code/comment line: {line!r}"
            )


def test_documents_update_skips_force_rls_bracket() -> None:
    """The documents table has neither RLS nor FORCE RLS (verified
    against migration 0036_strip_metadata_body), so the documents UPDATE
    must NOT wrap itself in NO FORCE / FORCE — only the graph_nodes
    UPDATE needs that pattern."""
    src = _MIGRATION_PATH.read_text()
    # There must be exactly two ALTER ... ROW LEVEL SECURITY statements,
    # and both must reference graph_nodes (one NO FORCE, one FORCE).
    alter_lines = [
        line for line in src.splitlines() if "ROW LEVEL SECURITY" in line
    ]
    # Two op.execute statements + one mention in the docstring/comment.
    # Filter to op.execute calls only.
    op_executes = [line for line in alter_lines if line.strip().startswith("op.execute")]
    assert len(op_executes) == 2
    for line in op_executes:
        assert "graph_nodes" in line, (
            f"FORCE RLS bracket must scope to graph_nodes only (got: {line!r})"
        )

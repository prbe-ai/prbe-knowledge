"""Smoke test for migration 0042_backfill_codex_doc_titles."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "db"
    / "migrations"
    / "versions"
    / "20260505_0042_backfill_codex_doc_titles.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_mig_0042", str(_MIGRATION_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_file_exists() -> None:
    assert _MIGRATION_PATH.is_file(), f"migration file missing at {_MIGRATION_PATH}"


def test_revision_chain() -> None:
    m = _load_migration()
    assert m.revision == "0042_backfill_codex_doc_titles"
    assert m.down_revision == "0041_wiki_v4_agent_loop"


def test_revision_id_fits_alembic_version_column() -> None:
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

    assert "source_system = 'codex'" in src
    assert "doc_type      = 'claude_code.session'" in src
    assert "valid_to IS NULL" in src

    assert 'neon_auth."user"' in src
    assert "u.id::text = d.author_id" in src

    assert "integration_tokens" in src
    assert "device_metadata->>'hostname'" in src
    assert "d.metadata->>'device_id'" in src

    assert "Codex session" in src
    assert "'''s '" in src
    assert "'(' || e.u_email || ') '" in src
    assert "' (' || e.u_hostname || ')'" in src

    assert "jsonb_strip_nulls" in src
    assert "'employee_name'" in src
    assert "'employee_email'" in src
    assert "'employee_hostname'" in src

    assert "IS DISTINCT FROM f.new_title" in src
    assert "d.metadata->>'employee_name'" in src
    assert "d.metadata->>'employee_hostname'" in src


def test_upgrade_sql_graph_nodes_clauses() -> None:
    src = _MIGRATION_PATH.read_text()

    assert "graph_node_provenance" in src
    assert "p.source_system = 'codex'" in src
    assert "'Person'" in src
    assert "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" in src

    assert "jsonb_build_object" in src
    assert "'name'" in src
    assert "'email'" in src
    assert "'hostname'" in src

    assert "g.properties->>'name' IS DISTINCT FROM e.u_name" in src
    assert "g.properties->>'email' IS DISTINCT FROM e.u_email" in src
    assert "g.properties->>'hostname' IS DISTINCT FROM e.u_hostname" in src

    assert "ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY" in src
    assert "ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY" in src


def test_no_unicode_confusables_in_python_comments() -> None:
    src = _MIGRATION_PATH.read_text()
    in_triple = False
    for line in src.splitlines():
        toggles = line.count('"""')
        if toggles % 2:
            in_triple = not in_triple
        if in_triple:
            continue
        ascii_only = line.encode("ascii", "ignore").decode("ascii")
        if "#" in line and ascii_only != line:
            raise AssertionError(
                f"non-ASCII char in code/comment line: {line!r}"
            )


def test_documents_update_skips_force_rls_bracket() -> None:
    src = _MIGRATION_PATH.read_text()
    alter_lines = [
        line for line in src.splitlines() if "ROW LEVEL SECURITY" in line
    ]
    op_executes = [line for line in alter_lines if line.strip().startswith("op.execute")]
    assert len(op_executes) == 2
    for line in op_executes:
        assert "graph_nodes" in line

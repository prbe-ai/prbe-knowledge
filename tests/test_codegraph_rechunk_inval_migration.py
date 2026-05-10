"""Smoke test for migration 0063_codegraph_rechunk_inval.

Same shape as test_backfill_cc_person_name_email_migration.py: load the
revision file by path and assert the load-bearing properties without
running the migration end-to-end.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "db"
    / "migrations"
    / "versions"
    / "20260510_0063_codegraph_rechunk_inval.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_mig_0063", str(_MIGRATION_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_file_exists() -> None:
    assert _MIGRATION_PATH.is_file(), f"migration file missing at {_MIGRATION_PATH}"


def test_revision_chain() -> None:
    m = _load_migration()
    assert m.revision == "0063_codegraph_rechunk_inval"
    assert m.down_revision == "0062_chunks_content_tsv"


def test_revision_id_fits_alembic_version_column() -> None:
    """alembic_version.version_num is varchar(32) by default."""
    m = _load_migration()
    assert len(m.revision) <= 32, (
        f"revision id {m.revision!r} is {len(m.revision)} chars; "
        "alembic_version.version_num is varchar(32)"
    )


def test_downgrade_is_noop() -> None:
    m = _load_migration()
    assert m.downgrade() is None


def _upgrade_sql(src: str) -> str:
    """Extract the executable body of upgrade() so blast-radius
    assertions check op.execute calls, not docstring prose.
    """
    lines = src.splitlines()
    in_upgrade = False
    body: list[str] = []
    for line in lines:
        if line.startswith("def upgrade("):
            in_upgrade = True
            continue
        if in_upgrade:
            if line.startswith("def ") or line.startswith("class "):
                break
            body.append(line)
    return "\n".join(body)


def test_upgrade_sql_invalidates_only_code_repo_state() -> None:
    """Lock in the blast radius. The migration MUST only touch
    code_repo_state.content_hash. Drift to graph_nodes, chunks, or
    documents would be a much bigger blast radius and almost certainly
    a mistake."""
    src = _MIGRATION_PATH.read_text()
    body = _upgrade_sql(src)

    assert "code_repo_state" in body
    assert "content_hash" in body
    assert "UPDATE code_repo_state SET content_hash = ''" in body

    # Negative assertions on the upgrade body only — docstrings legitimately
    # mention these tables to explain the design.
    assert "DELETE FROM" not in body
    assert "DROP TABLE" not in body
    assert "TRUNCATE" not in body
    assert "graph_nodes" not in body
    assert "graph_edges" not in body
    # `chunks` table: invalidation is at the FILE cache layer
    # (code_repo_state), not chunk-level. Existing chunks soft-delete
    # via SCD2 when fresh extraction emits replacements.
    assert "DELETE FROM chunks" not in body


def test_upgrade_does_not_require_rls_toggle() -> None:
    """code_repo_state has RLS ENABLED but not FORCED (migration 0049),
    unlike graph_nodes/graph_edges which need the NO FORCE / FORCE
    toggle (feedback_graph_nodes_rls_force memory). The migration body
    must not include the toggle."""
    body = _upgrade_sql(_MIGRATION_PATH.read_text())
    assert "NO FORCE ROW LEVEL SECURITY" not in body

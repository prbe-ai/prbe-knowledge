"""Smoke test for the embedding_v2 HNSW migration.

Verifies the revision id fits Alembic's version_num column, is chained to
the embedding_v2 columns migration, and uses CONCURRENTLY so production
ingest doesn't hit ACCESS EXCLUSIVE during the (long) build.

The actual index build is tested implicitly via CI: db/schema.sql is
applied verbatim to the test DB and the index appears in the chunks
schema. Live HNSW build behavior is exercised by every other test that
uses `live_db`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "versions"
    / "20260509_0061_embedding_v2_hnsw.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("hnsw_v2_mig", _MIGRATION_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_revision_chain() -> None:
    m = _load_migration()
    assert m.revision == "0061_embedding_v2_hnsw"
    assert m.down_revision == "0060_add_embedding_v2_cols"


def test_revision_id_fits_alembic_version_column() -> None:
    m = _load_migration()
    assert len(m.revision) <= 32, (
        f"revision id {m.revision!r} is {len(m.revision)} chars; "
        "alembic_version.version_num is varchar(32)"
    )


def test_uses_concurrently_and_autocommit_block() -> None:
    src = _MIGRATION_PATH.read_text()
    # CONCURRENTLY must be paired with autocommit_block; CREATE INDEX
    # CONCURRENTLY can't run inside a transaction.
    assert "CREATE INDEX CONCURRENTLY" in src
    assert "DROP INDEX CONCURRENTLY" in src
    assert "autocommit_block" in src


def test_index_targets_embedding_v2_with_halfvec_cosine() -> None:
    src = _MIGRATION_PATH.read_text()
    assert "embedding_v2 halfvec_cosine_ops" in src
    # Defaults that match the v1 index so retrieval tuning translates 1:1.
    assert "m = 16" in src
    assert "ef_construction = 64" in src


def test_schema_sql_carries_the_v2_index() -> None:
    """db/schema.sql is canonical for CI's test DB; the index must be there
    too so CI sees the same shape prod gets after the alembic upgrade."""
    schema = (
        Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    ).read_text()
    assert "idx_chunks_embedding_v2_hnsw" in schema
    assert "embedding_v2 halfvec_cosine_ops" in schema

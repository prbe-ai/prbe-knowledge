"""The DB backstop under the page cap.

Weaker than the preflight on purpose, and these tests say where the line is: it
catches a writer that never goes through `staged_graph`, and it cannot catch a
writer that misreports its own size.
"""

from __future__ import annotations

import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MIGRATION = _ROOT / "db/migrations/versions/20260813_0104_wiki_live_page_size.py"
_SCHEMA = _ROOT / "db/schema.sql"


def _migration():
    spec = importlib.util.spec_from_file_location("_m0104", _MIGRATION)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _schema_predicate() -> str:
    text = _SCHEMA.read_text()
    start = text.index("CONSTRAINT ck_wiki_live_page_size CHECK (")
    body = text[text.index("(", start) + 1 : text.index("\n    )", start)]
    return " ".join(body.split())


def test_the_migration_and_the_canonical_schema_agree() -> None:
    """schema-parity compares a fresh schema.sql against baseline + migrations,
    so a divergence fails CI -- but only after a full Postgres round trip. This
    says it in one assertion, at import speed."""
    assert " ".join(_migration().PREDICATE.split()) == _schema_predicate()


def test_historical_versions_are_exempt() -> None:
    """THE CLAUSE THAT DECIDES WHETHER THE MIGRATION APPLIES AT ALL.

    `documents` is temporal: superseding a page keeps the old row and sets
    `valid_to`. 97 historical wiki versions were over the cap when this landed
    (research_os alone reached 37,540 bytes) against 0 live ones, so without
    this clause `ADD CONSTRAINT` fails outright -- and rewriting published
    history to satisfy a new rule would corrupt the audit chain the version
    list exists to provide.

    It also protects the supersede path: setting `valid_to` is an UPDATE, and
    an UPDATE re-checks the constraint.
    """
    assert "valid_to IS NOT NULL" in _migration().PREDICATE


def test_other_connectors_are_exempt() -> None:
    """`documents` holds every source. Slack threads, GitHub PRs and transcripts
    routinely run past 8 KB and have no business being split; an unscoped
    constraint would break every ingest path in the product."""
    assert "source_system <> 'wiki'" in _migration().PREDICATE


def test_the_index_page_is_exempt() -> None:
    """Generated whole from every other page. Capping it would fail the render
    rather than produce a smaller front page."""
    assert "doc_type = 'wiki.index'" in _migration().PREDICATE


def test_the_cap_matches_the_application_constant() -> None:
    """The migration duplicates the number as a literal on purpose -- a
    migration must keep meaning what it meant the day it ran -- but they must
    not disagree the day it ships, or the preflight and the backstop refuse
    different writes."""
    from kb.synthesis.staged_graph import PAGE_CAP_BYTES

    assert f"body_size_bytes <= {PAGE_CAP_BYTES}" in _migration().PREDICATE

#!/usr/bin/env python3
"""Verify every declared index contract still holds. No database required.

Run:  python scripts/check_index_contracts.py
Exit: 0 all contracts hold, 1 one or more drifted.

Deliberately a standalone script rather than only a pytest test: this repo's CI
runs lint + typecheck and does NOT run pytest, so a test-only guard would never
execute automatically. This is wired into the lint job, which does.

It checks two halves of each contract in engine/retrieval/index_contracts.py:

  1. the index still exists in db/schema.sql with the expression the query
     depends on, and
  2. the query still contains the predicate that expression serves.

Either half drifting is the bug this exists to catch. See the module docstring
in index_contracts.py for the four production incidents that motivated it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from engine.retrieval.index_contracts import INDEX_CONTRACTS  # noqa: E402


def _normalise(text: str) -> str:
    """Reduce a SQL fragment to its identity so spelling differences are not
    read as drift.

    The same expression is written several legitimate ways across this repo:
    `db/schema.sql` hand-writes `LOWER(properties->>'name')`, while Postgres
    renders the identical index as `lower((properties ->> 'name'::text))`.
    A matcher that treats those as different reports drift that is not there,
    and a guard which cries wolf gets deleted.

    So: lowercase, drop whitespace, drop `::type` casts, drop parentheses.

    Dropping parens is the aggressive step and it is a deliberate trade. It
    could in principle equate two expressions that differ only in grouping.
    That is acceptable here because the question being asked is "does this
    index mention this expression at all", not "is this expression provably
    equivalent" -- and the alternative failure (rejecting a correct index over
    a stray bracket) is the one that gets the check switched off.
    """
    t = text.lower()
    t = re.sub(r"::[a-z0-9_ ]+", "", t)   # ::text, ::text[], ::halfvec
    t = re.sub(r"[()\s]+", "", t)
    return t


def _index_definitions(schema_sql: str) -> dict[str, str]:
    """Map index name -> its full definition text from db/schema.sql.

    Matches `CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] <name> ...`
    up to the statement terminator. Also catches definitions nested inside the
    DO blocks schema.sql uses to guard extension-dependent indexes.
    """
    out: dict[str, str] = {}
    pattern = re.compile(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?"
        r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_]+)\s+(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(schema_sql):
        out[m.group(1)] = _normalise(m.group(0))
    return out


def main() -> int:
    schema_path = REPO / "db" / "schema.sql"
    if not schema_path.exists():
        print(f"FAIL: {schema_path} not found", file=sys.stderr)
        return 1

    definitions = _index_definitions(schema_path.read_text())
    failures: list[str] = []

    for c in INDEX_CONTRACTS:
        # --- half 1: the index, with the expression, still exists -----------
        definition = definitions.get(c.index)
        if definition is None:
            failures.append(
                f"{c.index}: not found in db/schema.sql.\n"
                f"    The query in {c.source_file} depends on it.\n"
                f"    An index added by a migration but never backported to\n"
                f"    schema.sql does not exist on databases bootstrapped from\n"
                f"    schema.sql -- which is how the two data planes diverged.\n"
                f"    Why it matters: {c.why}"
            )
        elif _normalise(c.expression) not in definition:
            failures.append(
                f"{c.index}: exists, but no longer indexes "
                f"{c.expression!r}.\n"
                f"    Definition: {definition}\n"
                f"    An expression index only serves the EXACT expression it\n"
                f"    was built on, so {c.source_file} is now doing a scan.\n"
                f"    Why it matters: {c.why}"
            )

        # --- half 2: the query still uses the matching predicate ------------
        src = REPO / c.source_file
        if not src.exists():
            failures.append(f"{c.source_file}: not found (contract {c.index})")
            continue
        if _normalise(c.predicate) not in _normalise(src.read_text()):
            failures.append(
                f"{c.source_file}: predicate {c.predicate!r} is gone.\n"
                f"    It was the form that {c.index} can serve. If it was\n"
                f"    rewritten -- a coalesce() added, a sort key appended, a\n"
                f"    predicate moved across a join -- the index is now dead\n"
                f"    weight and the query is scanning.\n"
                f"    Re-measure with EXPLAIN before updating this contract.\n"
                f"    Why it matters: {c.why}"
            )

    if failures:
        print(
            f"\nINDEX CONTRACT DRIFT — {len(failures)} of "
            f"{len(INDEX_CONTRACTS)} contracts broken\n",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  ✗ {f}\n", file=sys.stderr)
        print(
            "These do not fail loudly at runtime. Postgres answers correctly,\n"
            "just by scanning, so the only symptom is latency. That is why\n"
            "this check exists.\n",
            file=sys.stderr,
        )
        return 1

    print(f"index contracts: {len(INDEX_CONTRACTS)}/{len(INDEX_CONTRACTS)} hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

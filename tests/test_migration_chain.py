"""The alembic revision chain resolves, and has exactly one head.

WHY THIS EXISTS. CI builds its database from `db/schema.sql` and then
`alembic stamp head` -- it never replays the chain (see the workflow comment,
and migration 0101's docstring on the drift that causes). So a migration whose
`down_revision` names a revision that does not exist passes every test in this
repo and fails at deploy time, in the pre-upgrade hook, before anything runs.

That is not hypothetical: 0103 shipped for review with
`down_revision = "0102_bm25_title_tokenizer"`, taken from 0102's FILENAME. Its
actual `revision` is `"0102_bm25_title_tok"`, abbreviated to fit
alembic_version's 32-char column. `alembic upgrade head` could not even build
the revision map.

These tests need no database -- they read the migration scripts off disk --
which is the point: the check has to run everywhere the suite runs, including
the environments that have no Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

_REPO_ROOT = Path(__file__).resolve().parents[1]
# alembic_version.version_num is VARCHAR(32); a longer revision string inserts
# fine locally on some backends and truncates or errors on others.
_MAX_REVISION_CHARS = 32


@pytest.fixture(scope="module")
def script() -> ScriptDirectory:
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "db" / "migrations"))
    return ScriptDirectory.from_config(config)


def test_every_down_revision_resolves(script: ScriptDirectory) -> None:
    """The failure mode: `down_revision` copied from a filename.

    `walk_revisions()` builds the whole map, so a dangling pointer raises here
    exactly as it does in `alembic upgrade head`.
    """
    revisions = list(script.walk_revisions())
    assert revisions, "no migrations found — check script_location"

    known = {rev.revision for rev in revisions}
    for rev in revisions:
        for down in rev._all_down_revisions:
            assert down in known, (
                f"{rev.revision} ({Path(rev.path).name}) declares "
                f"down_revision={down!r}, which is not any migration's "
                f"`revision`. Migration FILENAMES and revision STRINGS differ "
                f"in this repo — copy the `revision =` line, not the filename."
            )


def test_there_is_exactly_one_head(script: ScriptDirectory) -> None:
    """Two heads mean `upgrade head` is ambiguous and the deploy picks one.

    Happens when two branches each add a migration off the same parent, which
    is the normal cost of a shared main.
    """
    heads = script.get_heads()
    assert len(heads) == 1, (
        f"expected one head, found {len(heads)}: {heads}. Two migrations share "
        "a parent — one needs its `down_revision` repointed at the other."
    )


def test_revision_strings_fit_the_version_column(script: ScriptDirectory) -> None:
    for rev in script.walk_revisions():
        assert len(rev.revision) <= _MAX_REVISION_CHARS, (
            f"revision {rev.revision!r} is {len(rev.revision)} chars; "
            f"alembic_version.version_num holds {_MAX_REVISION_CHARS}. "
            "This is why 0102's revision is abbreviated relative to its file."
        )

"""Property-based round-trip test for BugClass (spec §12.1).

Hypothesis generates random valid BugClass instances; the test asserts
that model_dump(mode='json') is a fixed point for parse-then-dump.
This is the property-based equivalent of test_schema.test_round_trip_via_model_dump.

The frontmatter is stored as JSONB end-to-end (no YAML serialization in
the path), so the round-trip is JSON-only: parsed model -> dict (model_dump)
-> parsed model again (model_validate) -> dict. The two dicts must be
byte-equal.
"""
from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from services.kg.schema import (
    BugClass,
    ContextSource,
    Evidence,
    Frontmatter,
    Related,
    Signature,
)

# ---- strategies ------------------------------------------------------------

# Class-id slug: must match ^[a-z][a-z0-9-]{2,63}$ (services/kg/schema.py)
_slug = st.from_regex(r"^[a-z][a-z0-9-]{2,30}$", fullmatch=True)

# Short text avoiding control characters that JSON / asyncpg are picky about.
_short_text = st.text(
    min_size=1,
    max_size=120,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="\x00",
    ),
)

# ContextSource.tool is capped at 64 chars by the schema (services/kg/schema.py).
# Use a tighter strategy so Hypothesis doesn't generate values that fail
# Pydantic validation before the round-trip ever runs.
_tool_text = st.text(
    min_size=1,
    max_size=64,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="\x00",
    ),
)

# Embedding seed must be at least 3 chars after strip (Signature validator).
_seed = st.text(
    min_size=3,
    max_size=120,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="\x00",
    ),
).filter(lambda s: len(s.strip()) >= 3)

# must_match must be at least one non-blank entry after strip.
_must_match_entry = _short_text.filter(lambda s: s.strip())
_must_match_list = st.lists(_must_match_entry, min_size=1, max_size=4)

_signature = st.builds(
    Signature,
    must_match=_must_match_list,
    embedding_seed=_seed,
)

_related = st.builds(
    Related,
    analogous_to=st.lists(_slug, max_size=4, unique=True),
    overlaps_with=st.lists(_slug, max_size=4, unique=True),
    often_confused_with=st.lists(_slug, max_size=4, unique=True),
    regressed_by=st.lists(_slug, max_size=4, unique=True),
)

# ContextSource.params is an arbitrary dict (Pydantic field type is dict).
# Keep keys/values simple so JSON is well-defined.
_param_value = st.one_of(
    st.text(
        min_size=0,
        max_size=40,
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters="\x00",
        ),
    ),
    st.integers(min_value=-10_000, max_value=10_000),
    st.booleans(),
    st.none(),
)
_params = st.dictionaries(_short_text, _param_value, max_size=4)

_context_source = st.builds(
    ContextSource,
    priority=st.sampled_from([1, 2, 3]),
    name=_short_text,
    tool=_tool_text,
    params=_params,
)

# Datetimes that JSON can serialize; aware UTC only (avoid local-tz weirdness).
_dt = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(UTC),
)

_evidence = st.builds(
    Evidence,
    match_count=st.integers(min_value=0, max_value=10_000),
    last_updated=st.one_of(st.none(), _dt),
    recent_refinements=st.lists(_short_text, max_size=4),
)

_frontmatter = st.builds(
    Frontmatter,
    id=_slug,
    type=st.just("bug-class"),
    description=_short_text,
    signature=_signature,
    related=_related,
    context_sources=st.lists(_context_source, max_size=8),
    evidence=_evidence,
)

_bug_class = st.builds(
    BugClass,
    frontmatter=_frontmatter,
    body=st.text(
        min_size=0,
        max_size=2000,
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters="\x00",
        ),
    ),
)


# ---- the property ---------------------------------------------------------


@given(_bug_class)
@settings(max_examples=200, deadline=None)
def test_bug_class_dump_is_fixed_point(cls: BugClass) -> None:
    """parse(serialize(parse(serialize(x)))) == parse(serialize(x))."""
    raw = cls.model_dump(mode="json")
    parsed = BugClass.model_validate(raw)
    raw2 = parsed.model_dump(mode="json")
    assert raw == raw2

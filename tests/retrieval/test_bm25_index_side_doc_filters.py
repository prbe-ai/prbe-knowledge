"""`index_side_doc_filters`: source and source-key scope inside the BM25 index.

THE REGRESSION THIS EXISTS FOR
------------------------------
`sources` and `source_keys` land on the documents join AFTER the Tantivy pool's
LIMIT. On prod `probe` custom_ingest is 5.3% of live chunks, so a
`sources=["custom_ingest"]` query fills even the 4x scoped pool with transcript
chunks, and the join then throws every one of them away: the search answers
nothing for experiments, files or notes. The flag restates the scope as an
anchored chunk_id regex inside the boolean, so TopK only ranks rows in scope.

`match('doc_id', 'custom_ingest')` is NOT the fix, and the first test here pins
why: the index's default tokenizer does not split `custom_ingest:probe:...` at
':' or '_', so that query matches nothing at all.

The fixture is built to be the worst case: one custom_ingest document whose
chunk mentions the term once in a long body, against enough short, term-dense
transcript chunks to fill the scoped pool (top_k=2 -> 2 * 10 * 4 = 80 rows) on
their own. Without the flag the custom document cannot reach the join.

Seeding follows the shape production writes: doc_id from the real
constructors, chunk_id `{doc_id}:{suffix}` as the normalizer builds it. The
pre-filter's exactness rests on exactly those two facts. One custom document id
carries a newline, which a `.*` regex tail silently drops.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

import engine.shared.db as db_module
from engine.retrieval.retrievers import bm25
from engine.retrieval.retrievers.bm25 import bm25_search
from engine.shared.custom_ingest import custom_ingest_doc_id
from engine.shared.models import TemporalMode, TemporalSpec
from engine.shared.partitions import CHUNKS_PARENT, ensure_tenant_partition, is_partitioned

#: Hyphenated on purpose: the tenant id is tokenized on hyphens by the index,
#: and it is interpolated into every source-key prefix.
TENANT = "doc-scope-tenant"
TERM = "zephyrine"
WORKSPACE_KEY = "workspace:1d155c9c-4f05-4707-98a7-f69763c171e0"
#: Shares a prefix with "experiments" up to the separator, so a prefix that
#: forgot the trailing ':' would leak it.
LOOKALIKE_KEY = "experiments-archive"
#: The longest key the ingest charset allows (128 characters).
LONG_KEY = ("long-key-" + "0123456789abcdef" * 8)[:128]
TOP_K = 2
#: top_k of the source-key tests: the largest result the expected sets need.
KEYED_TOP_K = 3
#: Enough transcript chunks to fill the largest pool any test here can build
#: WITHOUT its index leg (a scoped pool, the 4x factor included), plus a
#: margin. Derived, not typed: if a pool knob grows, a fixed count would let
#: the transcripts stop filling the pool, and a test whose leg had been
#: removed would pass on the SQL post-filter alone.
TRANSCRIPT_CHUNKS = (
    max(TOP_K, KEYED_TOP_K) * bm25._BM25_POOL_MULTIPLIER * bm25._BM25_SCOPED_POOL_FACTOR + 20
)
APP_ROLE = "bm25_doc_scope_app"

EXPERIMENT_DOC = custom_ingest_doc_id(TENANT, "experiments", "page:738f0c2e")
WORKSPACE_DOC = custom_ingest_doc_id(TENANT, WORKSPACE_KEY, "file:notes.md")
LOOKALIKE_DOC = custom_ingest_doc_id(TENANT, LOOKALIKE_KEY, "page:1")
LONG_DOC = custom_ingest_doc_id(TENANT, LONG_KEY, "page:1")
#: A caller's document id is unrestricted and may hold a newline; the SQL
#: filter keeps such a row, so the index leg must too.
NEWLINE_DOC = custom_ingest_doc_id(TENANT, "experiments", "page:line one\nline two")
CUSTOM_DOCS = {EXPERIMENT_DOC, WORKSPACE_DOC, LOOKALIKE_DOC, LONG_DOC, NEWLINE_DOC}
_FILLER = " ".join(f"filler{i}" for i in range(80))


async def _doc(
    conn, doc_id: str, source: str, source_key: str | None, tenant: str = TENANT
) -> None:
    await conn.execute(
        """
        INSERT INTO documents (customer_id, doc_id, version, source_system,
                               source_id, source_url, doc_type, content_hash,
                               created_at, updated_at, valid_from, acl, metadata,
                               title, body_preview)
        VALUES ($1, $2, 1, $3, $2, 'https://x', 'custom.note', 'dh', NOW(), NOW(),
                NOW(), '{}'::jsonb,
                CASE WHEN $4::text IS NULL THEN '{}'::jsonb
                     ELSE jsonb_build_object('source_key', $4::text) END,
                'Run notes', 'p')
        """,
        tenant,
        doc_id,
        source,
        source_key,
    )


async def _chunk(
    conn, doc_id: str, n: int, content: str, tenant: str = TENANT, kind: str = "content"
) -> None:
    await conn.execute(
        """
        INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index, content,
                            content_hash, token_count, chunker_version,
                            first_seen_version, last_seen_version, kind, visibility)
        VALUES ($1, $2, $3, $4, $5, $6, 3, 'v1', 1, 2147483647, $7, 'approved')
        """,
        f"{doc_id}:c_{n:016x}",
        doc_id,
        tenant,
        n,
        content,
        f"h{n}",
        kind,
    )


@pytest_asyncio.fixture
async def seeded(pg_search_db):
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $1, $1) ON CONFLICT DO NOTHING",
            TENANT,
        )
        await ensure_tenant_partition(conn, TENANT)
        for doc_id, key in (
            (EXPERIMENT_DOC, "experiments"),
            (WORKSPACE_DOC, WORKSPACE_KEY),
            (LOOKALIKE_DOC, LOOKALIKE_KEY),
            (LONG_DOC, LONG_KEY),
            (NEWLINE_DOC, "experiments"),
        ):
            await _doc(conn, doc_id, "custom_ingest", key)
            # One mention in a long body: every transcript chunk outranks it.
            await _chunk(conn, doc_id, 0, f"{TERM} {_FILLER}")
        sessions = [f"claude_code:{TENANT}:3c325e11-2008-46a9-83f7-{i:012x}" for i in range(4)]
        for doc_id in sessions:
            await _doc(conn, doc_id, "claude_code", None)
        for n in range(TRANSCRIPT_CHUNKS):
            await _chunk(conn, sessions[n % len(sessions)], n, f"{TERM} {TERM} {TERM} t{n}")
        await _ensure_app_role(conn)
    yield


async def _ensure_app_role(conn) -> None:
    await conn.execute(
        f"""
        DO $$ BEGIN
            CREATE ROLE {APP_ROLE} NOSUPERUSER NOBYPASSRLS;
        EXCEPTION WHEN duplicate_object THEN NULL; END $$
        """
    )
    await conn.execute(f"GRANT USAGE ON SCHEMA public, paradedb TO {APP_ROLE}")
    await conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")


@pytest.fixture
def as_app_role(monkeypatch):
    """Run `bm25_search` as a role FORCE RLS binds, as production does.

    The suite connects as a superuser, which bypasses RLS: without this the
    test would prove the shape under a policy that never applied. Same
    `SET LOCAL ROLE` inside `with_tenant` as test_multitenant_isolation.
    """
    real = db_module.with_tenant

    @asynccontextmanager
    async def with_tenant_as_app(customer_id: str):  # type: ignore[no-untyped-def]
        async with real(customer_id) as conn:
            await conn.execute(f"SET LOCAL ROLE {APP_ROLE}")
            yield conn

    monkeypatch.setattr(bm25, "with_tenant", with_tenant_as_app)


async def _search(**kwargs: Any):
    return await bm25_search(TENANT, TERM, top_k=TOP_K, **kwargs)


_INDEX_COUNT = (
    "SELECT count(*) FROM chunks c "
    "WHERE c.customer_id = current_setting('app.current_customer_id', true) "
    "AND c.chunk_id @@@ {q}"
)


async def _index_count(query: str, arg: str) -> int:
    async with db_module.with_tenant(TENANT) as conn:
        return await conn.fetchval(_INDEX_COUNT.format(q=query), arg)


async def test_a_token_match_on_doc_id_finds_nothing(seeded) -> None:
    """Why the pre-filter is a chunk_id prefix and not a doc_id token match.

    `paradedb.match('doc_id', 'custom_ingest')` reads like the obvious clause
    and returns ZERO rows: the default tokenizer (UAX#29 words) splits this
    tenant's doc_ids at the hyphen and at a ':' before a digit, never at '_'
    or at a ':' between letters, so they index as `custom_ingest:doc`,
    `scope`, `tenant:experiments:page`, ... and the token `custom_ingest`
    never exists. The chunk_id regex, on the same index, finds every custom
    row.
    """
    by_token = await _index_count(
        "paradedb.match('doc_id', $1, conjunction_mode => true)", "custom_ingest"
    )
    by_prefix = await _index_count(
        "paradedb.regex('chunk_id', $1)",
        bm25._chunk_id_prefix_regexes([bm25._source_chunk_prefix("custom_ingest")])[0],
    )
    assert by_token == 0
    assert by_prefix == len(CUSTOM_DOCS)


async def test_the_key_leg_alone_is_exact(seeded) -> None:
    """The index leg by itself, without the SQL filter behind it.

    Search results cannot show a loose pre-filter -- the SQL source_key
    predicate removes whatever it lets through -- so this counts what the leg
    matches in the index. A key with ':' still matches through its `%3A`
    encoding, `experiments` does not reach `experiments-archive`, and the
    document id with a newline in it is still found (`(?s:.)*`, not `.*`).
    """
    for key, expected in (
        ("experiments", 2),
        (WORKSPACE_KEY, 1),
        (LOOKALIKE_KEY, 1),
        (LONG_KEY, 1),
    ):
        (regex,) = bm25._chunk_id_prefix_regexes([bm25._source_key_chunk_prefix(TENANT, key)])
        assert await _index_count("paradedb.regex('chunk_id', $1)", regex) == expected, key


@pytest.mark.parametrize("override", [None, CHUNKS_PARENT], ids=["partition", "parent"])
async def test_the_flag_finds_what_the_post_filter_loses(seeded, as_app_role, override) -> None:
    """THE regression, on both scan targets, as the app role under FORCE RLS.

    Flag off: the 80-row scoped pool is all transcript, so the join keeps
    nothing. Flag on: the pool holds only custom_ingest rows. Both paths are
    run because pg_search has rejected a pool shape on the partitioned parent
    while accepting it on the partition (`Unsupported query shape`); the
    parent is what the guardian canary forces and what a tenant still in
    DEFAULT uses.
    """
    scope = {"sources": ["custom_ingest"], "_scan_target_override": override}
    assert await _search(**scope) == [], (
        "precondition: without the flag the transcript chunks fill the pool -- if "
        "this fails the fixture no longer exercises the regression"
    )
    hits = await _search(index_side_doc_filters=True, **scope)
    assert hits, "the index-side scope found nothing"
    assert {h.doc_id for h in hits} <= CUSTOM_DOCS
    assert all(h.source_system == "custom_ingest" for h in hits)
    assert len(hits) == TOP_K


async def test_the_flag_does_not_change_scores(seeded) -> None:
    """`const_score(0.0, ...)`: a bare regex leg adds 1.0 to every hit. The
    pre-filter decides which rows compete, never how they score."""
    unscoped = {h.chunk_id: h.score for h in await bm25_search(TENANT, TERM, top_k=200)}
    scoped = await bm25_search(
        TENANT, TERM, top_k=200, sources=["custom_ingest"], index_side_doc_filters=True
    )
    assert scoped
    for h in scoped:
        assert h.score == pytest.approx(unscoped[h.chunk_id])


@pytest.mark.parametrize("override", [None, CHUNKS_PARENT], ids=["partition", "parent"])
async def test_source_keys_keep_only_the_keyed_docs(seeded, as_app_role, override) -> None:
    """The key prefix is `custom_ingest:{tenant}:{encoded key}:`, so a key with
    ':' (encoded `%3A`) still matches and a key sharing a prefix up to the
    separator does not.

    The pool is KEYED_TOP_K x 10 rows (x 4 if the key scope fell back to the
    join), and TRANSCRIPT_CHUNKS more than fill either with chunks that all
    outrank every custom chunk: with the key leg gone the pool is all
    transcript and the SQL filter leaves nothing, so this fails rather than
    passing on the post-filter alone.
    """
    for keys, expected in (
        (["experiments"], {EXPERIMENT_DOC, NEWLINE_DOC}),
        ([WORKSPACE_KEY], {WORKSPACE_DOC}),
        (["experiments", WORKSPACE_KEY], {EXPERIMENT_DOC, NEWLINE_DOC, WORKSPACE_DOC}),
    ):
        hits = await bm25_search(
            TENANT, TERM, top_k=KEYED_TOP_K, source_keys=keys, index_side_doc_filters=True,
            _scan_target_override=override,
        )
        assert {h.doc_id for h in hits} == expected, keys
        # The same scope, also narrowed by source: both legs ride the index.
        both = await bm25_search(
            TENANT, TERM, top_k=KEYED_TOP_K, sources=["custom_ingest"], source_keys=keys,
            index_side_doc_filters=True, _scan_target_override=override,
        )
        assert {h.doc_id for h in both} == expected, keys


#: research-os sends up to 50 keys (MAX_REQUEST_SOURCE_KEYS): its fixed corpora
#: plus one `workspace:<uuid>` per workspace. One regex over 31 of them
#: already exceeds tantivy-fst's 1,000-state limit; 8 maximum-length keys do.
FIXED_KEYS = ["artifacts", "experiments", "team_notes", f"shared:{TENANT}", "notes"]


def _research_os_keys() -> list[str]:
    """5 fixed keys + 45 workspaces, one of them seeded."""
    workspaces = [
        f"workspace:{uuid.UUID(hashlib.md5(f'w{i}'.encode()).hexdigest())}" for i in range(44)
    ]
    return [*FIXED_KEYS, WORKSPACE_KEY, *workspaces]


def _max_length_keys() -> list[str]:
    """50 distinct 128-character keys, one of them seeded."""
    return [LONG_KEY] + [hashlib.sha256(f"k{i}".encode()).hexdigest() * 2 for i in range(49)]


@pytest.mark.parametrize("override", [None, CHUNKS_PARENT], ids=["partition", "parent"])
@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (_research_os_keys(), {EXPERIMENT_DOC, NEWLINE_DOC, WORKSPACE_DOC}),
        (_max_length_keys(), {LONG_DOC}),
    ],
    ids=["5+45-workspaces", "50x128-chars"],
)
async def test_fifty_keys_stay_under_the_regex_state_limit(
    seeded, as_app_role, override, keys, expected
) -> None:
    """One regex over these keys is REFUSED by pg_search ("could not build
    regex" -- tantivy-fst's 1,000-state limit), which would cost the whole
    BM25 channel. The leg splits them by `_REGEX_PREFIX_BUDGET_BYTES`; this
    proves the unsplit form fails here and the split one answers exactly."""
    assert len(keys) == 50
    prefixes = [bm25._source_key_chunk_prefix(TENANT, key) for key in keys]
    unsplit = "(" + "|".join(bm25._regex_literal(p) for p in prefixes) + ")(?s:.)*"
    with pytest.raises(asyncpg.PostgresError, match="regex"):
        await _index_count("paradedb.regex('chunk_id', $1)", unsplit)
    assert len(bm25._chunk_id_prefix_regexes(prefixes)) > 1
    hits = await bm25_search(
        TENANT, TERM, top_k=KEYED_TOP_K, source_keys=keys, index_side_doc_filters=True,
        _scan_target_override=override,
    )
    assert {h.doc_id for h in hits} == expected


async def test_keyless_source_keys_stay_on_the_join(seeded) -> None:
    """`source_keys_include_keyless` admits docs with NO key, which carry no
    key prefix, so the key leg must not ride the index: the transcripts are
    keyless and must still come back."""
    hits = await bm25_search(
        TENANT, TERM, top_k=10, source_keys=["experiments"],
        source_keys_include_keyless=True, index_side_doc_filters=True,
    )
    assert any(h.source_system == "claude_code" for h in hits)


# ---------------------------------------------------------------------------
# max_chunks_per_doc: one long document must not be the whole answer.
# ---------------------------------------------------------------------------

CAP_TENANT = "doc-cap-tenant"
CAP_TERM = "quillwort"
LONG_SESSION = f"claude_code:{CAP_TENANT}:5b2f0f7e-1c1d-4e55-9a7c-2b0c8f3d4e10"
OTHER_DOCS = [custom_ingest_doc_id(CAP_TENANT, "experiments", f"run:{i}") for i in range(10)]


@pytest_asyncio.fixture
async def seeded_long_doc(pg_search_db):
    """One session whose 300 body chunks all match densely, and 10 documents
    with 2 body chunks that match once each, below every one of them. A
    40-chunk answer is all session unless each document is capped.

    Every document also has its `kind='metadata'` chunk, as ingest writes
    one. The session's is short and dense, so it scores highest of all --
    the shape that, counted in the cap, cost a result one of its snippets."""
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $1, $1) ON CONFLICT DO NOTHING",
            CAP_TENANT,
        )
        await ensure_tenant_partition(conn, CAP_TENANT)
        await _doc(conn, LONG_SESSION, "claude_code", None, tenant=CAP_TENANT)
        await _chunk(
            conn, LONG_SESSION, 9999, " ".join([CAP_TERM] * 4), tenant=CAP_TENANT,
            kind="metadata",
        )
        for n in range(300):
            await _chunk(
                conn, LONG_SESSION, n, f"{CAP_TERM} {CAP_TERM} {CAP_TERM} s{n}", tenant=CAP_TENANT
            )
        for i, doc_id in enumerate(OTHER_DOCS):
            await _doc(conn, doc_id, "custom_ingest", "experiments", tenant=CAP_TENANT)
            await _chunk(
                conn, doc_id, 2000 + i, f"{CAP_TERM} metadata {_FILLER}", tenant=CAP_TENANT,
                kind="metadata",
            )
            for part in range(2):
                await _chunk(
                    conn, doc_id, 1000 + 10 * i + part, f"{CAP_TERM} {part} {_FILLER}",
                    tenant=CAP_TENANT,
                )
        await _ensure_app_role(conn)
    yield


@pytest.mark.parametrize("override", [None, CHUNKS_PARENT], ids=["partition", "parent"])
async def test_max_chunks_per_doc_hands_the_slots_to_other_documents(
    seeded_long_doc, as_app_role, override
) -> None:
    """top_k=40 -> a 400-row pool holding all 331 matches. Uncapped, the 40
    best chunks are all the session's; capped at 2, every document keeps 2
    body chunks and its metadata chunk, which does not count against them,
    and the 10 other documents fill in behind the session."""
    uncapped = await bm25_search(
        CAP_TENANT, CAP_TERM, top_k=40, _scan_target_override=override
    )
    assert {h.doc_id for h in uncapped} == {LONG_SESSION}
    capped = await bm25_search(
        CAP_TENANT, CAP_TERM, top_k=40, max_chunks_per_doc=2, _scan_target_override=override
    )
    assert {h.doc_id for h in capped} == {LONG_SESSION, *OTHER_DOCS}
    for doc_id in (LONG_SESSION, *OTHER_DOCS):
        kinds = sorted(h.kind for h in capped if h.doc_id == doc_id)
        assert kinds == ["content", "content", "metadata"], doc_id
    # Best-scored first: all three of the session's before anything else.
    assert {h.doc_id for h in capped[:3]} == {LONG_SESSION}


async def test_a_bm25_only_direct_answer_is_not_one_document_wide(
    seeded_long_doc, as_app_role
) -> None:
    """End to end through /retrieve/direct's adapter: a bm25-only request asks
    for the cap, so the session is one result among many -- a caller that
    excludes its own live session still has an answer."""
    from engine.retrieval import direct

    response = await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query=CAP_TERM, top_k=10, channels=["bm25"]), CAP_TENANT
    )
    assert not response.degraded, response.lost_channels
    doc_ids = [r.doc_id for r in response.results]
    assert len(set(doc_ids)) >= 8
    assert doc_ids[0] == LONG_SESSION
    # Two snippets each: the metadata chunk, which a result never shows, did
    # not take one of the two slots.
    assert [r.chunk_count for r in response.results] == [2] * len(doc_ids)


# ---------------------------------------------------------------------------
# SQL shape, no database: the flag-off query is untouched.
# ---------------------------------------------------------------------------


class _Recording:
    def __init__(self, v3: bool = False) -> None:
        self.v3 = v3
        self.sql: str | None = None
        self.params: tuple[Any, ...] = ()

    async def fetchval(self, sql: str, *params: Any) -> Any:
        # The v3 probe answers as asked; the partition probe says "not
        # partitioned", so the pool scans `chunks`.
        return self.v3 if "indisvalid" in sql else None

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.sql, self.params = sql, params
        return []


async def _render(module: Any, v3: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The SQL and params `module.bm25_search` sends for `kwargs`.

    Takes the module so the same harness renders the pre-flag bm25.py for the
    golden below; patched and restored by hand for the same reason.
    """
    conn = _Recording(v3)

    @asynccontextmanager
    async def fake(_customer_id: str):  # type: ignore[no-untyped-def]
        yield conn

    real = module.with_tenant
    module.with_tenant = fake
    module._bm25_v3_available = module._CHUNKS_PARTITIONED = None
    try:
        await module.bm25_search("probe", "loss curve diverged", **kwargs)
    finally:
        module.with_tenant = real
        module._bm25_v3_available = module._CHUNKS_PARTITIONED = None
    return {"sql": conn.sql, "params": json.loads(json.dumps(list(conn.params), default=str))}


#: The SQL e97cf0e's bm25.py (the commit before `index_side_doc_filters`)
#: sent for each case, rendered by `_render` and stored. With the flag off the
#: current module must send exactly this: the gatherer and every other caller
#: never pass it. A DELIBERATE change to the flag-off SQL re-renders the file
#: from the new module and shows the diff in review.
GOLDEN = Path(__file__).parents[1] / "fixtures" / "bm25_flag_off_sql_e97cf0e.json"

GOLDEN_CASES: dict[str, tuple[bool, dict[str, Any]]] = {
    "unscoped": (False, {}),
    "sources": (False, {"top_k": 80, "sources": ["custom_ingest"]}),
    "every_doc_scope_v3": (
        True,
        {
            "top_k": 80,
            "sources": ["custom_ingest", "github"],
            "source_keys": ["experiments", "shared:probe"],
            "doc_types": ["custom.experiment.run"],
            "project_id": "240f2b75-a2ee-4189-ad05-9883bd5514a5",
        },
    ),
    "keyless_keys_per_source_recency": (
        False,
        {
            "top_k": 20,
            "source_keys": ["experiments"],
            "source_keys_include_keyless": True,
            "per_source_top_k": 3,
            "sort_by": "recency",
        },
    ),
    "all_versions_authors_drafts_v3": (
        True,
        {
            "top_k": 5,
            "sources": ["claude_code"],
            "author_ids": ["u1"],
            "include_drafts": True,
            "temporal": TemporalSpec(mode=TemporalMode.ALL),
        },
    ),
}


async def test_flag_off_sql_is_byte_identical_to_before_the_flag() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(GOLDEN_CASES)
    for name, (v3, kwargs) in GOLDEN_CASES.items():
        for flag in ({}, {"index_side_doc_filters": False, "max_chunks_per_doc": None}):
            assert await _render(bm25, v3, {**kwargs, **flag}) == golden[name], (name, flag)


SCOPE = {
    "top_k": 80,
    "sources": ["custom_ingest", "github"],
    "source_keys": ["experiments", "shared:probe"],
}


async def test_flag_off_sends_no_regex_leg_and_keeps_the_4x_pool() -> None:
    off = await _render(bm25, False, SCOPE)
    assert "paradedb.regex" not in off["sql"]
    assert "const_score" not in off["sql"]
    assert 80 * bm25._BM25_POOL_MULTIPLIER * bm25._BM25_SCOPED_POOL_FACTOR in off["params"]


async def test_flag_on_adds_two_must_legs_and_drops_only_their_pool_factor() -> None:
    on = await _render(bm25, False, {**SCOPE, "index_side_doc_filters": True})
    sql, params = on["sql"], on["params"]
    leg = "paradedb.const_score(0.0, paradedb.boolean(should => ARRAY[paradedb.regex('chunk_id', $"
    assert sql.count(leg) == 2
    # Inside the must array, before the ranking should-boolean -- a bare
    # `should` beside `must` would make it optional.
    must = sql.index("paradedb.boolean(must => ARRAY[")
    assert must < sql.index("paradedb.regex") < sql.index("paradedb.boolean(should => ARRAY[\n")
    assert r"(custom_ingest\x{3a}|github\x{3a})(?s:.)*" in params
    assert (
        r"(custom_ingest\x{3a}probe\x{3a}experiments\x{3a}"
        r"|custom_ingest\x{3a}probe\x{3a}shared\x{25}3Aprobe\x{3a})(?s:.)*"
    ) in params
    # Both scopes ride the index exactly, so the pool is the unscoped size...
    assert 80 * bm25._BM25_POOL_MULTIPLIER in params
    # ...and the SQL predicates still run as the correctness filter.
    assert "d.source_system = ANY(" in sql
    assert "d.metadata->>'source_key' = ANY(" in sql
    # doc_types still post-filters, so it brings the 4x back.
    typed = await _render(
        bm25, False, {**SCOPE, "index_side_doc_filters": True, "doc_types": ["custom.note"]}
    )
    assert 80 * bm25._BM25_POOL_MULTIPLIER * bm25._BM25_SCOPED_POOL_FACTOR in typed["params"]


async def test_the_document_cap_wraps_the_pool_and_leaves_its_limit_alone() -> None:
    plain = await _render(bm25, False, {"top_k": 40})
    capped = await _render(bm25, False, {"top_k": 40, "max_chunks_per_doc": 2})
    assert "_doc_rn" not in plain["sql"]
    assert capped["sql"].count("PARTITION BY t.doc_id, (t.kind = 'metadata')") == 1
    assert capped["params"] == [*plain["params"], 2]
    # The pool subquery -- the part pg_search executes as TopK -- is untouched.
    pool = plain["sql"][plain["sql"].index("SELECT c.chunk_id") : plain["sql"].index(") p")]
    assert pool in capped["sql"]


def test_prefixes_split_under_the_budget_and_each_lands_once() -> None:
    raw = {
        bm25._regex_literal(p): p
        for p in (bm25._source_key_chunk_prefix("probe", k) for k in _max_length_keys())
    }
    regexes = bm25._chunk_id_prefix_regexes(raw.values())
    assert len(regexes) > 1
    seen: list[str] = []
    for regex in regexes:
        assert regex.startswith("(") and regex.endswith(")(?s:.)*")
        group = regex[1 : -len(")(?s:.)*")].split("|")
        assert sum(len(raw[g].encode()) for g in group) <= bm25._REGEX_PREFIX_BUDGET_BYTES
        seen.extend(group)
    assert sorted(seen) == sorted(raw)
    params: list[Any] = []
    leg = bm25._chunk_id_prefix_leg(params, raw.values())
    assert params == regexes
    assert leg.count("paradedb.regex('chunk_id', $") == len(regexes)


def test_regex_literal_escapes_everything_but_word_characters() -> None:
    assert bm25._regex_literal("custom_ingest") == "custom_ingest"
    assert bm25._regex_literal("a.b*c") == r"a\x{2e}b\x{2a}c"
    assert bm25._regex_literal("é\n") == r"\x{e9}\x{a}"
    assert bm25._chunk_id_prefix_regexes(["a:", "b:", "a:"]) == [r"(a\x{3a}|b\x{3a})(?s:.)*"]

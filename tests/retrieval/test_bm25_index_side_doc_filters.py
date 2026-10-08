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
pre-filter's exactness rests on exactly those two facts.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio

import engine.shared.db as db_module
from engine.retrieval.retrievers import bm25
from engine.retrieval.retrievers.bm25 import bm25_search
from engine.shared.custom_ingest import custom_ingest_doc_id
from engine.shared.partitions import CHUNKS_PARENT, ensure_tenant_partition, is_partitioned

#: Hyphenated on purpose: the tenant id is tokenized on hyphens by the index,
#: and it is interpolated into every source-key prefix.
TENANT = "doc-scope-tenant"
TERM = "zephyrine"
WORKSPACE_KEY = "workspace:1d155c9c-4f05-4707-98a7-f69763c171e0"
#: Shares a prefix with "experiments" up to the separator, so a prefix that
#: forgot the trailing ':' would leak it.
LOOKALIKE_KEY = "experiments-archive"
TRANSCRIPT_CHUNKS = 120
TOP_K = 2
APP_ROLE = "bm25_doc_scope_app"

EXPERIMENT_DOC = custom_ingest_doc_id(TENANT, "experiments", "page:738f0c2e")
WORKSPACE_DOC = custom_ingest_doc_id(TENANT, WORKSPACE_KEY, "file:notes.md")
LOOKALIKE_DOC = custom_ingest_doc_id(TENANT, LOOKALIKE_KEY, "page:1")
CUSTOM_DOCS = {EXPERIMENT_DOC, WORKSPACE_DOC, LOOKALIKE_DOC}
_FILLER = " ".join(f"filler{i}" for i in range(80))


async def _doc(conn, doc_id: str, source: str, source_key: str | None) -> None:
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
        TENANT,
        doc_id,
        source,
        source_key,
    )


async def _chunk(conn, doc_id: str, n: int, content: str) -> None:
    await conn.execute(
        """
        INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index, content,
                            content_hash, token_count, chunker_version,
                            first_seen_version, last_seen_version, kind, visibility)
        VALUES ($1, $2, $3, $4, $5, $6, 3, 'v1', 1, 2147483647, 'content', 'approved')
        """,
        f"{doc_id}:c_{n:016x}",
        doc_id,
        TENANT,
        n,
        content,
        f"h{n}",
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
        ):
            await _doc(conn, doc_id, "custom_ingest", key)
            # One mention in a long body: every transcript chunk outranks it.
            await _chunk(conn, doc_id, 0, f"{TERM} {_FILLER}")
        sessions = [f"claude_code:{TENANT}:3c325e11-2008-46a9-83f7-{i:012x}" for i in range(4)]
        for doc_id in sessions:
            await _doc(conn, doc_id, "claude_code", None)
        for n in range(TRANSCRIPT_CHUNKS):
            await _chunk(conn, sessions[n % len(sessions)], n, f"{TERM} {TERM} {TERM} t{n}")
        await conn.execute(
            f"""
            DO $$ BEGIN
                CREATE ROLE {APP_ROLE} NOSUPERUSER NOBYPASSRLS;
            EXCEPTION WHEN duplicate_object THEN NULL; END $$
            """
        )
        await conn.execute(f"GRANT USAGE ON SCHEMA public, paradedb TO {APP_ROLE}")
        await conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")
    yield


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
        bm25._chunk_id_prefix_regex([bm25._source_chunk_prefix("custom_ingest")]),
    )
    assert by_token == 0
    assert by_prefix == len(CUSTOM_DOCS)


async def test_the_key_leg_alone_is_exact(seeded) -> None:
    """The index leg by itself, without the SQL filter behind it.

    Search results cannot show a loose pre-filter -- the SQL source_key
    predicate removes whatever it lets through -- so this counts what the leg
    matches in the index. A key with ':' still matches through its `%3A`
    encoding, and `experiments` does not reach `experiments-archive`.
    """
    for key, expected in (("experiments", 1), (WORKSPACE_KEY, 1), (LOOKALIKE_KEY, 1)):
        regex = bm25._chunk_id_prefix_regex([bm25._source_key_chunk_prefix(TENANT, key)])
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
    separator does not."""
    for keys, expected in (
        (["experiments"], {EXPERIMENT_DOC}),
        ([WORKSPACE_KEY], {WORKSPACE_DOC}),
        (["experiments", WORKSPACE_KEY], {EXPERIMENT_DOC, WORKSPACE_DOC}),
    ):
        hits = await bm25_search(
            TENANT, TERM, top_k=10, source_keys=keys, index_side_doc_filters=True,
            _scan_target_override=override,
        )
        assert {h.doc_id for h in hits} == expected, keys
        # The same scope, also narrowed by source: both legs ride the index.
        both = await bm25_search(
            TENANT, TERM, top_k=10, sources=["custom_ingest"], source_keys=keys,
            index_side_doc_filters=True, _scan_target_override=override,
        )
        assert {h.doc_id for h in both} == expected, keys


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
# SQL shape, no database: the flag-off query is untouched.
# ---------------------------------------------------------------------------


class _Recording:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[Any, ...] = ()

    async def fetchval(self, sql: str, *params: Any) -> Any:
        return None  # not partitioned, no v3 index: the parent, the join

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.sql, self.params = sql, params
        return []


async def _recorded(monkeypatch, **kwargs: Any) -> _Recording:
    conn = _Recording()

    @asynccontextmanager
    async def fake(_customer_id: str):  # type: ignore[no-untyped-def]
        yield conn

    monkeypatch.setattr(bm25, "with_tenant", fake)
    await bm25_search("probe", "loss curve", **kwargs)
    return conn


SCOPE = {
    "top_k": 80,
    "sources": ["custom_ingest", "github"],
    "source_keys": ["experiments", "shared:probe"],
}


async def test_flag_off_sql_is_the_sql_every_other_caller_sends(monkeypatch) -> None:
    """The agentic gatherer never passes the flag. Its query must not move:
    no regex leg, no new parameter, the same 4x scoped pool."""
    default = await _recorded(monkeypatch, **SCOPE)
    explicit = await _recorded(monkeypatch, index_side_doc_filters=False, **SCOPE)
    assert (default.sql, default.params) == (explicit.sql, explicit.params)
    assert "paradedb.regex" not in default.sql
    assert "const_score" not in default.sql
    assert not any(isinstance(p, str) and p.endswith(".*") for p in default.params)
    assert 80 * bm25._BM25_POOL_MULTIPLIER * bm25._BM25_SCOPED_POOL_FACTOR in default.params


async def test_flag_on_adds_two_must_legs_and_drops_only_their_pool_factor(monkeypatch) -> None:
    on = await _recorded(monkeypatch, index_side_doc_filters=True, **SCOPE)
    assert on.sql.count("paradedb.const_score(0.0, paradedb.regex('chunk_id', $") == 2
    # Inside the must array, before the ranking should-boolean -- a bare
    # `should` beside `must` would make it optional.
    must = on.sql.index("paradedb.boolean(must => ARRAY[")
    assert must < on.sql.index("paradedb.regex") < on.sql.index("paradedb.boolean(should")
    assert r"(custom_ingest\x{3a}|github\x{3a}).*" in on.params
    assert (
        r"(custom_ingest\x{3a}probe\x{3a}experiments\x{3a}"
        r"|custom_ingest\x{3a}probe\x{3a}shared\x{25}3Aprobe\x{3a}).*"
    ) in on.params
    # Both scopes ride the index exactly, so the pool is the unscoped size...
    assert 80 * bm25._BM25_POOL_MULTIPLIER in on.params
    # ...and the SQL predicates still run as the correctness filter.
    assert "d.source_system = ANY(" in on.sql
    assert "d.metadata->>'source_key' = ANY(" in on.sql
    # doc_types still post-filters, so it brings the 4x back.
    typed = await _recorded(
        monkeypatch, index_side_doc_filters=True, doc_types=["custom.note"], **SCOPE
    )
    assert 80 * bm25._BM25_POOL_MULTIPLIER * bm25._BM25_SCOPED_POOL_FACTOR in typed.params


def test_regex_literal_escapes_everything_but_word_characters() -> None:
    assert bm25._regex_literal("custom_ingest") == "custom_ingest"
    assert bm25._regex_literal("a.b*c") == r"a\x{2e}b\x{2a}c"
    assert bm25._regex_literal("é") == r"\x{e9}"
    assert bm25._chunk_id_prefix_regex(["a:", "b:", "a:"]) == r"(a\x{3a}|b\x{3a}).*"

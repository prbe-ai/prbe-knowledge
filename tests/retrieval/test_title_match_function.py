"""kb_match_document_titles_multi_v1() (plan T12, decision D76, migration 0146).

Pins: the function returns exactly the rows, in the same order, of grounding's
inline title statements -- under all three owners it can have (a superuser, as
this suite connects; a BYPASSRLS non-superuser, as `resolver` after the
hand-over; a plain role RLS applies to, as `app` before it) -- on data built to
sit on the edges: similarities exactly at the floor, full-text-only hits,
NULL/empty titles, closed versions, other tenants, ties broken by doc_id, short
and long probes, full-text hits above and below the cap. Also: the body creates
as a non-superuser and matches db/schema.sql byte for byte; the tenant comes
only from the GUC; the engine falls back on a missing or failing function
without losing its transaction; the migration's index build is idempotent.
"""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import pytest
import sqlalchemy as sa

import engine.retrieval.grounding as grounding
import engine.shared.db as db_module

REPO = Path(__file__).resolve().parents[2]
MIGRATION = REPO / "db/migrations/versions/20260930_0146_kb_match_document_titles.py"
FN = "kb_match_document_titles_multi_v1"
SIGNATURE = f"{FN}(text[], real, integer)"

TENANT = "t12-titles-a"
OTHER = "t12-titles-b"
SMALL = "t12-titles-c"

_VOCAB = [
    "session", "sessions", "retry", "retries", "grounding", "ground", "partition",
    "partitions", "timeout", "deploy", "search", "auth", "index", "latency",
    "pipeline", "token", "tokenizer", "title", "match", "probe", "kb", "xy",
    "engine", "retrieval", "trigram", "window", "cap", "floor",
]
_EXTRA = ["alpha", "beta", "gamma", "delta", "omega", "zeta"]

#: Probe sets: single words (< 16 trigrams), typos, words with many and with no
#: full-text hits, multi-word and path probes (>= 16 trigrams), a stopword, a
#: punctuation-only probe, a duplicate, and the two exact-threshold probes.
PROBE_SETS = [
    ["retry", "grounding", "partition", "timeout", "session"],
    ["sesion", "groundng", "retri", "partiton", "timout"],
    ["xy", "xy abcdef", "the", "!!", "retry"],
    ["engine retrieval grounding", "session partition timeout", "kb/match/title.py"],
    ["engine retreival grounding", "tokenizer pipeline latncy", "retry", "kb/mtch/title.py"],
    ["tokenizer pipeline latency", "search", "search", "omega"],
    ["zzzz-no-such-token"],
]


class _Log:
    """Stands in for grounding's structlog logger: records warnings."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kw) -> None:
        self.warnings.append((event, kw.get("extra", {})))


def _migration_text() -> str:
    return MIGRATION.read_text()


def _body() -> str:
    return re.search(r'MATCH_TITLES_SQL = r"""(.*?)"""', _migration_text(), re.S).group(1)


def test_schema_sql_carries_0146_body_verbatim():
    schema = (REPO / "db/schema.sql").read_text()
    assert _body() in schema, "kb_match_document_titles_multi_v1 drifted between 0146 and db/schema.sql"


def test_schema_sql_declares_the_index_and_statistics():
    schema = re.sub(r"\s+", " ", (REPO / "db/schema.sql").read_text())
    assert (
        "CREATE INDEX idx_documents_title_trgm_count ON documents "
        "(customer_id, (array_length(show_trgm(title), 1))) WHERE valid_to IS NULL;"
    ) in schema
    assert (
        "CREATE STATISTICS IF NOT EXISTS documents_title_trgm_count_stx "
        "ON (array_length(show_trgm(title), 1)) FROM documents;"
    ) in schema


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------


def _title(rng: random.Random) -> str | None:
    roll = rng.random()
    if roll < 0.03:
        return None
    if roll < 0.06:
        return ""
    if roll < 0.10:
        return "/".join(rng.choice(_VOCAB) for _ in range(rng.randint(2, 4))) + ".py"
    words = rng.randint(1, 5)
    return " ".join(rng.choice(_VOCAB + _EXTRA) for _ in range(words))


async def _seed() -> None:
    rng = random.Random(146)
    base = datetime(2026, 9, 30, tzinfo=UTC)
    rows = []
    for tenant, n in ((TENANT, 400), (OTHER, 150), (SMALL, 12)):
        for i in range(n):
            title = _title(rng)
            # body_preview carries words the title may not: full-text-only hits.
            preview = " ".join(rng.choice(_VOCAB + _EXTRA) for _ in range(rng.randint(0, 6)))
            # A handful of timestamps, so similarity and updated_at tie often
            # and doc_id has to break them.
            updated = base - timedelta(hours=rng.choice([0, 0, 1, 2, 3]))
            doc_id = f"custom:{tenant}:{rng.randint(0, 10**6):07d}:{i:04d}"
            closed = rng.random() < 0.25
            if closed:
                # A closed version with a matching title, then the live one.
                rows.append((doc_id, 1, tenant, title, preview, updated, updated))
                rows.append((doc_id, 2, tenant, _title(rng), preview, updated, None))
            else:
                rows.append((doc_id, 1, tenant, title, preview, updated, None))
    # Exactly on the threshold: similarity('xy', 'xy abcdef') = 3/10 = 0.3 in
    # both directions. Rounding the trigram-count window inward drops them.
    for k, title in enumerate(["xy", "xy abcdef", "xy abcdef", "xy"]):
        rows.append((f"custom:edge:{k}", 1, TENANT, title, "", base, None))
    # A title tied with others on every key but doc_id, crossing the cap.
    for k in range(6):
        rows.append((f"custom:tie:{5 - k}", 1, TENANT, "retry window", "", base, None))
    async with db_module.raw_conn() as conn:
        for tenant in (TENANT, OTHER, SMALL):
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, $1, $1)",
                tenant,
            )
        await conn.executemany(
            """
            INSERT INTO documents (
                doc_id, version, customer_id, source_system, source_id, source_url,
                doc_class, doc_type, content_type, content_hash, title, body_preview,
                body_size_bytes, body_token_count, created_at, updated_at, valid_from,
                valid_to, ingested_at, acl
            ) VALUES (
                $1, $2, $3, 'custom_ingest', $1, '/x', 'raw_source', 'custom.document',
                'text/markdown', 'h-' || $1 || '-' || $2::int::text, $4, $5, 10, 0, $6, $6, $6, $7, $6, '{}'::jsonb
            )
            """,
            rows,
        )
        await conn.execute("ANALYZE documents")


def _from_candidates(per_probe) -> list[list[tuple]]:
    return [
        [(c.canonical_id, c.display_name, c.last_seen_at) for c in cands] for cands in per_probe
    ]


def _from_rows(rows, n_probes: int) -> list[list[tuple]]:
    out: list[list[tuple]] = [[] for _ in range(n_probes)]
    for r in rows:
        out[r["ord"] - 1].append((r["doc_id"], r["title"], r["updated_at"]))
    return out


async def _inline_multi(monkeypatch, tenant: str, probes: list[str], cap: int):
    """grounding.py's own inline statement: the function switched off."""
    monkeypatch.setattr(grounding, "_title_match_fn_exists", False)
    return _from_candidates(
        await grounding._fuzzy_match_document_titles_multi(tenant, probes, cap=cap)
    )


_CALL = f"""
    SELECT f.ord, f.doc_id, f.source_system, f.title, f.updated_at
    FROM {FN}($1::text[], $2::real, $3::int) WITH ORDINALITY AS f
    ORDER BY f.ord, f.ordinality
"""


async def _call_as(conn, tenant: str, probes: list[str], cap: int):
    await conn.execute(db_module.TENANT_BIND_SQL, tenant)
    await conn.execute(f"SET LOCAL pg_trgm.similarity_threshold = {grounding._DOC_TITLE_TRGM_FLOOR}")
    rows = await conn.fetch(_CALL, probes, grounding._DOC_TITLE_TRGM_FLOOR, cap)
    return _from_rows(rows, len(probes))


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("cap", [4, 10])
async def test_engine_results_are_identical_with_and_without_the_function(live_db, monkeypatch, cap):
    """The engine end to end, as this suite connects (superuser: the fast
    path). No fallback may fire, or the comparison would be inline vs inline."""
    await _seed()
    log = _Log()
    monkeypatch.setattr(grounding, "log", log)
    for tenant in (TENANT, OTHER, SMALL):
        for probes in PROBE_SETS:
            inline = await _inline_multi(monkeypatch, tenant, probes, cap)
            monkeypatch.setattr(grounding, "_title_match_fn_exists", True)
            via_fn = _from_candidates(
                await grounding._fuzzy_match_document_titles_multi(tenant, probes, cap=cap)
            )
            assert via_fn == inline, (tenant, probes)
            for probe in probes:
                tokens = probe.split()
                monkeypatch.setattr(grounding, "_title_match_fn_exists", False)
                single_inline = await grounding._fuzzy_match_document_titles(tenant, tokens, cap=cap)
                monkeypatch.setattr(grounding, "_title_match_fn_exists", True)
                single_fn = await grounding._fuzzy_match_document_titles(tenant, tokens, cap=cap)
                assert single_fn == single_inline, (tenant, tokens)
    assert log.warnings == []


@pytest.mark.integration
async def test_the_data_reaches_every_branch(live_db):
    """Guards the parity test against passing vacuously: the seed must put
    probes on both sides of the full-text cap and of the 16-trigram split,
    and rows exactly on the floor."""
    await _seed()
    async with db_module.raw_conn() as conn, conn.transaction():
        await conn.execute(db_module.TENANT_BIND_SQL, TENANT)
        fts = {
            p: await conn.fetchval(
                "SELECT count(*) FROM documents WHERE customer_id = $1 AND valid_to IS NULL "
                "AND title <> '' AND title_preview_tsv @@ plainto_tsquery('english', $2)",
                TENANT, p,
            )
            for p in ("retry", "groundng", "engine retreival grounding")
        }
        long_trigram_only = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id = $1 AND valid_to IS NULL "
            "AND title <> '' AND similarity(title, 'engine retreival grounding') >= 0.3::real",
            TENANT,
        )
        pn = {
            p: await conn.fetchval("SELECT array_length(show_trgm($1), 1)", p)
            for p in ("sesion", "engine retreival grounding", "xy abcdef")
        }
        on_floor = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id = $1 AND valid_to IS NULL "
            "AND similarity(title, 'xy abcdef') = 0.3::real",
            TENANT,
        )
    assert fts["retry"] >= 10 and fts["groundng"] < 4  # skip and fill
    assert fts["engine retreival grounding"] == 0 and long_trigram_only >= 1
    assert pn["sesion"] < 16 <= pn["engine retreival grounding"]
    assert pn["xy abcdef"] == 10 and on_floor >= 2


async def _scratch_copy_owned_by(conn, owner: str, *, bypass: bool) -> None:
    """Inside the caller's transaction: a role, the 0146 body created by it in a
    scratch schema, and a plain caller role allowed to execute it."""
    await conn.execute(f"CREATE ROLE {owner} NOLOGIN {'BYPASSRLS' if bypass else ''}")
    await conn.execute(f"GRANT USAGE ON SCHEMA public TO {owner}")
    await conn.execute(f"GRANT SELECT ON documents TO {owner}")
    await conn.execute(f"CREATE SCHEMA t12_fns AUTHORIZATION {owner}")
    await conn.execute(f"SET LOCAL ROLE {owner}")
    await conn.execute("SET LOCAL search_path = t12_fns, public")
    await conn.execute(_body())
    await conn.execute("RESET ROLE")
    await conn.execute("CREATE ROLE t12_caller NOLOGIN")
    await conn.execute("GRANT USAGE ON SCHEMA t12_fns TO t12_caller")
    await conn.execute(f"GRANT EXECUTE ON FUNCTION t12_fns.{SIGNATURE} TO t12_caller")
    await conn.execute("SET LOCAL ROLE t12_caller")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("owner", "bypass"),
    [
        # `resolver` after the hand-over: BYPASSRLS, not a superuser.
        ("t12_resolver", True),
        # `app` before it: RLS applies inside the function.
        ("t12_app", False),
    ],
)
async def test_identical_rows_as_a_non_superuser_owner(live_db, monkeypatch, owner, bypass):
    await _seed()
    expected = {
        (tenant, tuple(probes)): await _inline_multi(monkeypatch, tenant, probes, 4)
        for tenant in (TENANT, OTHER)
        for probes in PROBE_SETS
    }
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await _scratch_copy_owned_by(conn, owner, bypass=bypass)
            branch = await conn.fetchval(
                f"SELECT proowner::regrole::text FROM pg_proc WHERE oid = 't12_fns.{SIGNATURE}'::regprocedure"
            )
            assert branch == owner
            for (tenant, probes), want in expected.items():
                async with conn.transaction():
                    got = await _call_as(conn, tenant, list(probes), 4)
                assert got == want, (owner, tenant, probes)
        finally:
            await tx.rollback()


@pytest.mark.integration
async def test_body_creates_as_a_non_superuser_and_leaves_caller_settings(live_db):
    # Prod migrates as `app`. 0144's first deploy failed on a function SET
    # clause only a superuser may use; this body must create as a plain role.
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await _scratch_copy_owned_by(conn, "t12_migrator", bypass=False)
            await conn.execute("SET LOCAL plan_cache_mode = force_generic_plan")
            await conn.execute("SET LOCAL search_path = t12_fns, public")
            await conn.execute(db_module.TENANT_BIND_SQL, TENANT)
            await conn.fetch(_CALL, ["retry"], 0.3, 4)
            assert await conn.fetchval("SHOW plan_cache_mode") == "force_generic_plan"
            assert await conn.fetchval("SHOW search_path") == "t12_fns, public"
            acl = await conn.fetchval(
                f"SELECT proacl::text FROM pg_proc WHERE oid = 't12_fns.{SIGNATURE}'::regprocedure"
            )
            # EXECUTE for the creating role and the grantee only, not PUBLIC.
            assert "=X/" in acl and not re.search(r"(^|[{,])=X", acl), acl
        finally:
            await tx.rollback()


# ---------------------------------------------------------------------------
# Tenant isolation (the fast path has no RLS behind it)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_tenant_comes_only_from_the_guc(live_db, settings):
    await _seed()
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await _scratch_copy_owned_by(conn, "t12_resolver", bypass=True)
            for tenant in (TENANT, OTHER, SMALL):
                async with conn.transaction():
                    await conn.execute(db_module.TENANT_BIND_SQL, tenant)
                    rows = await conn.fetch(_CALL, ["retry", "session", "xy"], 0.3, 50)
                    assert rows, tenant
                    assert all(r["doc_id"].startswith(("custom:edge", "custom:tie", f"custom:{tenant}:"))
                               for r in rows), tenant
                    if tenant != TENANT:
                        assert all(r["doc_id"].startswith(f"custom:{tenant}:") for r in rows)
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.current_customer_id', '', true)")
                assert await conn.fetch(_CALL, ["retry"], 0.3, 50) == []
        finally:
            await tx.rollback()
    # Never set on this session: current_setting(..., true) is NULL.
    fresh = await asyncpg.connect(settings.database_url)
    try:
        assert await fresh.fetchval("SELECT current_setting('app.current_customer_id', true)") is None
        assert await fresh.fetch(_CALL, ["retry"], 0.3, 50) == []
    finally:
        await fresh.close()


@pytest.mark.integration
async def test_invalid_floor_raises_and_empty_inputs_return_nothing(live_db):
    await _seed()
    async with db_module.raw_conn() as conn, conn.transaction():
        await conn.execute(db_module.TENANT_BIND_SQL, TENANT)
        # 1e-9 would overflow the count window's int bound; refused, not crashed.
        for floor in (0.0, -1.0, 1.5, 1e-9):
            with pytest.raises(asyncpg.InvalidParameterValueError, match="sim_floor must be in"):
                async with conn.transaction():
                    await conn.fetch(_CALL, ["retry"], floor, 4)
        assert await conn.fetch(_CALL, [], 0.3, 4) == []
        assert await conn.fetch(_CALL, ["retry"], 0.3, 0) == []


# ---------------------------------------------------------------------------
# Engine fallback
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_a_failing_function_falls_back_inside_a_savepoint(live_db, monkeypatch):
    await _seed()
    probes = ["retry", "sesion"]
    expected = await _inline_multi(monkeypatch, TENANT, probes, 4)
    log = _Log()
    monkeypatch.setattr(grounding, "log", log)
    monkeypatch.setattr(grounding, "_title_match_fn_exists", True)
    # Errors at execution, after the savepoint opened: the inline statement
    # must still run in the same transaction and answer.
    monkeypatch.setattr(
        grounding,
        "_TITLE_MATCH_FN_SQL",
        "SELECT 1 / 0 AS ord WHERE $1::text[] IS NOT NULL AND $2::real > 0 AND $3::int > 0",
    )
    got = _from_candidates(await grounding._fuzzy_match_document_titles_multi(TENANT, probes, cap=4))
    assert got == expected
    assert [w[0] for w in log.warnings] == ["grounding.title_function_fallback"]
    assert log.warnings[0][1]["error_type"] == "DivisionByZeroError"
    assert grounding._title_match_fn_exists is True


@pytest.mark.integration
async def test_a_dropped_function_stops_being_called(live_db, monkeypatch):
    await _seed()
    expected = await _inline_multi(monkeypatch, TENANT, ["retry"], 4)
    monkeypatch.setattr(grounding, "log", _Log())
    monkeypatch.setattr(grounding, "_title_match_fn_exists", True)
    monkeypatch.setattr(
        grounding, "_TITLE_MATCH_FN_SQL",
        "SELECT * FROM kb_match_no_such_fn($1::text[], $2::real, $3::int)",
    )
    got = _from_candidates(await grounding._fuzzy_match_document_titles_multi(TENANT, ["retry"], cap=4))
    assert got == expected
    assert grounding._title_match_fn_exists is False


@pytest.mark.integration
async def test_existence_is_asked_once_and_a_missing_function_is_not_called(live_db, monkeypatch):
    await _seed()
    expected = await _inline_multi(monkeypatch, TENANT, ["retry"], 4)
    monkeypatch.setattr(grounding, "_title_match_fn_exists", None)
    monkeypatch.setattr(grounding, "_TITLE_MATCH_FN", "public.kb_match_no_such_fn(text[], real, integer)")
    got = _from_candidates(await grounding._fuzzy_match_document_titles_multi(TENANT, ["retry"], cap=4))
    assert got == expected
    assert grounding._title_match_fn_exists is False
    monkeypatch.setattr(grounding, "_title_match_fn_exists", None)
    monkeypatch.setattr(grounding, "_TITLE_MATCH_FN", f"public.{SIGNATURE}")
    await grounding._fuzzy_match_document_titles_multi(TENANT, ["retry"], cap=4)
    assert grounding._title_match_fn_exists is True


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _load_0146():
    import importlib.util

    spec = importlib.util.spec_from_file_location("m0146", MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
async def test_0146_index_matches_schema_sql_and_rebuilds_a_wrong_one(live_db, settings):
    mod = _load_0146()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")
    name = mod.INDEX_NAME
    defn = f"SELECT pg_get_indexdef('public.{name}'::regclass)"
    try:
        with engine.connect() as bind:
            # db/schema.sql's index is the one the migration expects: left alone.
            assert bind.execute(sa.text(defn)).scalar() == mod.INDEX_DEFINITION
            oid = bind.execute(sa.text(f"SELECT 'public.{name}'::regclass::oid")).scalar()
            mod.run(bind)
            assert bind.execute(sa.text(f"SELECT 'public.{name}'::regclass::oid")).scalar() == oid
            # A same-named index without the tenant column is replaced.
            bind.execute(sa.text(f"DROP INDEX public.{name}"))
            bind.execute(sa.text(
                f"CREATE INDEX {name} ON public.documents ((array_length(show_trgm(title), 1)))"
            ))
            mod.run(bind)
            assert bind.execute(sa.text(defn)).scalar() == mod.INDEX_DEFINITION
            assert bind.execute(sa.text(
                "SELECT count(*) FROM pg_statistic_ext WHERE stxname = 'documents_title_trgm_count_stx'"
            )).scalar() == 1
            assert bind.execute(sa.text("SHOW lock_timeout")).scalar() == "0"
    finally:
        engine.dispose()


@pytest.mark.integration
async def test_0146_reinstall_accepts_a_handed_over_identical_copy(live_db, settings):
    """After the Job re-owns the function, `app` cannot replace it: a downgrade
    (which leaves it) and re-upgrade must accept the identical copy rather than
    fail the release, and must refuse a re-owned copy with another body."""
    mod = _load_0146()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as bind:
            bind.execute(sa.text(
                "DO $r$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kbm_owner') "
                "THEN CREATE ROLE kbm_owner NOLOGIN; END IF; END $r$"
            ))
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO kbm_owner"))
            bind.execute(sa.text("CREATE ROLE kbm_migrator NOLOGIN"))
            bind.execute(sa.text("GRANT CREATE ON SCHEMA public TO kbm_migrator"))
            bind.execute(sa.text("SET ROLE kbm_migrator"))
            try:
                mod.install_function(bind)  # identical, foreign-owned: accepted
            finally:
                bind.execute(sa.text("RESET ROLE"))
            # A foreign-owned copy with another body is refused.
            bind.execute(sa.text(
                f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO CURRENT_USER"
            ))
            bind.execute(sa.text(mod.MATCH_TITLES_SQL.replace("tenant text :=", "tenant  text :=")))
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO kbm_owner"))
            bind.execute(sa.text("SET ROLE kbm_migrator"))
            try:
                with pytest.raises(RuntimeError, match="_v2"):
                    mod.install_function(bind)
            finally:
                bind.execute(sa.text("RESET ROLE"))
            # Put the real body back under the suite's own role for later tests.
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO CURRENT_USER"))
            bind.execute(sa.text(mod.MATCH_TITLES_SQL))
            bind.execute(sa.text("REVOKE CREATE ON SCHEMA public FROM kbm_migrator"))
            bind.execute(sa.text("DROP ROLE kbm_migrator"))
    finally:
        engine.dispose()

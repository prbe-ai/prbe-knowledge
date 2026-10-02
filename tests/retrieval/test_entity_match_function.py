"""kb_match_entities_multi_v1() (plan T16, decision D77, migration 0147).

Pins: the function returns exactly the rows, in the same order, of grounding's
inline entity statements -- under all three owners it can have (a superuser, as
this suite connects; a BYPASSRLS non-superuser, as `resolver` after the
hand-over; a plain role RLS applies to, as `app` before it) -- on data built to
sit on the edges: similarities exactly at the threshold and exactly 0.5,
full-text-only hits (inside long names, above the per-label cap), trigram-only
hits, a full-text hit scoring between 0.5 and the threshold, labels outside
GROUNDING_ENTITY_LABELS, NULL/empty/punctuation names, mixed case and non-ASCII
names, one canonical_id under two labels, ties broken by canonical_id and
label, last_seen_at in several spellings, short and long probes, other tenants,
several thresholds and caps. Also: the body creates as a non-superuser and
matches db/schema.sql byte for byte; the tenant comes only from the GUC; an
unparsable last_seen_at errors in both; the engine falls back on a missing or
failing function without losing its transaction; the migration's index builds
are idempotent.
"""

from __future__ import annotations

import random
import re
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import pytest
import sqlalchemy as sa

import engine.retrieval.grounding as grounding
import engine.shared.db as db_module
from engine.shared.constants import GROUNDING_ENTITY_LABELS

REPO = Path(__file__).resolve().parents[2]
MIGRATION = REPO / "db/migrations/versions/20261002_0147_kb_match_entities.py"
FN = "kb_match_entities_multi_v1"
SIGNATURE = f"{FN}(text[], text[], integer, integer)"
LABELS = list(GROUNDING_ENTITY_LABELS)

TENANT = "t16-ent-a"
OTHER = "t16-ent-b"
SMALL = "t16-ent-c"
BROKEN = "t16-ent-bad-date"

_WORDS = [
    "retry", "retries", "window", "session", "sessions", "claude", "code", "grounding",
    "pipeline", "latency", "tokenizer", "match", "entity", "probe", "deploy", "xy", "abc",
    "kb", "index", "Retry", "SESSION", "Grounding",
]
_EXTRA = ["alpha", "beta", "gamma", "omega", "zeta", "ünïcode", "naïve", "Ärger", "sëssion"]
_OUTSIDE = ["Document", "CodeSymbol", "Repo"]
#: Several spellings, so last_seen_at_raw's TEXT order (the total cap) and
#: its timestamptz order (the per-label cap) disagree.
_SEEN = [
    "2026-09-30T12:00:00+00:00", "2026-09-30 12:00:00+00", "2026-09-29T08:00:00Z",
    "2026-10-01T00:00:00-07:00", "2026-09-30T05:00:00-07:00",
]

#: Single words (< 16 trigrams), typos, mixed case and non-ASCII, a stopword,
#: punctuation only, '', a duplicate, long multi-word and path probes
#: (>= 16 trigrams), and the probes the edge rows were built for.
PROBE_SETS = [
    ["retry", "session", "sesion", "claude", "xy"],
    ["retry window", "claude code session", "abc", "the", "!!"],
    ["Retry", "SESSION", "ünïcode", "naïve", "ärger"],
    ["xy abcdef", "abc", "abc", ""],
    ["kb/match/entity.py", "grounding pipeline latency tokenizer", "richard"],
    ["zzzz-no-such-token"],
]
#: (pg_trgm.similarity_threshold or None for the default, per_type_cap, total_cap).
#: 0.55 turns the 0.5 shortcut off; 0.005 takes the statement-as-is branch.
SETTINGS = [(None, 5, 20), (None, 2, 3), (0.2, 5, 20), (0.55, 5, 20), (0.005, 3, 7), (None, 1, 50)]


class _Log:
    """Stands in for grounding's structlog logger: records warnings."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kw) -> None:
        self.warnings.append((event, kw.get("extra", {})))


def _migration_text() -> str:
    return MIGRATION.read_text()


def _body() -> str:
    return re.search(r'MATCH_ENTITIES_SQL = r"""(.*?)"""', _migration_text(), re.S).group(1)


def test_schema_sql_carries_0147_body_verbatim():
    schema = (REPO / "db/schema.sql").read_text()
    assert _body() in schema, "kb_match_entities_multi_v1 drifted between 0147 and db/schema.sql"


def test_schema_sql_declares_the_indexes():
    schema = re.sub(r"\s+", " ", (REPO / "db/schema.sql").read_text())
    assert (
        "CREATE INDEX idx_graph_nodes_name_tsv ON graph_nodes USING gin "
        "(to_tsvector('english', coalesce(properties->>'name', '')));"
    ) in schema
    assert (
        "CREATE INDEX idx_graph_nodes_name_trgm_count ON graph_nodes "
        "(customer_id, label, (array_length(show_trgm(lower(properties->>'name')), 1)));"
    ) in schema


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------


def _name(rng: random.Random) -> str | None:
    roll = rng.random()
    if roll < 0.04:
        return None
    if roll < 0.06:
        return ""
    if roll < 0.08:
        return "!!"
    if roll < 0.18:
        person = rng.choice(["richard wei ", "alex chen ", ""])
        return f"{person}claude code session {rng.getrandbits(64):016x}"
    if roll < 0.23:
        return "/".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 4))) + ".py"
    return " ".join(rng.choice(_WORDS + _EXTRA) for _ in range(rng.randint(1, 5)))


def _props(name: str | None, rng: random.Random) -> dict:
    props: dict = {}
    if name is not None:
        props["name"] = name
    roll = rng.random()
    if roll < 0.2:
        props["last_seen_at"] = rng.choice(_SEEN)
    elif roll < 0.25:
        props["last_seen_at"] = None  # JSON null: ->> gives SQL NULL
    if rng.random() < 0.3:
        props["kind"] = rng.choice(["pr", "issue", "service", "person"])
    return props


async def _seed() -> None:
    import json

    rng = random.Random(147)
    rows: list[tuple[str, str, str, str]] = []

    def add(tenant: str, label: str, cid: str, props: dict) -> None:
        rows.append((tenant, label, cid, json.dumps(props)))

    for tenant, n in ((TENANT, 420), (OTHER, 160), (SMALL, 12)):
        for i in range(n):
            label = rng.choice(_OUTSIDE) if rng.random() < 0.15 else rng.choice(LABELS)
            add(tenant, label, f"{label.lower()}:{tenant}:{i:04d}", _props(_name(rng), rng))
    # Exactly on the default threshold: similarity('xy', 'xy abcdef') = 3/10.
    for k, name in enumerate(["xy", "xy abcdef", "xy abcdef", "xy"]):
        add(TENANT, "Person", f"edge:xy:{k}", {"name": name})
    # 'abcd' scores exactly 0.5 against 'abc' without being a full-text hit;
    # the long names are full-text-only hits for 'abc' at 0.5 as well.
    add(TENANT, "Feature", "edge:abcd", {"name": "abcd"})
    for k in range(3):
        add(TENANT, "Feature", f"edge:abc-long:{k}", {"name": f"abc and a much longer tail of words {k}"})
    # A full-text hit for 'retry' with similarity 6/11 = 0.545: a trigram
    # match at 0.3, full-text-only at 0.55, where it must still rank above 0.5.
    add(TENANT, "Decision", "edge:retry-wxyz", {"name": "retry wxyz"})
    # Tied on rel and recency: canonical_id decides who crosses the caps.
    for k in range(7):
        add(TENANT, "Service", f"tie:{6 - k}", {"name": "Retry Window"})
    for k in range(3):
        add(TENANT, "Service", f"tie-seen:{k}", {"name": "Retry Window", "last_seen_at": _SEEN[0]})
    # One canonical_id under two labels: label breaks the total-cap tie.
    for label in ("Person", "Feature"):
        add(TENANT, label, "shared:0", {"name": "retry window"})
    # Many more full-text hits than the per-label cap, none a trigram match.
    for k in range(30):
        add(TENANT, "AgentSession", f"sess:{k:02d}",
            {"name": f"richard wei claude code session {k:04d}-{rng.getrandbits(32):08x}"})
    # Exact matches under labels grounding never scans.
    for label in _OUTSIDE:
        for name in ("retry", "session", "claude code session"):
            add(TENANT, label, f"outside:{label}:{name}", {"name": name})
    # Only for the error test: a candidate whose last_seen_at does not parse.
    add(BROKEN, "Person", "bad:0", {"name": "retry", "last_seen_at": "not a date"})

    async with db_module.raw_conn() as conn:
        for tenant in (TENANT, OTHER, SMALL, BROKEN):
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, $1, $1)",
                tenant,
            )
        await conn.executemany(
            "INSERT INTO graph_nodes (customer_id, label, canonical_id, properties) "
            "VALUES ($1, $2, $3, $4::jsonb)",
            rows,
        )
        await conn.execute("ANALYZE graph_nodes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _from_candidates(per_probe) -> list[list[tuple]]:
    return [
        [(c.entity_type, c.canonical_id, c.display_name, c.last_seen_at, c.match_source) for c in cands]
        for cands in per_probe
    ]


def _threshold(monkeypatch, thr: float | None) -> None:
    """Make the engine's transaction run under `thr` (None: the default)."""
    if thr is None:
        return
    original = db_module.with_tenant

    @asynccontextmanager
    async def with_tenant(customer_id: str):
        async with original(customer_id) as conn:
            await conn.execute(f"SET LOCAL pg_trgm.similarity_threshold = {thr}")
            yield conn

    monkeypatch.setattr(grounding, "with_tenant", with_tenant)


async def _engine_multi(monkeypatch, use_fn: bool, tenant: str, probes, ptc: int, tc: int):
    monkeypatch.setattr(grounding, "_entity_match_fn_exists", use_fn)
    return _from_candidates(
        await grounding._fuzzy_match_entities_multi(tenant, probes, per_type_cap=ptc, total_cap=tc)
    )


async def _inline_sql(monkeypatch) -> str:
    """grounding.py's own multi-probe statement, as the engine passes it on."""
    captured: list[str] = []

    async def recorder(conn, customer_id, probes, labels, ptc, tc, inline_sql, *inline_args):
        captured.append(inline_sql)
        return await conn.fetch(inline_sql, *inline_args)

    with monkeypatch.context() as m:
        m.setattr(grounding, "_entity_rows", recorder)
        await grounding._fuzzy_match_entities_multi(TENANT, ["x"])
    return captured[0]


async def _bind(conn, tenant: str, thr: float | None) -> None:
    await conn.execute(db_module.TENANT_BIND_SQL, tenant)
    # Always set: inside a savepoint, an earlier case's SET LOCAL lasts until
    # the OUTER transaction ends.
    value = "DEFAULT" if thr is None else thr
    await conn.execute(f"SET LOCAL pg_trgm.similarity_threshold = {value}")


def _call(schema: str) -> str:
    return f"""
        SELECT f.ord, f.label, f.canonical_id, f.kind, f.display_name, f.last_seen_at_raw, f.rel
        FROM {schema}.{FN}($1::text[], $2::text[], $3::int, $4::int) WITH ORDINALITY AS f
        ORDER BY f.ord, f.ordinality
    """


async def _expected_rows(inline_sql: str) -> dict:
    """Inline rows for every (tenant, probes, setting), as the suite's role."""
    out = {}
    async with db_module.raw_conn() as conn:
        for tenant in (TENANT, OTHER, SMALL):
            for probes in PROBE_SETS:
                for thr, ptc, tc in SETTINGS:
                    async with conn.transaction():
                        await _bind(conn, tenant, thr)
                        rows = await conn.fetch(inline_sql, tenant, probes, LABELS, ptc, tc)
                    out[(tenant, tuple(probes), thr, ptc, tc)] = [tuple(r) for r in rows]
    return out


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_engine_results_are_identical_with_and_without_the_function(live_db, monkeypatch):
    """The engine end to end, as this suite connects (superuser: the fast
    path). No fallback may fire, or the comparison would be inline vs inline."""
    await _seed()
    log = _Log()
    monkeypatch.setattr(grounding, "log", log)
    for thr, ptc, tc in SETTINGS:
        with monkeypatch.context() as m:
            _threshold(m, thr)
            for tenant in (TENANT, OTHER, SMALL):
                for probes in PROBE_SETS:
                    inline = await _engine_multi(m, False, tenant, probes, ptc, tc)
                    via_fn = await _engine_multi(m, True, tenant, probes, ptc, tc)
                    assert via_fn == inline, (thr, ptc, tc, tenant, probes)
                    for probe in probes:
                        tokens = probe.split()
                        m.setattr(grounding, "_entity_match_fn_exists", False)
                        single_inline = await grounding._fuzzy_match_entities(
                            tenant, tokens, per_type_cap=ptc, total_cap=tc
                        )
                        m.setattr(grounding, "_entity_match_fn_exists", True)
                        single_fn = await grounding._fuzzy_match_entities(
                            tenant, tokens, per_type_cap=ptc, total_cap=tc
                        )
                        assert single_fn == single_inline, (thr, ptc, tc, tenant, tokens)
    assert log.warnings == []


@pytest.mark.integration
async def test_the_data_reaches_every_branch(live_db):
    """Guards the parity tests against passing vacuously."""
    await _seed()
    labels_sql = "label = ANY($2::text[])"
    async with db_module.raw_conn() as conn, conn.transaction():
        await conn.execute(db_module.TENANT_BIND_SQL, TENANT)

        async def count(where: str, *args) -> int:
            return await conn.fetchval(
                f"SELECT count(*) FROM graph_nodes WHERE customer_id = $1 AND {labels_sql} AND {where}",
                TENANT, LABELS, *args,
            )

        fts = "to_tsvector('english', coalesce(properties->>'name', '')) @@ plainto_tsquery('english', $3)"
        trgm = "similarity(lower(properties->>'name'), $3) >= 0.3::real"
        # More full-text-only hits in one label than the cap: the 0.5 shortcut.
        assert await count(f"label = 'AgentSession' AND {fts} AND NOT {trgm}", "session") > 5
        # Trigram-only hits (a typo) and full-text-only hits.
        assert await count(f"{trgm} AND NOT {fts}", "sesion") >= 1
        assert await count(f"{fts} AND NOT {trgm}", "retry") >= 1
        # Exactly on the threshold, and exactly 0.5 without a full-text hit.
        assert await count("similarity(lower(properties->>'name'), $3) = 0.3::real", "xy abcdef") >= 2
        assert await count(f"similarity(properties->>'name', $3) = 0.5::real AND NOT {fts}", "abc") == 1
        # A full-text hit between 0.5 and 0.55: full-text-only at 0.55.
        assert await count(
            f"{fts} AND similarity(properties->>'name', $3) > 0.5 "
            "AND similarity(properties->>'name', $3) < 0.55", "retry",
        ) >= 1
        # Exact matches outside the grounding labels exist (and must not appear).
        assert await conn.fetchval(
            "SELECT count(*) FROM graph_nodes WHERE customer_id = $1 AND NOT (label = ANY($2::text[])) "
            "AND lower(properties->>'name') = 'retry'", TENANT, LABELS,
        ) >= 3
        # Mixed case and non-ASCII names; both sides of the 16-trigram split.
        assert await count("properties->>'name' <> lower(properties->>'name')") >= 5
        assert await count("properties->>'name' ~ '[^[:ascii:]]'") >= 5
        pn = {
            p: await conn.fetchval("SELECT array_length(show_trgm($1), 1)", p)
            for p in ("sesion", "claude code session", "grounding pipeline latency tokenizer")
        }
    assert pn["sesion"] < 16 <= pn["claude code session"]


@pytest.mark.integration
async def test_names_fold_alike_in_pg_trgm_and_lower(live_db):
    """The 0.5 shortcut's premise, on this database: similarity() of a name
    equals similarity() of its lower(), and the locale provider is libc."""
    await _seed()
    async with db_module.raw_conn() as conn:
        assert await conn.fetchval(
            "SELECT datlocprovider = 'c' FROM pg_database WHERE datname = current_database()"
        )
        probes = sorted({p for ps in PROBE_SETS for p in ps})
        differing = await conn.fetchval(
            """
            SELECT count(*) FROM graph_nodes n CROSS JOIN unnest($1::text[]) AS p(p)
            WHERE similarity(coalesce(n.properties->>'name', ''), p.p)
                  IS DISTINCT FROM similarity(coalesce(lower(n.properties->>'name'), ''), p.p)
            """,
            probes,
        )
    assert differing == 0


async def _scratch_copy_owned_by(conn, owner: str, *, bypass: bool) -> None:
    """Inside the caller's transaction: a role, the 0147 body created by it in a
    scratch schema, and a plain caller role allowed to execute it."""
    await conn.execute(f"CREATE ROLE {owner} NOLOGIN {'BYPASSRLS' if bypass else ''}")
    await conn.execute(f"GRANT USAGE ON SCHEMA public TO {owner}")
    await conn.execute(f"GRANT SELECT ON graph_nodes TO {owner}")
    await conn.execute(f"CREATE SCHEMA t16_fns AUTHORIZATION {owner}")
    await conn.execute(f"SET LOCAL ROLE {owner}")
    await conn.execute("SET LOCAL search_path = t16_fns, public")
    await conn.execute(_body())
    await conn.execute("RESET ROLE")
    await conn.execute("CREATE ROLE t16_caller NOLOGIN")
    await conn.execute("GRANT USAGE ON SCHEMA t16_fns TO t16_caller")
    await conn.execute(f"GRANT EXECUTE ON FUNCTION t16_fns.{SIGNATURE} TO t16_caller")
    await conn.execute("SET LOCAL ROLE t16_caller")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("owner", "bypass"),
    [
        # The suite's own role: db/schema.sql's copy in public.
        (None, None),
        # `resolver` after the hand-over: BYPASSRLS, not a superuser.
        ("t16_resolver", True),
        # `app` before it: RLS applies inside the function.
        ("t16_app", False),
    ],
)
async def test_identical_rows_for_every_owner(live_db, monkeypatch, owner, bypass):
    """Whole rows (rel, kind, last_seen_at_raw included), not just what the
    engine maps them to, against grounding.py's inline statement."""
    await _seed()
    expected = await _expected_rows(await _inline_sql(monkeypatch))
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            schema = "public"
            if owner is not None:
                await _scratch_copy_owned_by(conn, owner, bypass=bypass)
                schema = "t16_fns"
                assert await conn.fetchval(
                    f"SELECT proowner::regrole::text FROM pg_proc WHERE oid = 't16_fns.{SIGNATURE}'::regprocedure"
                ) == owner
            for (tenant, probes, thr, ptc, tc), want in expected.items():
                async with conn.transaction():
                    await _bind(conn, tenant, thr)
                    got = [tuple(r) for r in await conn.fetch(_call(schema), list(probes), LABELS, ptc, tc)]
                assert got == want, (owner, tenant, probes, thr, ptc, tc)
        finally:
            await tx.rollback()
    assert any(want for want in expected.values())


@pytest.mark.integration
async def test_body_creates_as_a_non_superuser_and_leaves_caller_settings(live_db):
    # Prod migrates as `app`. 0144's first deploy failed on a function SET
    # clause only a superuser may use; this body must create as a plain role.
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await _scratch_copy_owned_by(conn, "t16_migrator", bypass=False)
            await conn.execute("SET LOCAL plan_cache_mode = force_generic_plan")
            await conn.execute("SET LOCAL search_path = t16_fns, public")
            await conn.execute(db_module.TENANT_BIND_SQL, TENANT)
            await conn.fetch(_call("t16_fns"), ["retry"], LABELS, 5, 20)
            assert await conn.fetchval("SHOW plan_cache_mode") == "force_generic_plan"
            assert await conn.fetchval("SHOW search_path") == "t16_fns, public"
            acl = await conn.fetchval(
                f"SELECT proacl::text FROM pg_proc WHERE oid = 't16_fns.{SIGNATURE}'::regprocedure"
            )
            # EXECUTE for the creating role and the grantee only, not PUBLIC.
            assert "=X/" in acl and not re.search(r"(^|[{,])=X", acl), acl
        finally:
            await tx.rollback()


# ---------------------------------------------------------------------------
# Tenant isolation and inputs (the fast path has no RLS behind it)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_tenant_comes_only_from_the_guc(live_db, settings):
    await _seed()
    got: dict[str, list] = {}
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await _scratch_copy_owned_by(conn, "t16_resolver", bypass=True)
            for tenant in (TENANT, OTHER, SMALL):
                async with conn.transaction():
                    await conn.execute(db_module.TENANT_BIND_SQL, tenant)
                    got[tenant] = await conn.fetch(
                        _call("t16_fns"), ["retry", "session", "xy"], LABELS, 50, 500
                    )
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.current_customer_id', '', true)")
                assert await conn.fetch(_call("t16_fns"), ["retry"], LABELS, 5, 20) == []
            await conn.execute("RESET ROLE")
            for tenant, rows in got.items():
                assert rows, tenant
                # (label, canonical_id) is unique across tenants in the seed.
                owners = await conn.fetch(
                    "SELECT DISTINCT customer_id FROM graph_nodes "
                    "WHERE (label, canonical_id) IN (SELECT * FROM unnest($1::text[], $2::text[]))",
                    [r["label"] for r in rows], [r["canonical_id"] for r in rows],
                )
                assert [o["customer_id"] for o in owners] == [tenant]
        finally:
            await tx.rollback()
    # Never set on this session: current_setting(..., true) is NULL.
    fresh = await asyncpg.connect(settings.database_url)
    try:
        assert await fresh.fetchval("SELECT current_setting('app.current_customer_id', true)") is None
        assert await fresh.fetch(_call("public"), ["retry"], LABELS, 5, 20) == []
    finally:
        await fresh.close()


@pytest.mark.integration
async def test_empty_inputs_and_caps_return_what_the_statement_does(live_db, monkeypatch):
    await _seed()
    inline_sql = await _inline_sql(monkeypatch)
    cases = [
        (["retry"], LABELS, 0, 20), (["retry"], LABELS, 5, 0), (["retry"], LABELS, -1, 20),
        (["retry"], LABELS, None, 20), (["retry"], LABELS, 5, None), ([], LABELS, 5, 20),
        (["retry"], [], 5, 20), (["retry", None, "session"], LABELS, 5, 20),
        (["retry"], ["Person", "Person", None, "Service"], 5, 20),
    ]
    async with db_module.raw_conn() as conn, conn.transaction():
        await conn.execute(db_module.TENANT_BIND_SQL, TENANT)
        for probes, labels, ptc, tc in cases:
            want = [tuple(r) for r in await conn.fetch(inline_sql, TENANT, probes, labels, ptc, tc)]
            got = [tuple(r) for r in await conn.fetch(_call("public"), probes, labels, ptc, tc)]
            assert got == want, (probes, labels, ptc, tc)
        assert await conn.fetch(_call("public"), None, LABELS, 5, 20) == []
        assert await conn.fetch(_call("public"), ["retry"], None, 5, 20) == []


@pytest.mark.integration
async def test_an_unparsable_last_seen_at_errors_in_both(live_db, monkeypatch):
    """Today's statement casts every candidate's last_seen_at; so does the
    function. The engine then falls back and raises what it raised before."""
    await _seed()
    inline_sql = await _inline_sql(monkeypatch)
    async with db_module.raw_conn() as conn:
        for sql, args in ((inline_sql, (BROKEN, ["retry"], LABELS, 5, 20)),
                          (_call("public"), (["retry"], LABELS, 5, 20))):
            with pytest.raises(asyncpg.InvalidDatetimeFormatError):
                async with conn.transaction():
                    await conn.execute(db_module.TENANT_BIND_SQL, BROKEN)
                    await conn.fetch(sql, *args)
    monkeypatch.setattr(grounding, "log", _Log())
    for use_fn in (False, True):
        monkeypatch.setattr(grounding, "_entity_match_fn_exists", use_fn)
        with pytest.raises(asyncpg.InvalidDatetimeFormatError):
            await grounding._fuzzy_match_entities_multi(BROKEN, ["retry"])


# ---------------------------------------------------------------------------
# Engine fallback
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_a_failing_function_falls_back_inside_a_savepoint(live_db, monkeypatch):
    await _seed()
    probes = ["retry", "session"]
    expected = await _engine_multi(monkeypatch, False, TENANT, probes, 5, 20)
    log = _Log()
    monkeypatch.setattr(grounding, "log", log)
    monkeypatch.setattr(grounding, "_entity_match_fn_exists", True)
    # Errors at execution, after the savepoint opened: the inline statement
    # must still run in the same transaction and answer.
    monkeypatch.setattr(
        grounding,
        "_ENTITY_MATCH_FN_SQL",
        "SELECT 1 / 0 AS ord WHERE $1::text[] IS NOT NULL AND $2::text[] IS NOT NULL "
        "AND $3::int > 0 AND $4::int > 0",
    )
    got = _from_candidates(await grounding._fuzzy_match_entities_multi(TENANT, probes))
    assert got == expected
    assert [w[0] for w in log.warnings] == ["grounding.entity_function_fallback"]
    assert log.warnings[0][1]["error_type"] == "DivisionByZeroError"
    assert grounding._entity_match_fn_exists is True


@pytest.mark.integration
async def test_a_dropped_function_stops_being_called(live_db, monkeypatch):
    await _seed()
    expected = await _engine_multi(monkeypatch, False, TENANT, ["retry"], 5, 20)
    monkeypatch.setattr(grounding, "log", _Log())
    monkeypatch.setattr(grounding, "_entity_match_fn_exists", True)
    monkeypatch.setattr(
        grounding, "_ENTITY_MATCH_FN_SQL",
        "SELECT * FROM kb_match_no_such_fn($1::text[], $2::text[], $3::int, $4::int)",
    )
    got = _from_candidates(await grounding._fuzzy_match_entities_multi(TENANT, ["retry"]))
    assert got == expected
    assert grounding._entity_match_fn_exists is False


@pytest.mark.integration
async def test_existence_is_asked_once_and_a_missing_function_is_not_called(live_db, monkeypatch):
    await _seed()
    expected = await _engine_multi(monkeypatch, False, TENANT, ["retry"], 5, 20)
    monkeypatch.setattr(grounding, "_entity_match_fn_exists", None)
    monkeypatch.setattr(
        grounding, "_ENTITY_MATCH_FN", "public.kb_match_no_such_fn(text[], text[], integer, integer)"
    )
    got = _from_candidates(await grounding._fuzzy_match_entities_multi(TENANT, ["retry"]))
    assert got == expected
    assert grounding._entity_match_fn_exists is False
    monkeypatch.setattr(grounding, "_entity_match_fn_exists", None)
    monkeypatch.setattr(grounding, "_ENTITY_MATCH_FN", f"public.{SIGNATURE}")
    await grounding._fuzzy_match_entities_multi(TENANT, ["retry"])
    assert grounding._entity_match_fn_exists is True


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _load_0147():
    import importlib.util

    spec = importlib.util.spec_from_file_location("m0147", MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
async def test_0147_indexes_match_schema_sql_and_rebuild_a_wrong_one(live_db, settings):
    mod = _load_0147()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")

    def oid(bind, name):
        return bind.execute(sa.text(f"SELECT 'public.{name}'::regclass::oid")).scalar()

    def definition(bind, name):
        return bind.execute(sa.text(f"SELECT pg_get_indexdef('public.{name}'::regclass)")).scalar()

    try:
        with engine.connect() as bind:
            # db/schema.sql's indexes are the ones the migration expects: left alone.
            before = {}
            for name, _create, want in mod.INDEXES:
                assert definition(bind, name) == want
                before[name] = oid(bind, name)
            mod.run(bind)
            assert {name: oid(bind, name) for name in before} == before
            # Same-named indexes of another shape are replaced.
            bind.execute(sa.text("DROP INDEX public.idx_graph_nodes_name_tsv"))
            bind.execute(sa.text(
                "CREATE INDEX idx_graph_nodes_name_tsv ON public.graph_nodes "
                "USING gin (to_tsvector('simple', coalesce(properties->>'name', '')))"
            ))
            bind.execute(sa.text("DROP INDEX public.idx_graph_nodes_name_trgm_count"))
            bind.execute(sa.text(
                "CREATE INDEX idx_graph_nodes_name_trgm_count ON public.graph_nodes "
                "(customer_id, (array_length(show_trgm(properties->>'name'), 1)))"
            ))
            mod.run(bind)
            for name, _create, want in mod.INDEXES:
                assert definition(bind, name) == want
            assert bind.execute(sa.text("SHOW lock_timeout")).scalar() == "0"
    finally:
        engine.dispose()


@pytest.mark.integration
async def test_0147_reinstall_accepts_a_handed_over_identical_copy(live_db, settings):
    """After the Job re-owns the function, `app` cannot replace it: a downgrade
    (which leaves it) and re-upgrade must accept the identical copy rather than
    fail the release, and must refuse a re-owned copy with another body."""
    mod = _load_0147()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as bind:
            bind.execute(sa.text(
                "DO $r$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kbe_owner') "
                "THEN CREATE ROLE kbe_owner NOLOGIN; END IF; END $r$"
            ))
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO kbe_owner"))
            bind.execute(sa.text("CREATE ROLE kbe_migrator NOLOGIN"))
            bind.execute(sa.text("GRANT CREATE ON SCHEMA public TO kbe_migrator"))
            bind.execute(sa.text("SET ROLE kbe_migrator"))
            try:
                mod.install_function(bind)  # identical, foreign-owned: accepted
            finally:
                bind.execute(sa.text("RESET ROLE"))
            # A foreign-owned copy with another body is refused.
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO CURRENT_USER"))
            bind.execute(sa.text(mod.MATCH_ENTITIES_SQL.replace("tenant text :=", "tenant  text :=")))
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO kbe_owner"))
            bind.execute(sa.text("SET ROLE kbe_migrator"))
            try:
                with pytest.raises(RuntimeError, match="_v2"):
                    mod.install_function(bind)
            finally:
                bind.execute(sa.text("RESET ROLE"))
            # Put the real body back under the suite's own role for later tests.
            bind.execute(sa.text(f"ALTER FUNCTION {mod.FUNCTION_SIGNATURE} OWNER TO CURRENT_USER"))
            bind.execute(sa.text(mod.MATCH_ENTITIES_SQL))
            bind.execute(sa.text("REVOKE CREATE ON SCHEMA public FROM kbe_migrator"))
            bind.execute(sa.text("DROP ROLE kbe_migrator"))
    finally:
        engine.dispose()

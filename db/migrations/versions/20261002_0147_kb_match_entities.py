"""kb_match_entities_multi_v1(): grounding's entity lookup, index-backed.

WHY. Grounding's entity channel (engine/retrieval/grounding.py, the two
`_fuzzy_match_entities*` matchers) runs as `app` under FORCE RLS on
graph_nodes. Neither `%` (similarity_op) nor `@@` (ts_match_vq) is LEAKPROOF,
so no trigram or full-text index may run ahead of the tenant policy: every
call reads every grounding-label node of the tenant and, per probe, rebuilds
to_tsvector() of its name and computes similarity(). In prod pg_stat_statements
(30 days) the multi-probe statement averaged 1,559 ms over 4,381 calls and the
single-probe one 406 ms over 5,199. Quiet plane, read-only EXPLAIN ANALYZE as
`app` (2026-10-02), 1/3/5 probes: `probe` 142-162/427-494/622-659 ms,
`new-workspace` 135-144/303-339/490-501 ms; the single-probe statement
131-183 ms on both.

WHAT. A SECURITY DEFINER function that returns exactly the rows of the
multi-probe statement, in the same order, for the same inputs. Once it is
owned by the BYPASSRLS role `resolver` (D71: a research-os Job re-owns
`kb_match_%` functions in kb after migrations; `app` is never a member of
`resolver`) the indexes become usable. Per probe it runs:

  1. the trigram leg, as node ids: names whose lower() is `%` the probe. One
     word (< 16 trigrams) reads the exact trigram-count window off
     idx_graph_nodes_name_trgm_count, behind an OFFSET 0 fence, then `%`
     decides:

       similarity(A, B) = |A n B| / |A u B| over the two trigram sets, and
       |A n B| <= min(|A|, |B|), |A u B| >= max(|A|, |B|). So similarity >= t
       implies t * |B| <= |A| <= |B| / t.

     Bounds rounded OUTWARD (floor/ceil; 0146 has the case inward rounding
     drops) and cast to int, or the index is not used. The indexed count is
     of lower(name), the very string `%` compares, so the window is exact
     whatever lower() does. Why the fence: idx_graph_nodes_name_trgm admits
     every name holding ~30% of a short probe's trigrams -- 'session' is in
     all 7,386 AgentSession names of `probe` -- and the planner, estimating
     `%`'s FINAL selectivity, takes it alone: T16 rig 40 ms through the GIN
     vs 5 ms through the window ('xy' 23 vs 1). Longer probes (multi-word: the
     single-probe matcher's input) leave it to the planner, which takes the
     GIN: 'lora training baseline' 5 ms vs 51 ms through the window.
  2. the full-text leg, through idx_graph_nodes_name_tsv, ANDed by the
     planner with idx_graph_nodes_customer_label. The expression is today's,
     character for character, or the index serves nothing.
  3. the ranking, unchanged: rel = GREATEST(similarity(name), 0.5 if a
     full-text hit), the top `per_type_cap` per label by (rel DESC,
     last_seen_at::timestamptz DESC NULLS LAST, canonical_id), then the top
     `total_cap` by (rel DESC, last_seen_at_raw DESC NULLS LAST, label,
     canonical_id) -- written as ORDER BY ... LIMIT per label present (a
     top-N sort) instead of a window over every candidate: same keys, same
     total order (UNIQUE (customer_id, label, canonical_id)), so the same
     rows. Every candidate's last_seen_at is cast, as today, so a value that
     does not parse errors in both.

     One shortcut: a full-text hit that is NOT a trigram match has rel = 0.5
     exactly whenever the threshold is <= 0.5, because its similarity is
     below the threshold. That skips similarity() for the bulk of a common
     word's hits ('session': 7,386 rows, ~35 ms on the T16 rig). It rests on
     similarity(name, p) = similarity(lower(name), p) -- `%` filters on the
     one, rel ranks on the other, as in today's statement. pg_trgm folds case
     with the database's libc ctype, which is what lower() uses under a libc
     default collation; under ICU lower() can fold differently, so the
     shortcut is taken only when the database's locale provider is libc
     (prod: libc, C).

While the function is still owned by `app` (between this migration and the
re-own, or on a plane where the Job never runs), RLS applies inside it and no
index can help, so under row_security_active() it runs the current statement
unchanged; likewise for a threshold below 0.01, where the window degenerates.
Same rows either way; the engine falls back to its own copy of the statement
if the function is missing or errors.

SECURITY. Tenant ONLY from `app.current_customer_id`, read once into a
variable (NULL or '' -> no rows), and every graph_nodes read is filtered on it
-- under `resolver` nothing else isolates tenants. search_path pinned.
EXECUTE revoked from PUBLIC and granted to the creating role. Reads ONE
table, public.graph_nodes, so the hand-over needs `GRANT SELECT ON
graph_nodes TO resolver` (beside 0146's documents grant) and, after the
re-own, `GRANT EXECUTE ON FUNCTION kb_match_entities_multi_v1(text[], text[],
integer, integer) TO app` (ALTER FUNCTION ... OWNER rewrites the creator's ACL
entry to the new owner). Without that grant every call is refused and the
engine falls back (correct, slow, one warning per call).

`plan_cache_mode = force_custom_plan`: each statement is planned with this
call's tenant, probe and trigram ids as constants. Both SET clauses name core
settings; a custom one would fail as `app` on PG15+ (0144's first deploy).

The threshold is the caller's pg_trgm.similarity_threshold (show_limit(),
which also loads pg_trgm), exactly what `%` reads in today's statement.

Named _v1 because once `resolver` owns it, `app` can neither replace nor drop
it: a change ships as _v2 beside it.

INDEXES. Both on the whole table (not partial): their expression statistics
are then collected by ANALYZE and used by the planner, and no partial
predicate has to track GROUNDING_ENTITY_LABELS. CONCURRENTLY, through 0143's
machinery (polled migrator lock, one wall-clock budget, definition check,
invalid-leftover rebuild), then ANALYZE so the statistics exist before the
first call. graph_nodes is not partitioned and stays so here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision = "0147_kb_match_entities"
down_revision = "0146_kb_match_titles"
branch_labels = None
depends_on = None

#: (name, CREATE INDEX CONCURRENTLY, pg_get_indexdef() of the result). Anything
#: else of that name is rebuilt. db/schema.sql declares the same two.
INDEXES: tuple[tuple[str, str, str], ...] = (
    (
        "idx_graph_nodes_name_tsv",
        "CREATE INDEX CONCURRENTLY idx_graph_nodes_name_tsv ON public.graph_nodes "
        "USING gin (to_tsvector('english', coalesce(properties->>'name', '')))",
        "CREATE INDEX idx_graph_nodes_name_tsv ON public.graph_nodes USING gin "
        "(to_tsvector('english'::regconfig, COALESCE((properties ->> 'name'::text), ''::text)))",
    ),
    (
        "idx_graph_nodes_name_trgm_count",
        "CREATE INDEX CONCURRENTLY idx_graph_nodes_name_trgm_count ON public.graph_nodes "
        "(customer_id, label, (array_length(show_trgm(lower(properties->>'name')), 1)))",
        "CREATE INDEX idx_graph_nodes_name_trgm_count ON public.graph_nodes USING btree "
        "(customer_id, label, array_length(show_trgm(lower((properties ->> 'name'::text))), 1))",
    ),
)

# db/schema.sql carries this body byte for byte (tests pin it).
MATCH_ENTITIES_SQL = r"""
CREATE OR REPLACE FUNCTION kb_match_entities_multi_v1(
    probes text[], labels text[], per_type_cap integer, total_cap integer
)
RETURNS TABLE (
    ord bigint, label text, canonical_id text, kind text, display_name text,
    last_seen_at_raw text, rel real
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
SET plan_cache_mode = force_custom_plan
AS $$
DECLARE
    tenant text := current_setting('app.current_customer_id', true);
    thr real;
    folds_alike boolean;
    probe text;
    i bigint := 0;
    q tsquery;
    pn integer;
    trgm_ids bigint[];
BEGIN
    IF tenant IS NULL OR tenant = '' OR probes IS NULL OR labels IS NULL
       OR per_type_cap IS NULL OR per_type_cap <= 0
       OR total_cap IS NULL OR total_cap <= 0 THEN
        RETURN;
    END IF;
    -- What `%` compares against: the caller's pg_trgm.similarity_threshold.
    thr := show_limit();

    -- Owned by a role RLS applies to (before the hand-over to `resolver`):
    -- no index can help, and per-probe legs would each re-read the tenant.
    -- Or a threshold the count window cannot express. Run grounding.py's
    -- statement as it is.
    IF row_security_active('public.graph_nodes'::regclass) OR NOT (thr >= 0.01) THEN
        RETURN QUERY
        WITH probe_list AS (
            SELECT u.p, u.ord FROM unnest(probes) WITH ORDINALITY AS u(p, ord)
        ),
        ranked AS (
            SELECT
                pl.ord,
                n.label, n.canonical_id,
                n.properties->>'kind' AS kind,
                coalesce(n.properties->>'name', n.canonical_id) AS display_name,
                n.properties->>'last_seen_at' AS last_seen_at_raw,
                GREATEST(
                    similarity(coalesce(n.properties->>'name',''), pl.p),
                    CASE
                        WHEN to_tsvector('english', coalesce(n.properties->>'name', ''))
                             @@ plainto_tsquery('english', pl.p) THEN 0.5
                        ELSE 0.0
                    END
                ) AS rel,
                row_number() OVER (
                    PARTITION BY pl.ord, n.label
                    ORDER BY GREATEST(
                        similarity(coalesce(n.properties->>'name',''), pl.p),
                        CASE
                            WHEN to_tsvector('english', coalesce(n.properties->>'name', ''))
                                 @@ plainto_tsquery('english', pl.p) THEN 0.5
                            ELSE 0.0
                        END
                    ) DESC,
                    (n.properties->>'last_seen_at')::timestamptz DESC NULLS LAST,
                    n.canonical_id
                ) AS rn
            FROM public.graph_nodes n
            CROSS JOIN probe_list pl
            WHERE n.customer_id = tenant
              AND n.label = ANY(labels)
              AND (
                  lower(n.properties->>'name') % pl.p
                  OR to_tsvector('english', coalesce(n.properties->>'name', ''))
                     @@ plainto_tsquery('english', pl.p)
              )
        ),
        capped AS (
            SELECT r.ord, r.label, r.canonical_id, r.kind, r.display_name,
                   r.last_seen_at_raw, r.rel,
                   row_number() OVER (
                       PARTITION BY r.ord
                       ORDER BY r.rel DESC, r.last_seen_at_raw DESC NULLS LAST,
                                r.label, r.canonical_id
                   ) AS rn2
            FROM ranked r
            WHERE r.rn <= per_type_cap
        )
        SELECT c.ord, c.label, c.canonical_id, c.kind, c.display_name,
               c.last_seen_at_raw, c.rel
        FROM capped c
        WHERE c.rn2 <= total_cap
        ORDER BY c.ord, c.rn2;
        RETURN;
    END IF;

    -- pg_trgm folds case with the libc ctype; so does lower() under a libc
    -- default collation, and then a full-text hit outside the trigram leg
    -- scores exactly 0.5 for a threshold <= 0.5 (see the shortcut below).
    folds_alike := (SELECT d.datlocprovider = 'c'
                    FROM pg_catalog.pg_database d
                    WHERE d.datname = current_database());

    FOREACH probe IN ARRAY probes LOOP
        i := i + 1;
        CONTINUE WHEN probe IS NULL;
        q := plainto_tsquery('english', probe);
        pn := array_length(show_trgm(probe), 1);

        -- Trigram leg: the nodes whose lower(name) is `%` the probe.
        IF pn IS NULL THEN
            -- No trigrams (punctuation only, ''): similarity is 0.
            trgm_ids := '{}';
        ELSIF pn < 16 THEN
            -- One word: the trigram-count window. OFFSET 0 keeps `%` out of
            -- index selection; the GIN admits too much for a short probe.
            SELECT coalesce(array_agg(w.node_id), '{}') INTO trgm_ids
            FROM (
                SELECT n.node_id, n.properties
                FROM public.graph_nodes n
                WHERE n.customer_id = tenant
                  AND n.label = ANY(labels)
                  AND array_length(show_trgm(lower(n.properties->>'name')), 1)
                      BETWEEN floor(pn * thr)::int AND ceil(pn / thr)::int
                OFFSET 0
            ) w
            WHERE lower(w.properties->>'name') % probe;
        ELSE
            -- Longer probes: the trigram GIN is selective; the planner chooses.
            SELECT coalesce(array_agg(n.node_id), '{}') INTO trgm_ids
            FROM public.graph_nodes n
            WHERE n.customer_id = tenant
              AND n.label = ANY(labels)
              AND lower(n.properties->>'name') % probe;
        END IF;

        RETURN QUERY
        WITH cand AS MATERIALIZED (
            SELECT c.label, c.canonical_id, c.properties,
                   CASE
                       -- Full-text hit, not a trigram match: similarity is
                       -- below the threshold, so GREATEST(.., 0.5) is 0.5.
                       WHEN c.fts_hit AND folds_alike AND thr <= 0.5
                            AND NOT (c.node_id = ANY(trgm_ids)) THEN 0.5::real
                       ELSE GREATEST(
                           similarity(coalesce(c.properties->>'name',''), probe),
                           CASE WHEN c.fts_hit THEN 0.5 ELSE 0.0 END
                       )
                   END AS rel,
                   (c.properties->>'last_seen_at')::timestamptz AS last_seen_at
            FROM (
                -- Full-text leg.
                SELECT n.node_id, n.label, n.canonical_id, n.properties, true AS fts_hit
                FROM public.graph_nodes n
                WHERE n.customer_id = tenant
                  AND n.label = ANY(labels)
                  AND to_tsvector('english', coalesce(n.properties->>'name', '')) @@ q
                UNION ALL
                -- Trigram matches that are not full-text hits.
                SELECT n.node_id, n.label, n.canonical_id, n.properties, false
                FROM public.graph_nodes n
                WHERE n.node_id = ANY(trgm_ids)
                  AND n.customer_id = tenant
                  AND NOT (to_tsvector('english', coalesce(n.properties->>'name', '')) @@ q)
            ) c
        ),
        per_label AS (
            -- The top per_type_cap of each label present: today's per-label
            -- row_number() <= per_type_cap, as a top-N sort.
            SELECT t.label, t.canonical_id, t.properties, t.rel
            FROM (SELECT DISTINCT c.label FROM cand c) l
            CROSS JOIN LATERAL (
                SELECT c.label, c.canonical_id, c.properties, c.rel
                FROM cand c
                WHERE c.label = l.label
                ORDER BY c.rel DESC, c.last_seen_at DESC NULLS LAST, c.canonical_id
                LIMIT per_type_cap
            ) t
        ),
        capped AS (
            SELECT p.label, p.canonical_id,
                   p.properties->>'kind' AS kind,
                   coalesce(p.properties->>'name', p.canonical_id) AS display_name,
                   p.properties->>'last_seen_at' AS last_seen_at_raw,
                   p.rel,
                   row_number() OVER (
                       ORDER BY p.rel DESC, p.properties->>'last_seen_at' DESC NULLS LAST,
                                p.label, p.canonical_id
                   ) AS rn2
            FROM per_label p
        )
        SELECT i, c.label, c.canonical_id, c.kind, c.display_name,
               c.last_seen_at_raw, c.rel
        FROM capped c
        WHERE c.rn2 <= total_cap
        ORDER BY c.rn2;
    END LOOP;
END
$$;

REVOKE EXECUTE ON FUNCTION kb_match_entities_multi_v1(text[], text[], integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION kb_match_entities_multi_v1(text[], text[], integer, integer) TO CURRENT_ROLE;
"""

FUNCTION_SIGNATURE = "public.kb_match_entities_multi_v1(text[], text[], integer, integer)"


def _concurrent_index_helpers():
    """0143's polled lock, budget and build-and-verify, loaded by path: the
    migrations directory is not a package, and a path relative to this file
    does not depend on the runner's working directory."""
    path = Path(__file__).with_name("20260929_0143_canonical_and_purge_indexes.py")
    spec = importlib.util.spec_from_file_location("prbe_kb_migration_0143", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(bind, budget_seconds: float | None = None) -> None:
    """Both indexes, then ANALYZE, on an autocommit connection (tests call this)."""
    m0143 = _concurrent_index_helpers()
    budget = m0143.BUDGET_SECONDS if budget_seconds is None else budget_seconds
    with m0143.migrator_session(bind, budget) as deadline:
        for name, create_sql, definition in INDEXES:
            m0143.build_index(bind, name, create_sql, definition, deadline)
        # Expression statistics of both indexes exist only after an ANALYZE;
        # the full-text leg's plan reads the lexeme frequencies. Same SHARE
        # UPDATE EXCLUSIVE lock autovacuum takes: reads and writes go on.
        m0143._budget(bind, deadline)
        bind.execute(sa.text("ANALYZE public.graph_nodes"))


def _function_body() -> str:
    """The plpgsql body inside MATCH_ENTITIES_SQL's dollar quotes (pg_proc.prosrc)."""
    start = MATCH_ENTITIES_SQL.index("AS $$") + len("AS $$")
    return MATCH_ENTITIES_SQL[start : MATCH_ENTITIES_SQL.index("$$;", start)]


def install_function(bind) -> None:
    """CREATE OR REPLACE the function, unless the hand-over already re-owned it.

    After the research-os Job hands it to `resolver`, `app` may not replace it
    -- so a downgrade (which leaves it, see below) followed by a re-upgrade
    would fail here and block the release. An existing copy that `app` does
    not own is accepted when its body is exactly this one; any other body
    means a change that must ship as _v2.
    """
    row = bind.execute(
        sa.text(
            "SELECT p.proowner = (SELECT oid FROM pg_roles WHERE rolname = current_user) AS mine,"
            " p.proowner::regrole::text AS owner, p.prosrc AS body"
            " FROM pg_proc p WHERE p.oid = to_regprocedure(:sig)"
        ),
        {"sig": FUNCTION_SIGNATURE},
    ).first()
    if row is None or row.mine:
        bind.execute(sa.text(MATCH_ENTITIES_SQL))
        return
    if row.body != _function_body():
        raise RuntimeError(
            f"{FUNCTION_SIGNATURE} is owned by {row.owner} with a different body; "
            "a changed lookup ships as _v2 (D26)"
        )


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    with op.get_context().autocommit_block():
        run(op.get_bind())
    install_function(op.get_bind())


def downgrade() -> None:
    # The function goes only while the migration role still owns it: after the
    # hand-over to `resolver`, `app` may not drop it, and an older engine never
    # calls it.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_proc
                WHERE oid = to_regprocedure('public.kb_match_entities_multi_v1(text[], text[], integer, integer)')
                  AND proowner = (SELECT oid FROM pg_roles WHERE rolname = current_user)
            ) THEN
                DROP FUNCTION public.kb_match_entities_multi_v1(text[], text[], integer, integer);
            ELSE
                RAISE NOTICE 'kb_match_entities_multi_v1 is not owned by %; left in place',
                    current_user;
            END IF;
        END $$;
        """
    )
    with op.get_context().autocommit_block():
        for name, _create_sql, _definition in reversed(INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{name}")

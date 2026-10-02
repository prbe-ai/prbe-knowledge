"""kb_match_document_titles_multi_v1(): grounding's document-title lookup, index-backed.

WHY. Grounding's title channel (engine/retrieval/grounding.py) is the biggest
grounding cost in production: the multi-probe statement averaged 9.3 s over
4,381 calls (pg_stat_statements, 30 days) and 2.4 s / 5.4 s for five probes on
`probe` / `new-workspace` when the plane was quiet (2026-10-02). It runs as
`app` under FORCE RLS, and neither `%` (similarity_op) nor `@@` (ts_match_vq) is
LEAKPROOF, so neither the trigram GIN nor the tsvector GIN may run ahead of the
tenant policy: every call reads every live titled document of the tenant and
computes similarity() for each one, once per probe.

WHAT. A SECURITY DEFINER function that returns exactly the rows of that
statement, in the same order, for the same inputs. Once it is owned by the
BYPASSRLS role `resolver` (decision D71: a research-os Job re-owns `kb_match_%`
functions in kb after migrations; `app` is never a member of `resolver`), the
indexes become usable. Per probe it runs:

  1. the full-text leg: live titled documents whose title_preview_tsv matches,
     top `per_probe_cap` by (similarity DESC, updated_at DESC NULLS LAST,
     doc_id). A full-text hit outranks every trigram-only row (the ranking
     leads with fts_hit DESC), so when this leg fills the cap the trigram leg
     cannot change the answer and is skipped. For the probes measured in
     production that is the common case (>= 10 live hits for 8 of 8 sample
     words on `probe`, 7 of 8 on `new-workspace`).
  2. otherwise the trigram-only leg, filling the remaining slots: `title %
     probe`, similarity >= floor, not a full-text hit -- with an exact prefilter
     on the title's trigram count, served by idx_documents_title_trgm_count:

       similarity(A, B) = |A n B| / |A u B| over the two trigram sets, and
       |A n B| <= min(|A|, |B|), |A u B| >= max(|A|, |B|). So similarity >= t
       implies min/max >= t, i.e.  t * |B| <= |A| <= |B| / t.

     A title whose trigram count lies outside that window can never reach the
     floor. The bounds are rounded OUTWARD (floor/ceil): similarity() is a
     float4 and the floor a real, and rounding inward drops rows sitting exactly
     on the threshold ('xy' vs 'xy abcdef' is exactly 0.3; ceil(10 * 0.3::real)
     is 4, not 3). Outward rounding only widens the window, so the prefilter
     never removes a row today's statement keeps; `%` and the floor still
     decide. The bounds are cast to int, or the index is not used.

     Short probes (< 16 trigrams, i.e. one word) read the count window. The
     trigram GIN is a poor filter for them -- it admits any title sharing 30%
     of a short probe's trigrams, and the planner estimates ~20 rows where
     there are 30,000+ -- so the window is read behind an OFFSET 0 fence that
     keeps `%` out of index selection (measured on the T12 rig: 'sesion' 63 ms
     through the window vs 267 ms through the GIN). Longer probes (multi-word,
     paths -- the single-probe path's input) leave the choice to the planner,
     which takes the GIN: 'engine/retrieval/grounding.py' 123 ms vs 377 ms
     through the window. The expression STATISTICS below are what give the
     planner the window's size; without them it guessed 838 rows for 37,417.

While the function is still owned by `app` (between this migration and the
re-own, or on a plane where the Job never runs), RLS applies inside it and
none of that is faster -- the per-probe legs would each re-read the tenant.
So it checks row_security_active() and, under RLS, runs the current statement
unchanged. Same rows either way; the engine falls back to its own copy of that
statement if the function is missing or errors.

SECURITY. Tenant ONLY from `app.current_customer_id`, read once into a
variable (NULL or '' -> no rows), and every table read is filtered on it --
under `resolver` nothing else isolates tenants. search_path pinned. EXECUTE
revoked from PUBLIC and granted to the creating role. Returns what the caller
maps and nothing more. NOTE FOR THE RE-OWN: ALTER FUNCTION ... OWNER rewrites
the creator's own ACL entry to the new owner (verified: {app=X/app} becomes
{resolver=X/resolver}), so the Job must `GRANT EXECUTE ON FUNCTION
kb_match_document_titles_multi_v1(text[], real, integer) TO app` after it, and
`GRANT USAGE ON SCHEMA public, SELECT ON documents TO resolver`. Without the
EXECUTE grant every call is refused and the engine falls back (correct, slow,
one warning per call).

`plan_cache_mode = force_custom_plan`: each statement is planned with this
call's tenant and probe as constants, which the window estimate and the
full-text GIN need. Both SET clauses name core settings; a custom one (the
tenant GUC) would fail as `app` on PG15+ (0144's first deploy).

The trigram operator reads pg_trgm.similarity_threshold from the caller, as
today's statement does (the engine sets it from the same constant as the
floor). The prefilter is exact whatever it is set to: a trigram-only row in the
output has similarity >= the floor, so it lies inside the floor's window.

Named _v1 because once `resolver` owns it, `app` can neither replace nor drop
it: a change ships as _v2 beside it.

INDEX. CONCURRENTLY, through 0143's machinery (polled migrator lock, one
wall-clock budget, definition check, invalid-leftover rebuild). Then the
expression statistics and an ANALYZE so they exist before the first call.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision = "0146_kb_match_titles"
down_revision = "0144_kb_provision_tenant"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_documents_title_trgm_count"
INDEX_SQL = (
    f"CREATE INDEX CONCURRENTLY {INDEX_NAME} ON public.documents "
    "(customer_id, (array_length(show_trgm(title), 1))) WHERE valid_to IS NULL"
)
#: pg_get_indexdef() of the index above; anything else of that name is rebuilt.
INDEX_DEFINITION = (
    f"CREATE INDEX {INDEX_NAME} ON public.documents USING btree "
    "(customer_id, array_length(show_trgm(title), 1)) WHERE (valid_to IS NULL)"
)
STATISTICS_SQL = (
    "CREATE STATISTICS IF NOT EXISTS public.documents_title_trgm_count_stx "
    "ON (array_length(show_trgm(title), 1)) FROM public.documents"
)

# db/schema.sql carries this body byte for byte (tests pin it).
MATCH_TITLES_SQL = r"""
CREATE OR REPLACE FUNCTION kb_match_document_titles_multi_v1(
    probes text[], sim_floor real, per_probe_cap integer
)
RETURNS TABLE (ord bigint, doc_id text, source_system text, title text, updated_at timestamptz)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
SET plan_cache_mode = force_custom_plan
AS $$
DECLARE
    tenant text := current_setting('app.current_customer_id', true);
    probe text;
    i bigint := 0;
    q tsquery;
    pn integer;
    n bigint;
BEGIN
    IF tenant IS NULL OR tenant = '' OR probes IS NULL
       OR per_probe_cap IS NULL OR per_probe_cap <= 0 THEN
        RETURN;
    END IF;
    -- >= 0.01: the count window's upper bound is ceil(pn / sim_floor)::int,
    -- which overflows int for a vanishing floor. Grounding uses 0.3.
    IF sim_floor IS NULL OR NOT (sim_floor >= 0.01 AND sim_floor <= 1) THEN
        RAISE EXCEPTION 'kb_match_document_titles_multi_v1: sim_floor must be in [0.01, 1], got %',
            sim_floor USING ERRCODE = 'invalid_parameter_value';
    END IF;

    -- Owned by a role RLS applies to (before the hand-over to `resolver`):
    -- no index can help, and per-probe legs would each re-read the tenant.
    -- Run grounding.py's statement as it is.
    IF row_security_active('public.documents'::regclass) THEN
        RETURN QUERY
        WITH probe_list AS (
            SELECT u.p, u.ord FROM unnest(probes) WITH ORDINALITY AS u(p, ord)
        ),
        ranked AS (
            SELECT pl.ord, d.doc_id, d.source_system, d.title, d.updated_at,
                   similarity(d.title, pl.p) AS trgm_sim,
                   CASE WHEN d.title_preview_tsv @@ plainto_tsquery('english', pl.p)
                        THEN 1 ELSE 0 END AS fts_hit
            FROM public.documents d
            CROSS JOIN probe_list pl
            WHERE d.customer_id = tenant
              AND d.valid_to IS NULL
              AND d.title IS NOT NULL
              AND d.title <> ''
              AND (d.title % pl.p
                   OR d.title_preview_tsv @@ plainto_tsquery('english', pl.p))
        ),
        capped AS (
            SELECT r.ord, r.doc_id, r.source_system, r.title, r.updated_at,
                   row_number() OVER (
                       PARTITION BY r.ord
                       ORDER BY r.fts_hit DESC, r.trgm_sim DESC,
                                r.updated_at DESC NULLS LAST, r.doc_id
                   ) AS rn
            FROM ranked r
            WHERE r.trgm_sim >= sim_floor OR r.fts_hit = 1
        )
        SELECT c.ord, c.doc_id, c.source_system, c.title, c.updated_at
        FROM capped c
        WHERE c.rn <= per_probe_cap
        ORDER BY c.ord, c.rn;
        RETURN;
    END IF;

    FOREACH probe IN ARRAY probes LOOP
        i := i + 1;
        q := plainto_tsquery('english', probe);

        -- Full-text hits rank above every trigram-only row.
        RETURN QUERY
        SELECT i, d.doc_id, d.source_system, d.title, d.updated_at
        FROM public.documents d
        WHERE d.customer_id = tenant
          AND d.valid_to IS NULL
          AND d.title IS NOT NULL
          AND d.title <> ''
          AND d.title_preview_tsv @@ q
        ORDER BY similarity(d.title, probe) DESC, d.updated_at DESC NULLS LAST, d.doc_id
        LIMIT per_probe_cap;
        GET DIAGNOSTICS n = ROW_COUNT;
        CONTINUE WHEN n >= per_probe_cap;

        -- No trigrams (punctuation only): similarity is 0, below any floor.
        pn := array_length(show_trgm(probe), 1);
        CONTINUE WHEN pn IS NULL;

        IF pn < 16 THEN
            -- One word: read the trigram-count window. OFFSET 0 keeps `%` out
            -- of index selection; the GIN admits too much for a short probe.
            RETURN QUERY
            SELECT i, w.doc_id, w.source_system, w.title, w.updated_at
            FROM (
                SELECT d.doc_id, d.source_system, d.title, d.updated_at, d.title_preview_tsv
                FROM public.documents d
                WHERE d.customer_id = tenant
                  AND d.valid_to IS NULL
                  AND d.title IS NOT NULL
                  AND d.title <> ''
                  AND array_length(show_trgm(d.title), 1)
                      BETWEEN floor(pn * sim_floor)::int AND ceil(pn / sim_floor)::int
                OFFSET 0
            ) w
            WHERE w.title % probe
              AND similarity(w.title, probe) >= sim_floor
              AND NOT (w.title_preview_tsv @@ q)
            ORDER BY similarity(w.title, probe) DESC, w.updated_at DESC NULLS LAST, w.doc_id
            LIMIT per_probe_cap - n;
        ELSE
            -- Longer probes: the trigram GIN is selective; the planner chooses.
            RETURN QUERY
            SELECT i, d.doc_id, d.source_system, d.title, d.updated_at
            FROM public.documents d
            WHERE d.customer_id = tenant
              AND d.valid_to IS NULL
              AND d.title IS NOT NULL
              AND d.title <> ''
              AND array_length(show_trgm(d.title), 1)
                  BETWEEN floor(pn * sim_floor)::int AND ceil(pn / sim_floor)::int
              AND d.title % probe
              AND similarity(d.title, probe) >= sim_floor
              AND NOT (d.title_preview_tsv @@ q)
            ORDER BY similarity(d.title, probe) DESC, d.updated_at DESC NULLS LAST, d.doc_id
            LIMIT per_probe_cap - n;
        END IF;
    END LOOP;
END
$$;

REVOKE EXECUTE ON FUNCTION kb_match_document_titles_multi_v1(text[], real, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION kb_match_document_titles_multi_v1(text[], real, integer) TO CURRENT_ROLE;
"""


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
    """Index, statistics, ANALYZE on an autocommit connection (tests call this)."""
    m0143 = _concurrent_index_helpers()
    budget = m0143.BUDGET_SECONDS if budget_seconds is None else budget_seconds
    with m0143.migrator_session(bind, budget) as deadline:
        m0143.build_index(bind, INDEX_NAME, INDEX_SQL, INDEX_DEFINITION, deadline)
        m0143._budget(bind, deadline)
        bind.execute(sa.text(STATISTICS_SQL))
        # The statistics object is empty until the table is analyzed, and the
        # long-probe leg's plan depends on it. ANALYZE takes the same SHARE
        # UPDATE EXCLUSIVE lock autovacuum does: reads and writes go on.
        m0143._budget(bind, deadline)
        bind.execute(sa.text("ANALYZE public.documents"))


FUNCTION_SIGNATURE = "public.kb_match_document_titles_multi_v1(text[], real, integer)"


def _function_body() -> str:
    """The plpgsql body inside MATCH_TITLES_SQL's dollar quotes (pg_proc.prosrc)."""
    start = MATCH_TITLES_SQL.index("AS $$") + len("AS $$")
    return MATCH_TITLES_SQL[start : MATCH_TITLES_SQL.index("$$;", start)]


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
        bind.execute(sa.text(MATCH_TITLES_SQL))
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
                WHERE oid = to_regprocedure('public.kb_match_document_titles_multi_v1(text[], real, integer)')
                  AND proowner = (SELECT oid FROM pg_roles WHERE rolname = current_user)
            ) THEN
                DROP FUNCTION public.kb_match_document_titles_multi_v1(text[], real, integer);
            ELSE
                RAISE NOTICE 'kb_match_document_titles_multi_v1 is not owned by %; left in place',
                    current_user;
            END IF;
        END $$;
        """
    )
    with op.get_context().autocommit_block():
        op.execute("DROP STATISTICS IF EXISTS public.documents_title_trgm_count_stx")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{INDEX_NAME}")

"""Which retrieval predicates depend on which index, declared so drift fails loudly.

WHY THIS EXISTS
---------------
Four separate times, a retrieval query was written so that an index which
existed, was built, and was paid for could not serve it. Nothing failed. The
results stayed correct. It just got slow, and stayed slow until somebody
happened to run EXPLAIN:

  1. bm25.py       `(chunks.content_tsv @@ q) OR (documents.title_tsv @@ q ...)`
                   -- an OR across two JOINed tables, servable by no single
                   index. Parallel Seq Scan over 4.4 GB, 30s statement timeout.
  2. grounding.py  `coalesce(d.title,'') % $2` against an index on `title`.
                   Expression mismatch. 742 ms -> 77 ms once aligned.
  3. vector.py     `ORDER BY embedding <=> $2, chunk_id`. The tiebreaker means
                   an ANN index cannot answer it. 3,355 ms -> 12.7 ms.
  4. grounding.py  `coalesce(properties->>'name','') % $2` against an index on
                   `lower(properties->>'name')`. 115 ms -> 1.6 ms.

Every one has the same shape: the query's expression and the index's expression
drifted apart, and an expression index only ever serves the EXACT expression it
was built on. The failure is silent by construction -- Postgres has no reason to
complain, because a sequential scan is a perfectly valid way to answer.

WHY NOT AN EXPLAIN TEST
-----------------------
The obvious guard is "run EXPLAIN, assert no Seq Scan". It does not work here,
and the reason is worth writing down so nobody rebuilds it:

  * On a seeded test table of a few dozen rows a sequential scan is the CORRECT
    plan. The assertion only passes under `enable_seqscan = off`, which tests
    the override rather than the SQL.
  * A seq scan is legitimate in production too, at high selectivity. A blanket
    prohibition encodes a rule that is false in general.
  * Plans move with statistics, table size and Postgres version, so the test
    would fail for reasons unrelated to the bug it is meant to catch.

So this checks the thing that is actually invariant: the TEXT of the expression
the query uses has to appear in the definition of the index it claims to use.
That is deterministic, needs no database, no statistics and no fixtures, and is
exactly the property that broke all four times.

WHAT A FAILURE MEANS
--------------------
Either the query drifted (someone wrapped a column in coalesce(), added a sort
key, moved a predicate across a join) or the index did (renamed, re-expressed,
dropped from schema.sql). Both are the bug. Fix whichever moved -- do not
"fix" the contract to match, unless the index genuinely changed and you have
re-measured the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class IndexContract:
    """A promise that one predicate is served by one index.

    `expression` is matched as a SUBSTRING of the index's definition in
    db/schema.sql, whitespace-normalised. Substring rather than equality
    because an index definition carries the access method, opclass and any
    partial predicate around the expression we care about.

    `predicate` is matched as a substring of `source_file`. It is the literal
    SQL fragment in the query. If someone edits the query, this stops matching
    and the guard fires -- which is the entire point.
    """

    index: str
    table: str
    expression: str
    source_file: str
    predicate: str
    why: str


# The predicates that carry retrieval latency. Adding a hot predicate without
# adding it here is how the fifth instance ships.
INDEX_CONTRACTS: Final[tuple[IndexContract, ...]] = (
    IndexContract(
        index="idx_documents_title_trgm",
        table="documents",
        expression="title gin_trgm_ops",
        source_file="engine/retrieval/grounding.py",
        predicate="d.title % $2",
        why=(
            "Grounding's doc-title channel is ~95% of the grounding stage. This "
            "predicate was `coalesce(d.title,'') % $2`, which the index on "
            "`title` cannot serve, and an OR is only as indexable as its worst "
            "branch -- so the whole predicate seq-scanned. 742 ms -> 77 ms."
        ),
    ),
    IndexContract(
        index="idx_graph_nodes_name_trgm",
        table="graph_nodes",
        expression="lower((properties ->> 'name'::text)) gin_trgm_ops",
        source_file="engine/retrieval/grounding.py",
        predicate="lower(properties->>'name') % $2",
        why=(
            "Same bug, second location. Was `coalesce(properties->>'name','')`, "
            "against an index on `lower(...)`. Seq scan, 20,173 rows filtered. "
            "115 ms -> 1.6 ms. pg_trgm case-folds internally, so lower() is an "
            "index fix and not a semantic change (verified: symmetric "
            "difference 0 over 46 rows)."
        ),
    ),
    IndexContract(
        index="idx_chunks_bm25_v2",
        table="chunks",
        # The COLUMN-LIST neighbourhood, not a bare "title". A bare token
        # passes even when title is removed from the indexed columns, because
        # the `text_fields` tokenizer config on the same statement still
        # mentions it by name -- mutation-tested, and it escaped.
        expression="content, title, customer_id",
        source_file="engine/retrieval/retrievers/bm25.py",
        predicate="paradedb.match('title', $2)",
        why=(
            "BM25 matched titles through a cross-table OR against "
            "`documents.title_tsv`, which no single index can serve -- 30s "
            "statement timeout. The title now lives ON the chunk so the query "
            "is single-table and the cross-table OR is unwriteable. 230 ms."
        ),
    ),
    IndexContract(
        # The ANN ORDER BY shape contract, carried by the LIVE index since
        # 0126 dropped the full one (audited consumer-free; see that
        # migration). The guard is unchanged in substance: the sort key the
        # query emits must stay a bare distance an HNSW index can serve.
        index="idx_chunks_embedding_v2_hnsw_live",
        table="chunks",
        expression="embedding_v2 halfvec_cosine_ops",
        source_file="engine/retrieval/retrievers/vector.py",
        predicate='ann_order_sql = "c.embedding_v2 <=> $2::halfvec"',
        why=(
            "An ANN index can only answer `ORDER BY <distance>` and nothing "
            "else. This read `<=> $2::halfvec, c.chunk_id`; the tiebreaker "
            "forced exact distances for every row and the planner fell back to "
            "a Parallel Seq Scan + Sort. 3,355 ms -> 12.7 ms. Determinism moved "
            "to an outer sort over the bounded pool -- see the callsite. "
            "Documented pgvector behaviour (pgvector#760)."
        ),
    ),
    IndexContract(
        index="idx_chunks_embedding_v2_hnsw_live",
        table="chunks",
        expression="halfvec_cosine_ops) WHERE valid_to IS NULL",
        source_file="engine/retrieval/temporal.py",
        predicate="chunk_sql=f\"AND {chunk_alias}.valid_to IS NULL\"",
        why=(
            "The live-only twin (0124) is chosen by the planner ONLY while "
            "TemporalMode.LATEST's chunk predicate exactly implies the index's "
            "WHERE clause. Only ~35% of chunks are live, so losing the "
            "implication silently returns every default ANN query to walking "
            "3x the graph on the full index -- with no error, no test failure "
            "and no guardian signal, since hnsw is outside the pg_search "
            "scans. If LATEST ever grows an OR-form (the AS_OF branch already "
            "has one), this contract is what fires."
        ),
    ),
)

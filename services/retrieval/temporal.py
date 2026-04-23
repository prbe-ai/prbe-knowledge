"""SQL predicate fragments for TemporalSpec.

Each retriever composes its query from the same helpers so behavior stays
consistent across vector / BM25 / graph. Fragments return (`sql`, `params`)
pairs where `sql` references $N placeholders and `params` is the tail that
the caller appends to its own parameter list.

The `doc_alias` / `chunk_alias` args let us emit the right correlated names
for whichever query shape a retriever happens to use.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.models import TemporalMode, TemporalSpec


@dataclass(slots=True)
class TemporalPredicate:
    doc_sql: str  # predicate fragment to AND into the documents join
    chunk_sql: str  # predicate fragment to AND into the chunks clause
    params: list  # trailing parameters referenced by $N placeholders


def build_predicate(
    spec: TemporalSpec,
    doc_alias: str,
    chunk_alias: str,
    next_param_index: int,
) -> TemporalPredicate:
    """Translate a TemporalSpec into doc + chunk SQL fragments.

    $-numbering starts at `next_param_index`. Return value's `params` must be
    appended to the caller's params list in the same order. Empty strings
    when no filter applies.
    """
    if spec.mode == TemporalMode.ALL:
        return TemporalPredicate(doc_sql="", chunk_sql="", params=[])

    if spec.mode == TemporalMode.LATEST:
        return TemporalPredicate(
            doc_sql=f"AND {doc_alias}.valid_to IS NULL",
            chunk_sql=f"AND {chunk_alias}.valid_to IS NULL",
            params=[],
        )

    if spec.mode == TemporalMode.AS_OF:
        i = next_param_index
        return TemporalPredicate(
            doc_sql=(
                f"AND {doc_alias}.valid_from <= ${i} "
                f"AND ({doc_alias}.valid_to IS NULL OR {doc_alias}.valid_to > ${i})"
            ),
            chunk_sql=(
                f"AND {chunk_alias}.valid_from <= ${i} "
                f"AND ({chunk_alias}.valid_to IS NULL OR {chunk_alias}.valid_to > ${i})"
            ),
            params=[spec.as_of],
        )

    if spec.mode == TemporalMode.CHANGED_BETWEEN:
        # Basis: source-time uses documents.updated_at (how Linear/Slack/GitHub
        # saw the edit happen). Ingest-time uses documents.ingested_at. Default
        # is source — what the agent usually wants.
        basis_col = "updated_at" if spec.time_basis == "source" else "ingested_at"
        i = next_param_index
        j = next_param_index + 1
        # Chunks are scoped by the doc-level filter (live chunks of the doc
        # version that changed). If we additionally wanted chunks whose own
        # valid_from lands in the window, we'd OR it in — but that produces
        # noisier results. Keep it doc-scoped.
        return TemporalPredicate(
            doc_sql=(
                f"AND {doc_alias}.{basis_col} >= ${i} "
                f"AND {doc_alias}.{basis_col} < ${j} "
                f"AND {doc_alias}.valid_to IS NULL"
            ),
            chunk_sql=f"AND {chunk_alias}.valid_to IS NULL",
            params=[spec.since, spec.until],
        )

    raise ValueError(f"unhandled TemporalMode: {spec.mode}")

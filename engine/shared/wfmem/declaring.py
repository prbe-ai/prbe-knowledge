"""Writing a declared rule down, and refusing to resolve a collision quietly.

TWO FUNCTIONS, AND THE SPLIT BETWEEN THEM IS THE DESIGN.

`find_neighbours` reads. It answers "what does this tenant already have that
looks like the rule you just typed", and it decides NOTHING. `declare` writes,
and it executes a relation it was TOLD -- merge, variant, conflict, or nothing.
Between the two sits a human looking at the echo.

That is what "never silent resolution" has to mean in practice. The tempting
version puts a similarity threshold in the write path and merges above it, which
is silent resolution with an extra step: the author sees a success message and
never learns that their rule was folded into somebody else's, or that the team
now holds two rules that contradict each other. Both of those are only visible
at the moment of declaring, to the one person who can say which it is.

WHAT v0 DELIBERATELY DOES NOT DO: detect contradiction. The design calls for a
`conflicts_with` edge when a new rule contradicts an existing one, and this
module writes that edge -- but only when a human says so. Nothing here decides
that two rules contradict, because a similarity score cannot tell "always squash
before merging" from "never squash before merging" (those two are NEAR-IDENTICAL
by embedding, which is the whole problem) and an LLM asked to judge it will
answer confidently either way. Surfacing the neighbour and letting the author say
"that one contradicts this" is both cheaper and more honest than a detector that
is wrong in the direction of writing an edge nobody meant. If the falsification
pilot shows authors routinely miss the contradiction sitting in front of them,
THEN a detector earns its place -- as a proposer, still not as a decider.

CONFLICT EDGES ARE RECIPROCAL, and this is easy to get wrong. If A records
`conflicts_with: [B]` but B records nothing, then serving B shows a clean rule
with no hint that a competing one exists -- and B is exactly as likely to be the
one retrieved. A one-directional conflict edge is worse than none, because it
reads as a guarantee that was checked.

THE SECRET SCAN'S FIRST CALL SITE. `assert_clean` / `assert_clean_json` have
existed since Stage 0 with nothing calling them, which made §4's "secret-scan at
ingest" a paper guarantee. This is the ingest. It runs BEFORE the transaction
opens, so a refused rule leaves nothing behind, and it covers every field that
carries author-supplied text: `body`, `semantic_action`, `binding`, `scope` and
`source_ref`. Not just `body` and `binding` as the spec's wording names -- a
credential pasted into a rule does not know which column it is heading for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from engine.shared.db import with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.logging import get_logger
from engine.shared.wfmem.secret_scan import assert_clean, assert_clean_json
from engine.shared.wfmem.structuring import ClauseDraft

log = get_logger(__name__)

#: Status every clause written through this path carries. The ladder in the
#: CHECK constraint has eleven values and Phase 0 writes exactly one of them:
#: a human said it, on purpose, and nothing has corroborated it yet.
DECLARED_STATUS = "declared"

#: Source class for the evidence row a declaration produces.
DECLARED_SOURCE_CLASS = "declared"

#: Cosine at or above which two rule bodies are "the same rule" for the purpose
#: of showing the author a neighbour. NOT a merge threshold -- nothing merges
#: without a person saying so -- which is why it can be generous. The cost of a
#: false neighbour is one line in a confirmation echo; the cost of a missed one
#: is a duplicate rule nobody notices for a month.
NEIGHBOUR_FLOOR = 0.72

#: How many neighbours to surface. A confirmation echo a person actually reads
#: has room for a couple, and past that the list stops being read at all.
MAX_NEIGHBOURS = 3


class Relation(StrEnum):
    """What the author said this rule is, relative to an existing one."""

    #: Unrelated to anything, or related and the author does not care. Plain write.
    NEW = "new"
    #: The same rule. NO new clause is created -- the declarer is added as
    #: evidence on the existing one, which is what makes a second human's
    #: agreement visible to the visibility guard rather than forking the rule.
    MERGE = "merge"
    #: Same intent, different binding. New clause, `lineage.variant_of`.
    VARIANT = "variant"
    #: Contradicts an existing rule. New clause, reciprocal `conflicts_with`.
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Neighbour:
    """An existing clause that resembles the one being declared."""

    clause_id: UUID
    body: str
    status: str
    author_ref: str
    similarity: float
    #: Whether it operates on the same thing. Same body + same binding is a
    #: duplicate; same body + different binding is the variant case, and the
    #: author is the only one who can tell those apart from the outside.
    same_binding: bool


@dataclass(frozen=True)
class Declaration:
    """What the write actually did."""

    clause_id: UUID
    #: False when `Relation.MERGE` folded the declaration into an existing
    #: clause. A caller reporting "rule created" on a merge would be lying about
    #: the one outcome the author most needs to know.
    created: bool
    evidence_id: UUID
    situation_id: UUID | None
    #: Who published it on their own authority, or None if it is visible only
    #: under the ordinary two-human rule. Returned so a surface can tell the
    #: author which of the two just happened -- "your rule is live for the team"
    #: and "your rule is saved and private until somebody agrees" are different
    #: sentences, and guessing wrong in either direction is a bad outcome.
    shared_by: str | None = None


class DeclarationRefused(ValueError):
    """The declaration cannot be written as asked. Carries a human-readable why."""


async def find_neighbours(
    customer_id: str,
    draft: ClauseDraft,
    *,
    embedder: Any | None = None,
    limit: int = MAX_NEIGHBOURS,
) -> list[Neighbour]:
    """Existing clauses that resemble `draft`, most similar first.

    Reads EVERY clause in the tenant, not just those in the draft's situation. A
    duplicate filed under a different situation is still a duplicate, and it is
    the one a situation-scoped search would never show -- which is precisely how
    the same rule ends up in the store three times with three different labels.

    Deliberately ignores the visibility guard. A neighbour list is not a serving
    surface: it exists so the author does not unknowingly duplicate a colleague's
    single-author rule, and hiding it would guarantee the duplicate. Only the
    body, status and author are exposed, which the author would see the moment
    the guard unlocked anyway -- and the alternative is a system that lets you
    create a conflict it can see and you cannot.
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT id, body, status, author_ref, binding
              FROM clauses
             WHERE customer_id = $1
             ORDER BY updated_at DESC
             LIMIT 500
            """,
            customer_id,
        )
    if not rows:
        return []

    embedder = embedder or get_embedder_v2()
    bodies = [row["body"] for row in rows]
    vectors = await _embed_all(embedder, [draft.body, *bodies])
    if vectors is None:
        # Degrade to "no neighbours" rather than raising. A declaration that
        # cannot be enriched with neighbours is still a declaration worth
        # accepting; refusing the write because the embedder is down would take
        # the whole declared path out over a nice-to-have.
        log.warning("wfmem_declaring.neighbour_embedding_failed", customer=customer_id)
        return []

    query_vector, body_vectors = vectors[0], vectors[1:]
    scored: list[Neighbour] = []
    for row, vector in zip(rows, body_vectors, strict=True):
        similarity = _cosine(query_vector, vector)
        if similarity < NEIGHBOUR_FLOOR:
            continue
        scored.append(
            Neighbour(
                clause_id=row["id"],
                body=row["body"],
                status=row["status"],
                author_ref=row["author_ref"],
                similarity=similarity,
                same_binding=_as_dict(row["binding"]) == draft.binding,
            )
        )
    scored.sort(key=lambda n: (-n.similarity, str(n.clause_id)))
    return scored[: max(0, limit)]


async def declare(
    customer_id: str,
    draft: ClauseDraft,
    *,
    actor_ref: str,
    source_ref: dict[str, Any],
    situation_id: UUID | None = None,
    classification: dict[str, Any] | None = None,
    relation: Relation = Relation.NEW,
    related_clause_id: UUID | None = None,
    publish: bool = False,
) -> Declaration:
    """Write the rule. One transaction: clause, situation edge, evidence.

    `relation` is the author's decision, not a guess. `MERGE` needs
    `related_clause_id` and writes no new clause; `VARIANT` and `CONFLICT` need
    one and record lineage in both directions where the relation is symmetric.

    `publish=True` makes the clause visible to the team IMMEDIATELY, on this
    author's own authority, instead of waiting for a second human's independent
    agreement. It is the difference between "saved, and private until somebody
    else says the same thing" and "this is now team policy". Without it the
    important case is impossible: a lead declaring twenty existing team rules
    produces twenty invisible clauses, silently, because a lead is one person.

    It is NOT the default, and should not become one. The two-human rule is what
    stops a private habit from becoming false policy, and most declarations
    genuinely are one person's note. Publishing is a deliberate act, it is
    attributed to `actor_ref` in `shared_by`, and serving surfaces label it --
    a reader can always tell "one person says so" from "the team demonstrably
    does this".

    Raises `SecretDetected` before opening the transaction, and
    `DeclarationRefused` for a relation whose arguments do not make sense.
    """
    if not actor_ref:
        raise DeclarationRefused("a declaration must record who made it")

    _scan(draft, source_ref)

    if relation is Relation.MERGE:
        if related_clause_id is None:
            raise DeclarationRefused("a merge must name the clause it merges into")
        return await _merge(
            customer_id,
            into=related_clause_id,
            actor_ref=actor_ref,
            source_ref=source_ref,
            situation_id=situation_id,
            classification=classification,
            publish=publish,
        )

    if relation in (Relation.VARIANT, Relation.CONFLICT) and related_clause_id is None:
        raise DeclarationRefused(f"a {relation.value} must name the clause it relates to")

    return await _insert(
        customer_id,
        draft,
        actor_ref=actor_ref,
        source_ref=source_ref,
        situation_id=situation_id,
        classification=classification,
        relation=relation,
        related_clause_id=related_clause_id,
        publish=publish,
    )


async def publish_clause(customer_id: str, clause_id: UUID, *, actor_ref: str) -> str | None:
    """Publish an EXISTING clause on `actor_ref`'s authority. Returns the publisher.

    The separate verb matters. Declaring and publishing are different decisions
    made at different times: somebody writes a note for themselves, and weeks
    later decides the team should follow it. Forcing that through `declare`
    would mean re-typing the rule, which produces a second clause instead of
    publishing the first.

    IDEMPOTENT, AND FIRST PUBLISHER WINS. A second call leaves `shared_by` and
    `shared_at` alone rather than overwriting them. The column answers "who put
    this in front of the team", and that is the FIRST person to do so -- letting
    a later caller overwrite it would quietly reassign responsibility for a
    decision they did not make.

    Returns the publisher's ref (which may be somebody else, on a repeat call),
    or None if no such clause exists in this tenant.
    """
    if not actor_ref:
        raise DeclarationRefused("publishing a rule must record who did it")

    async with with_tenant(customer_id) as conn:
        return await conn.fetchval(
            """
            UPDATE clauses
               SET shared_by = COALESCE(shared_by, $3),
                   shared_at = COALESCE(shared_at, now())
             WHERE customer_id = $1 AND id = $2
            RETURNING shared_by
            """,
            customer_id,
            clause_id,
            actor_ref,
        )


async def unpublish_clause(customer_id: str, clause_id: UUID) -> bool:
    """Withdraw a publication. Returns True if a row changed.

    The undo half, and it exists because publication is unilateral. Somebody
    publishes a rule the team turns out not to agree with, and the only
    alternatives without this are deleting the clause -- destroying its evidence
    and history -- or leaving false policy in front of everybody. Withdrawing
    returns the clause to the ordinary two-human rule; it stays visible if it
    genuinely earned corroboration in the meantime, which is the correct
    outcome and not something a caller has to work out.
    """
    async with with_tenant(customer_id) as conn:
        status = await conn.execute(
            """
            UPDATE clauses
               SET shared_by = NULL, shared_at = NULL
             WHERE customer_id = $1 AND id = $2 AND shared_by IS NOT NULL
            """,
            customer_id,
            clause_id,
        )
    return status.rsplit(" ", 1)[-1] != "0"


def _scan(draft: ClauseDraft, source_ref: dict[str, Any]) -> None:
    """Every author-supplied field, before anything is written.

    Ordered cheapest-first only incidentally; what matters is that ALL of it runs
    before the transaction opens, so a refusal leaves no half-written rule and no
    row for somebody to find later and wonder about.
    """
    assert_clean(draft.body)
    if draft.semantic_action:
        assert_clean(draft.semantic_action)
    assert_clean_json(draft.binding)
    assert_clean_json(draft.scope)
    assert_clean_json(source_ref)


async def _insert(
    customer_id: str,
    draft: ClauseDraft,
    *,
    actor_ref: str,
    source_ref: dict[str, Any],
    situation_id: UUID | None,
    classification: dict[str, Any] | None,
    relation: Relation,
    related_clause_id: UUID | None,
    publish: bool,
) -> Declaration:
    lineage = _lineage_for(relation, related_clause_id)
    shared_by = actor_ref if publish else None

    async with with_tenant(customer_id) as conn:
        if related_clause_id is not None and not await _clause_exists(
            conn, customer_id, related_clause_id
        ):
            # Checked inside the tenant scope so a caller cannot probe for the
            # existence of another tenant's clause id by watching which error
            # comes back -- the same oracle the composite foreign keys close at
            # the schema level.
            raise DeclarationRefused("the related clause does not exist in this workspace")

        clause_id = await conn.fetchval(
            """
            INSERT INTO clauses
                (customer_id, kind, body, semantic_action, binding, scope,
                 status, author_ref, lineage, shared_by, shared_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9::jsonb,
                    $10,
                    -- Timestamp derived from the publisher rather than passed
                    -- separately, so the CHECK that keeps the pair together
                    -- cannot be violated by a caller that sets one of them.
                    CASE WHEN $10::text IS NULL THEN NULL ELSE now() END)
            RETURNING id
            """,
            customer_id,
            draft.kind,
            draft.body,
            draft.semantic_action,
            json.dumps(draft.binding),
            json.dumps(draft.scope),
            DECLARED_STATUS,
            actor_ref,
            json.dumps(lineage),
            shared_by,
        )

        if relation is Relation.CONFLICT and related_clause_id is not None:
            await _add_reciprocal_conflict(conn, customer_id, related_clause_id, clause_id)

        if situation_id is not None:
            await _attach_situation(conn, customer_id, clause_id, situation_id, classification)

        evidence_id = await _add_evidence(
            conn, customer_id, clause_id, actor_ref=actor_ref, source_ref=source_ref
        )

    return Declaration(
        clause_id=clause_id,
        created=True,
        evidence_id=evidence_id,
        situation_id=situation_id,
        shared_by=shared_by,
    )


async def _merge(
    customer_id: str,
    *,
    into: UUID,
    actor_ref: str,
    source_ref: dict[str, Any],
    situation_id: UUID | None,
    classification: dict[str, Any] | None,
    publish: bool = False,
) -> Declaration:
    """Add the declarer as evidence on an existing clause. No new clause.

    THIS IS THE CASE THAT DOES REAL WORK for the visibility guard. A rule one
    person wrote is invisible to everybody else until a second distinct human's
    evidence appears; a colleague declaring the same rule and choosing "merge" IS
    that second human, and folding their declaration in here is what unlocks the
    clause for the team. Writing a near-identical second clause instead would
    leave both invisible forever, each with one author.
    """
    async with with_tenant(customer_id) as conn:
        if not await _clause_exists(conn, customer_id, into):
            raise DeclarationRefused("the clause to merge into does not exist in this workspace")

        if situation_id is not None:
            # A merge may still teach us a situation the original was never
            # attached to; ON CONFLICT DO NOTHING keeps it idempotent.
            await _attach_situation(conn, customer_id, into, situation_id, classification)

        evidence_id = await _add_evidence(
            conn, customer_id, into, actor_ref=actor_ref, source_ref=source_ref
        )

        shared_by: str | None = None
        if publish:
            # COALESCE, so merging-with-publish onto an already-published clause
            # does not reassign authorship of that decision to whoever merged
            # second. Same first-publisher-wins rule as `publish_clause`.
            shared_by = await conn.fetchval(
                """
                UPDATE clauses
                   SET shared_by = COALESCE(shared_by, $3),
                       shared_at = COALESCE(shared_at, now())
                 WHERE customer_id = $1 AND id = $2
                RETURNING shared_by
                """,
                customer_id,
                into,
                actor_ref,
            )
        else:
            shared_by = await conn.fetchval(
                "SELECT shared_by FROM clauses WHERE customer_id = $1 AND id = $2",
                customer_id,
                into,
            )

    return Declaration(
        clause_id=into,
        created=False,
        evidence_id=evidence_id,
        situation_id=situation_id,
        shared_by=shared_by,
    )


def _lineage_for(relation: Relation, related_clause_id: UUID | None) -> dict[str, Any]:
    if related_clause_id is None or relation is Relation.NEW:
        return {}
    if relation is Relation.VARIANT:
        return {"variant_of": [str(related_clause_id)]}
    if relation is Relation.CONFLICT:
        return {"conflicts_with": [str(related_clause_id)]}
    return {}


async def _add_reciprocal_conflict(
    conn: Any, customer_id: str, existing_id: UUID, new_id: UUID
) -> None:
    """Point the existing clause back at the new one.

    Without this, serving the existing clause shows a clean rule with no hint
    that a competing one exists -- and it is exactly as likely to be the one
    retrieved. A one-directional conflict edge is worse than none, because it
    reads as a check that was performed.

    `jsonb_set` with a merged array rather than a blind overwrite: a clause can
    conflict with more than one other, and `-` before the append keeps a repeat
    declaration from stacking the same id twice.
    """
    await conn.execute(
        """
        UPDATE clauses
           SET lineage = jsonb_set(
                   COALESCE(lineage, '{}'::jsonb),
                   ARRAY['conflicts_with'],
                   (
                       COALESCE(lineage -> 'conflicts_with', '[]'::jsonb)
                       - $3::text
                   ) || to_jsonb(ARRAY[$3::text]),
                   true
               )
         WHERE customer_id = $1 AND id = $2
        """,
        customer_id,
        existing_id,
        str(new_id),
    )


async def _attach_situation(
    conn: Any,
    customer_id: str,
    clause_id: UUID,
    situation_id: UUID,
    classification: dict[str, Any] | None,
) -> None:
    await conn.execute(
        """
        INSERT INTO clause_situation_edges
            (customer_id, clause_id, situation_id, classification)
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (clause_id, situation_id) DO NOTHING
        """,
        customer_id,
        clause_id,
        situation_id,
        json.dumps(classification or {"method": "human"}),
    )


async def _add_evidence(
    conn: Any,
    customer_id: str,
    clause_id: UUID,
    *,
    actor_ref: str,
    source_ref: dict[str, Any],
) -> UUID:
    """The declaration's own evidence row.

    `exposure_tainted` is FALSE and is passed explicitly rather than left to a
    column default -- the column has none, deliberately, because a default here
    fails open: an omitted taint flag would silently count a served-then-echoed
    clause as independent support. A declaration typed by a human is the one case
    that is genuinely untainted, so this is the right place to say so out loud.
    """
    return await conn.fetchval(
        """
        INSERT INTO clause_evidence
            (customer_id, clause_id, source_class, source_ref,
             author_ref, exposure_tainted, ts)
        VALUES ($1, $2, $3, $4::jsonb, $5, FALSE, now())
        RETURNING id
        """,
        customer_id,
        clause_id,
        DECLARED_SOURCE_CLASS,
        json.dumps(source_ref),
        actor_ref,
    )


async def _clause_exists(conn: Any, customer_id: str, clause_id: UUID) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM clauses WHERE customer_id = $1 AND id = $2)",
            customer_id,
            clause_id,
        )
    )


async def _embed_all(embedder: Any, texts: list[str]) -> list[list[float]] | None:
    """Embed in one batch, reordered by `chunk_index`.

    Same rule as the classifier and for the same reason: `EmbedResult` splits
    into `embedded` and `failed`, and the base embedder's half-split appends
    sub-batches as they resolve, so the i-th embedded entry is not the i-th
    input. Positional zipping here would score the draft against the wrong
    clause -- and the symptom would be neighbour lists that are subtly, silently
    wrong rather than an error.
    """
    result = await embedder.embed_many(texts)
    by_index: dict[int, list[float]] = {}
    for chunk in getattr(result, "embedded", None) or []:
        by_index[chunk.chunk_index] = chunk.embedding
    if len(by_index) != len(texts) or any(i not in by_index for i in range(len(texts))):
        return None
    return [by_index[i] for i in range(len(texts))]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = (norm_a**0.5) * (norm_b**0.5)
    return dot / denom if denom else 0.0


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

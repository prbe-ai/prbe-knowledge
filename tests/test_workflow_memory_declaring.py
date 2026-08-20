"""The declared write path: ingest scanning, and collisions nobody resolves quietly.

WHAT THIS FILE HAS TO DEFEND:

* THE SECRET SCAN ACTUALLY RUNS. `assert_clean` / `assert_clean_json` shipped in
  Stage 0 with no call site anywhere, which made §4's "secret-scan at ingest" a
  paper guarantee -- the scanner was tested exhaustively against strings nobody
  fed it. This is the ingest, and a refusal must leave NOTHING behind.
* EVERY AUTHOR-SUPPLIED FIELD IS SCANNED, not only the two the spec names. A
  credential pasted into a rule does not know which column it is heading for.
* A MERGE DOES NOT CREATE A CLAUSE. It adds the declarer as evidence, which is
  the act that unlocks a single-author rule for the team. Writing a second
  near-identical clause instead leaves both invisible forever, each with one
  author -- the failure this case exists to prevent.
* CONFLICT EDGES POINT BOTH WAYS. A one-directional edge is worse than none: it
  reads as a check that was performed, while the clause on the other end serves
  clean with no hint a competing rule exists.
* NOTHING IS RESOLVED WITHOUT BEING TOLD. `find_neighbours` decides nothing;
  `declare` executes only the relation it was handed.

Run with the isolated wfmem database (these fixtures TRUNCATE):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_declaring.py -q
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from engine.shared.db import raw_conn, with_tenant
from engine.shared.wfmem.declaring import (
    DECLARED_STATUS,
    NEIGHBOUR_FLOOR,
    DeclarationRefused,
    Relation,
    declare,
    find_neighbours,
)
from engine.shared.wfmem.secret_scan import SecretDetected
from engine.shared.wfmem.situations import seed_situations
from engine.shared.wfmem.structuring import ClauseDraft

TENANT = "cust-wfmem-decl-a"
OTHER_TENANT = "cust-wfmem-decl-b"

ALICE = "user:alice"
BOB = "user:bob"

SOURCE_REF = {"session": "sess-1", "span": [0, 40]}


def _draft(
    body: str = "open a Probe run before the first GPU step",
    *,
    kind: str = "step",
    binding: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    semantic_action: str | None = None,
) -> ClauseDraft:
    return ClauseDraft(
        kind=kind,
        body=body,
        semantic_action=semantic_action,
        binding=binding or {},
        scope=scope or {},
    )


@pytest_asyncio.fixture
async def tenant(live_db: None) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-decl-a', 'h-wfmem-decl-a'),
                   ($2, 'wfmem-decl-b', 'h-wfmem-decl-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT,
            OTHER_TENANT,
        )
    yield TENANT


async def _row(customer_id: str, clause_id: UUID) -> Any:
    async with with_tenant(customer_id) as conn:
        return await conn.fetchrow(
            "SELECT * FROM clauses WHERE customer_id = $1 AND id = $2",
            customer_id,
            clause_id,
        )


async def _counts(customer_id: str) -> tuple[int, int, int]:
    async with with_tenant(customer_id) as conn:
        clauses = await conn.fetchval(
            "SELECT count(*) FROM clauses WHERE customer_id = $1", customer_id
        )
        evidence = await conn.fetchval(
            "SELECT count(*) FROM clause_evidence WHERE customer_id = $1", customer_id
        )
        edges = await conn.fetchval(
            "SELECT count(*) FROM clause_situation_edges WHERE customer_id = $1",
            customer_id,
        )
    return clauses, evidence, edges


def _lineage(row: Any) -> dict[str, Any]:
    raw = row["lineage"]
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


async def _situation(customer_id: str, slug: str = "launch-run") -> UUID:
    async with with_tenant(customer_id) as conn:
        await seed_situations(conn, customer_id)
        return await conn.fetchval(
            "SELECT id FROM situations WHERE customer_id = $1 AND slug = $2",
            customer_id,
            slug,
        )


# --------------------------------------------------------------------------
# The plain write
# --------------------------------------------------------------------------


async def test_a_declaration_writes_clause_evidence_and_edge(tenant: str) -> None:
    situation_id = await _situation(tenant)
    result = await declare(
        tenant,
        _draft(),
        actor_ref=ALICE,
        source_ref=SOURCE_REF,
        situation_id=situation_id,
        classification={"method": "llm", "confidence": 0.81, "prompt_version": "1"},
    )

    assert result.created is True
    assert await _counts(tenant) == (1, 1, 1)

    row = await _row(tenant, result.clause_id)
    assert row["body"] == "open a Probe run before the first GPU step"
    assert row["status"] == DECLARED_STATUS
    assert row["author_ref"] == ALICE
    assert row["version"] == 1


async def test_the_evidence_row_is_explicitly_untainted(tenant: str) -> None:
    """The column has no default, deliberately: a default would fail OPEN.

    An omitted taint flag would silently count a served-then-echoed clause as
    independent support, which is the self-reinforcement loop the design exists
    to prevent. A human typing a rule is the one genuinely untainted case, so
    this path says so out loud rather than relying on an absence.
    """
    result = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    async with with_tenant(tenant) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM clause_evidence WHERE id = $1", result.evidence_id
        )
    assert row["exposure_tainted"] is False
    assert row["source_class"] == "declared"
    assert row["author_ref"] == ALICE


async def test_a_declaration_without_a_situation_still_writes(tenant: str) -> None:
    """An unclassified rule is still a rule. The classifier answering `unknown`
    must not cost the author their declaration."""
    result = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    assert result.situation_id is None
    assert await _counts(tenant) == (1, 1, 0)


async def test_a_declaration_must_record_who_made_it(tenant: str) -> None:
    with pytest.raises(DeclarationRefused, match="who made it"):
        await declare(tenant, _draft(), actor_ref="", source_ref=SOURCE_REF)
    assert await _counts(tenant) == (0, 0, 0)


# --------------------------------------------------------------------------
# The secret scan -- its first call site
# --------------------------------------------------------------------------


async def test_a_credential_in_the_body_is_refused(tenant: str) -> None:
    draft = _draft(body="deploy with token ghp_" + "a" * 36 + " every time")
    with pytest.raises(SecretDetected):
        await declare(tenant, draft, actor_ref=ALICE, source_ref=SOURCE_REF)


async def test_a_refused_declaration_leaves_nothing_behind(tenant: str) -> None:
    """The scan runs BEFORE the transaction opens.

    A half-written rule would be worse than a rejected one: somebody finds the
    row later, cannot tell why it is orphaned, and the credential is already at
    rest.
    """
    draft = _draft(body="use ghp_" + "b" * 36)
    with pytest.raises(SecretDetected):
        await declare(tenant, draft, actor_ref=ALICE, source_ref=SOURCE_REF)
    assert await _counts(tenant) == (0, 0, 0)


@pytest.mark.parametrize(
    ("field", "draft_kwargs"),
    [
        ("binding", {"binding": {"argv_template": "psql postgres://u:hunter2xyz@h/db"}}),
        ("scope", {"scope": {"note": "ghp_" + "c" * 36}}),
        ("semantic_action", {"semantic_action": "ghp_" + "d" * 36}),
    ],
)
async def test_every_author_supplied_field_is_scanned(
    tenant: str, field: str, draft_kwargs: dict[str, Any]
) -> None:
    """Not only `body` and `binding`, which is all the spec's wording names.

    A credential pasted into a rule does not know which column it is heading
    for, and the fields differ only in which key the author happened to put it
    under.
    """
    with pytest.raises(SecretDetected):
        await declare(tenant, _draft(**draft_kwargs), actor_ref=ALICE, source_ref=SOURCE_REF)
    assert await _counts(tenant) == (0, 0, 0), f"{field} left a row behind"


async def test_the_source_ref_is_scanned_too(tenant: str) -> None:
    with pytest.raises(SecretDetected):
        await declare(
            tenant,
            _draft(),
            actor_ref=ALICE,
            source_ref={"session": "ghp_" + "e" * 36},
        )
    assert await _counts(tenant) == (0, 0, 0)


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


async def test_a_merge_adds_evidence_instead_of_a_second_clause(tenant: str) -> None:
    first = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)

    second = await declare(
        tenant,
        _draft(),
        actor_ref=BOB,
        source_ref={"session": "sess-2"},
        relation=Relation.MERGE,
        related_clause_id=first.clause_id,
    )

    assert second.created is False
    assert second.clause_id == first.clause_id
    clauses, evidence, _ = await _counts(tenant)
    assert (clauses, evidence) == (1, 2)


async def test_a_merge_is_what_unlocks_a_single_author_rule(tenant: str) -> None:
    """The visibility guard needs a second DISTINCT human, and this is how one arrives.

    Writing a near-identical second clause instead would leave both rules
    invisible forever, each with exactly one author -- the same team ending up
    with two private copies of one shared practice.
    """
    from engine.shared.wfmem.visibility import fetch_visible_clauses

    first = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)

    async with with_tenant(tenant) as conn:
        assert await fetch_visible_clauses(conn, BOB) == []

    await declare(
        tenant,
        _draft(),
        actor_ref=BOB,
        source_ref={"session": "sess-2"},
        relation=Relation.MERGE,
        related_clause_id=first.clause_id,
    )

    async with with_tenant(tenant) as conn:
        visible = await fetch_visible_clauses(conn, "user:carol")
    assert [row["id"] for row in visible] == [first.clause_id]


async def test_a_merge_must_name_its_target(tenant: str) -> None:
    with pytest.raises(DeclarationRefused, match="merges into"):
        await declare(
            tenant,
            _draft(),
            actor_ref=ALICE,
            source_ref=SOURCE_REF,
            relation=Relation.MERGE,
        )
    assert await _counts(tenant) == (0, 0, 0)


async def test_a_merge_may_add_a_situation_the_original_lacked(tenant: str) -> None:
    situation_id = await _situation(tenant)
    first = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    assert (await _counts(tenant))[2] == 0

    await declare(
        tenant,
        _draft(),
        actor_ref=BOB,
        source_ref={"session": "sess-2"},
        situation_id=situation_id,
        relation=Relation.MERGE,
        related_clause_id=first.clause_id,
    )
    assert (await _counts(tenant))[2] == 1


async def test_merging_into_a_missing_clause_is_refused(tenant: str) -> None:
    with pytest.raises(DeclarationRefused, match="does not exist"):
        await declare(
            tenant,
            _draft(),
            actor_ref=ALICE,
            source_ref=SOURCE_REF,
            relation=Relation.MERGE,
            related_clause_id=uuid4(),
        )


async def test_merging_into_another_tenants_clause_is_refused(tenant: str) -> None:
    """The refusal must be indistinguishable from "no such clause".

    A caller that could tell the two apart would have an existence oracle over
    every other tenant's clause ids -- the same leak the composite foreign keys
    close at the schema level.
    """
    theirs = await declare(OTHER_TENANT, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    with pytest.raises(DeclarationRefused, match="does not exist"):
        await declare(
            tenant,
            _draft(),
            actor_ref=BOB,
            source_ref=SOURCE_REF,
            relation=Relation.MERGE,
            related_clause_id=theirs.clause_id,
        )


# --------------------------------------------------------------------------
# Variant and conflict
# --------------------------------------------------------------------------


async def test_a_variant_records_what_it_varies_from(tenant: str) -> None:
    first = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    second = await declare(
        tenant,
        _draft(binding={"cwd_glob": "research/**"}),
        actor_ref=BOB,
        source_ref=SOURCE_REF,
        relation=Relation.VARIANT,
        related_clause_id=first.clause_id,
    )

    assert second.created is True
    assert _lineage(await _row(tenant, second.clause_id)) == {
        "variant_of": [str(first.clause_id)]
    }
    # One-directional on purpose: "is a variant of" has a direction, unlike
    # "conflicts with", and the original is not a variant of its own offshoot.
    assert _lineage(await _row(tenant, first.clause_id)) == {}


async def test_a_conflict_is_recorded_on_both_clauses(tenant: str) -> None:
    """A one-directional conflict edge is worse than none.

    Serving the original would show a clean rule with no hint that a competing
    one exists -- and it is exactly as likely to be the one retrieved. The edge
    reads as a check that was performed.
    """
    first = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    second = await declare(
        tenant,
        _draft(body="never open a Probe run before the first GPU step"),
        actor_ref=BOB,
        source_ref=SOURCE_REF,
        relation=Relation.CONFLICT,
        related_clause_id=first.clause_id,
    )

    assert _lineage(await _row(tenant, second.clause_id)) == {
        "conflicts_with": [str(first.clause_id)]
    }
    assert _lineage(await _row(tenant, first.clause_id)) == {
        "conflicts_with": [str(second.clause_id)]
    }


async def test_a_clause_may_conflict_with_several_others(tenant: str) -> None:
    first = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    second = await declare(
        tenant,
        _draft(body="never do that"),
        actor_ref=BOB,
        source_ref=SOURCE_REF,
        relation=Relation.CONFLICT,
        related_clause_id=first.clause_id,
    )
    third = await declare(
        tenant,
        _draft(body="do it only on Tuesdays"),
        actor_ref=BOB,
        source_ref=SOURCE_REF,
        relation=Relation.CONFLICT,
        related_clause_id=first.clause_id,
    )

    recorded = _lineage(await _row(tenant, first.clause_id))["conflicts_with"]
    assert set(recorded) == {str(second.clause_id), str(third.clause_id)}
    assert len(recorded) == 2, "the reciprocal append must not stack duplicates"


async def test_a_variant_or_conflict_must_name_its_counterpart(tenant: str) -> None:
    for relation in (Relation.VARIANT, Relation.CONFLICT):
        with pytest.raises(DeclarationRefused, match="relates to"):
            await declare(
                tenant,
                _draft(),
                actor_ref=ALICE,
                source_ref=SOURCE_REF,
                relation=relation,
            )
    assert await _counts(tenant) == (0, 0, 0)


async def test_relating_to_another_tenants_clause_is_refused(tenant: str) -> None:
    theirs = await declare(OTHER_TENANT, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    with pytest.raises(DeclarationRefused, match="does not exist"):
        await declare(
            tenant,
            _draft(),
            actor_ref=BOB,
            source_ref=SOURCE_REF,
            relation=Relation.CONFLICT,
            related_clause_id=theirs.clause_id,
        )
    assert await _counts(tenant) == (0, 0, 0)


# --------------------------------------------------------------------------
# Neighbours -- which decide nothing
# --------------------------------------------------------------------------


@dataclass
class _Chunk:
    chunk_index: int
    embedding: list[float]


@dataclass
class _EmbedResult:
    """The two-list shape of the real `EmbedResult`, minimally."""

    embedded: list[_Chunk]
    failed: list[Any]


#: Neighbour tests use the REAL embedder, which conftest puts in stub mode by
#: blanking the provider keys: `_hash_vector` derives a deterministic unit
#: vector from the text's SHA-256. That is better than a hand-authored fake
#: here, not a compromise -- identical bodies produce identical vectors
#: (cosine 1.0) and different bodies produce near-orthogonal ones, which is
#: exactly the property these tests assert. It also exercises the real
#: dimension (3072), and a fake with a convenient 2-element vector would fail
#: the halfvec column rather than testing anything.


async def test_a_duplicate_rule_is_surfaced(tenant: str) -> None:
    """Somebody declaring a rule the team already has, which is THE case.

    Verbatim rather than merely similar, because the stub embedder is a hash:
    identical text scores 1.0 and anything else is near-orthogonal, so this
    proves retrieval, storage and the model filter, NOT semantic nearness.
    True near-duplicate behaviour is only exercised by the opt-in real-key
    tests -- see `test_semantic_neighbours_need_a_real_embedder` below, which
    says so out loud rather than leaving the gap implied.
    """
    await declare(tenant, _draft(body="existing rule"), actor_ref=ALICE, source_ref=SOURCE_REF)

    found = await find_neighbours(tenant, _draft(body="existing rule"))

    assert len(found) == 1
    assert found[0].body == "existing rule"
    assert found[0].similarity == pytest.approx(1.0, abs=1e-3)
    assert found[0].same_binding is True


async def test_a_distant_clause_is_not_surfaced(tenant: str) -> None:
    """Guarded against passing vacuously: an identical body IS found in the same
    tenant, so the empty result below is the floor working rather than the
    search being broken."""
    await declare(tenant, _draft(body="unrelated rule"), actor_ref=ALICE, source_ref=SOURCE_REF)

    assert await find_neighbours(tenant, _draft(body="unrelated rule")) != []
    assert await find_neighbours(tenant, _draft(body="a completely different rule")) == []
    assert NEIGHBOUR_FLOOR > 0.0


async def test_a_neighbour_with_a_different_binding_is_flagged_as_such(tenant: str) -> None:
    """Same words, different target. The author is the only one who can say
    whether that is a duplicate or a deliberate variant, so the write path is
    told rather than guessing."""
    await declare(
        tenant,
        _draft(body="rule", binding={"cwd_glob": "a/**"}),
        actor_ref=ALICE,
        source_ref=SOURCE_REF,
    )

    found = await find_neighbours(
        tenant, _draft(body="rule", binding={"cwd_glob": "b/**"}),
    )
    assert found[0].same_binding is False


async def test_neighbours_span_situations(tenant: str) -> None:
    """A duplicate filed under a different situation is still a duplicate, and it
    is the one a situation-scoped search would never show."""
    launch = await _situation(tenant, "launch-run")
    await declare(
        tenant,
        _draft(body="existing rule"),
        actor_ref=ALICE,
        source_ref=SOURCE_REF,
        situation_id=launch,
    )

    found = await find_neighbours(tenant, _draft(body="existing rule"))
    assert len(found) == 1


async def test_another_tenants_clauses_are_never_neighbours(tenant: str) -> None:
    await declare(OTHER_TENANT, _draft(body="their rule"), actor_ref=ALICE, source_ref=SOURCE_REF)

    assert await find_neighbours(tenant, _draft()) == []


async def test_an_embedder_failure_costs_neighbours_not_the_declaration(tenant: str) -> None:
    """Degrade, do not raise. Refusing the write because the embedder is down
    would take the whole declared path out over a nice-to-have."""

    class BrokenEmbedder:
        """Answers with no vectors at all -- the provider-down shape."""

        model_id = "broken"

        async def embed_many(self, texts: list[str]) -> Any:
            return _EmbedResult(embedded=[], failed=[])

    await declare(tenant, _draft(body="existing"), actor_ref=ALICE, source_ref=SOURCE_REF)
    assert await find_neighbours(tenant, _draft(), embedder=BrokenEmbedder()) == []

    result = await declare(tenant, _draft(body="another"), actor_ref=BOB, source_ref=SOURCE_REF)
    assert result.created is True


async def test_finding_neighbours_writes_nothing(tenant: str) -> None:
    """The read half decides nothing and leaves nothing. Everything it surfaces
    is a suggestion for a human, and a suggestion that wrote a row would be a
    resolution."""
    await declare(tenant, _draft(body="existing"), actor_ref=ALICE, source_ref=SOURCE_REF)
    before = await _counts(tenant)

    await find_neighbours(tenant, _draft(body="same"))

    assert await _counts(tenant) == before


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="needs a real embedding key; the stub is a hash and has no semantics",
)
async def test_semantic_neighbours_need_a_real_embedder(tenant: str) -> None:
    """The gap the stub leaves, made explicit instead of implied.

    Everything else in this section proves plumbing: storage, the indexed query,
    the model filter, the floor. NONE of it proves the thing the feature is FOR
    -- catching a rule that means the same as an existing one in different
    words. That needs real embeddings, so it lives here and skips by default
    rather than being quietly absent.
    """
    await declare(
        tenant,
        _draft(body="always open a Probe run before the first GPU step"),
        actor_ref=ALICE,
        source_ref=SOURCE_REF,
    )

    found = await find_neighbours(
        tenant, _draft(body="start a run in Probe before you begin training")
    )

    assert found, "a paraphrase of an existing rule must surface as a neighbour"
    assert found[0].similarity >= NEIGHBOUR_FLOOR


async def test_a_clause_stores_its_embedding_and_the_model_that_made_it(tenant: str) -> None:
    """Both columns, or the row is unusable for search.

    A vector with no model id cannot be safely compared -- a cosine across two
    embedders is a plausible-looking number rather than an error -- so the CHECK
    keeps them together and this pins that the write path fills both.
    """
    result = await declare(tenant, _draft(), actor_ref=ALICE, source_ref=SOURCE_REF)
    row = await _row(tenant, result.clause_id)
    assert row["body_embedding"] is not None
    assert row["body_embedding_model"]


async def test_a_clause_embedded_by_another_model_is_excluded(tenant: str) -> None:
    """Not merely deprioritised -- excluded.

    Comparing vectors from two embedders degrades the result INVISIBLY: the
    numbers look reasonable and the ranking is meaningless. Being absent from
    the list is at least a gap somebody can notice.
    """
    await declare(tenant, _draft(body="existing rule"), actor_ref=ALICE, source_ref=SOURCE_REF)
    assert await find_neighbours(tenant, _draft(body="existing rule")) != []

    async with with_tenant(tenant) as conn:
        await conn.execute(
            "UPDATE clauses SET body_embedding_model = 'some-older-model' "
            "WHERE customer_id = $1",
            tenant,
        )

    assert await find_neighbours(tenant, _draft(body="existing rule")) == []


async def test_editing_a_body_clears_the_stale_embedding(tenant: str) -> None:
    """A stored vector describes text that can change under it, and the
    staleness is invisible -- the search keeps returning results, just wrong
    ones. The trigger drops the clause out of neighbour search instead, which is
    the safe direction to fail. Nothing in v0 edits a body; this guards the edit
    path somebody adds later."""
    result = await declare(tenant, _draft(body="existing rule"), actor_ref=ALICE, source_ref=SOURCE_REF)

    async with with_tenant(tenant) as conn:
        await conn.execute(
            "UPDATE clauses SET body = 'a materially different rule' WHERE id = $1",
            result.clause_id,
        )

    row = await _row(tenant, result.clause_id)
    assert row["body_embedding"] is None
    assert row["body_embedding_model"] is None
    assert await find_neighbours(tenant, _draft(body="a materially different rule")) == []


async def test_a_non_body_update_keeps_the_embedding(tenant: str) -> None:
    """The trigger keys on the BODY changing, not on any update. Publishing a
    rule must not silently cost it its place in duplicate detection."""
    result = await declare(tenant, _draft(body="existing rule"), actor_ref=ALICE, source_ref=SOURCE_REF)

    async with with_tenant(tenant) as conn:
        await conn.execute(
            "UPDATE clauses SET shared_by = $2, shared_at = now() WHERE id = $1",
            result.clause_id,
            ALICE,
        )

    assert (await _row(tenant, result.clause_id))["body_embedding"] is not None
    assert await find_neighbours(tenant, _draft(body="existing rule")) != []

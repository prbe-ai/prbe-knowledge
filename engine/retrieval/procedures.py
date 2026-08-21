"""HTTP surface for workflow memory: preview, declare, query.

Three endpoints, and the shape of the set is the point. `/preview` runs the
structuring pass, the classifier and the neighbour search and WRITES NOTHING;
`/declare` writes exactly what a human confirmed; `/query` serves. The two-call
declaration flow exists so the confirmation echo is real -- a single endpoint
that structured and wrote in one round trip would make `clauses.body`
"whatever the model produced", and the reclassification contract depends on it
being what a person approved.

GATING IS PER-CAPABILITY AND FAILS CLOSED. Writes need `wfmem_input_declared`,
reads need `wfmem_output_retrieval`, and a tenant without the cell gets an empty
result rather than a 403. That is deliberate: the three-state envelope carries
`enabled` / `entitled` / `upgrade_url`, and an HTTP status cannot spell the
difference between "off because nobody turned it on" and "off because the plan
does not include it". A client that cannot tell those apart concludes the switch
is broken -- renders a toggle, flips it, nothing happens, files a bug.

`actor_ref` IS ACCEPTED FROM THE CALLER HERE, and that is safe only because of
where this sits. This service is trusted-internal: `authenticate_query` admits
either a customer API key or the `X-Internal-Knowledge-Key` + `X-Prbe-Customer`
pair, and research-os is the only caller of the latter. research-os derives the
actor from its own authenticated principal and never accepts it from ITS client.
The trust boundary is there, not here -- which is exactly why the research-os
route must never pass a client-supplied actor through. If this service ever
becomes reachable by an untrusted caller, this field is the first thing to
revisit: a caller who could name any actor could both read another person's
single-author clauses and write audit rows in their name.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from engine.retrieval.auth import authenticate_query
from engine.shared.db import with_tenant
from engine.shared.logging import get_logger
from engine.shared.wfmem.capabilities import (
    InputPath,
    OutputSurface,
    capability_envelope,
    input_capability_key,
    is_capability_enabled,
    output_capability_key,
)
from engine.shared.wfmem.classifier import Outcome, classify
from engine.shared.wfmem.declaring import (
    DeclarationRefused,
    Neighbour,
    Relation,
    declare,
    find_neighbours,
    publish_clause,
    unpublish_clause,
)
from engine.shared.wfmem.secret_scan import SecretDetected
from engine.shared.wfmem.serving import DEFAULT_LIMIT, ServedClause, serve_clauses
from engine.shared.wfmem.structuring import ClauseDraft, StructuringFailed, structure

log = get_logger(__name__)

procedures_router = APIRouter()

_DECLARED_KEY = input_capability_key(InputPath.DECLARED)
_RETRIEVAL_KEY = output_capability_key(OutputSurface.RETRIEVAL)


# --------------------------------------------------------------------------
# Wire shapes
# --------------------------------------------------------------------------


class DraftOut(BaseModel):
    kind: str
    body: str
    semantic_action: str | None = None
    binding: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)


class DraftIn(DraftOut):
    """The draft coming back for the write.

    Same fields as `DraftOut` because the client is expected to send back what it
    was shown -- POSSIBLY EDITED. Accepting edits is the point: a confirmation
    the human can only accept or reject is a worse instrument than one they can
    correct, and the corrected text is what `clauses.body` should hold. The
    server re-validates everything regardless; nothing here trusts the round
    trip, and the secret scan runs on the way in either way.
    """


class ClassificationOut(BaseModel):
    outcome: str
    slug: str | None = None
    situation_id: UUID | None = None
    confidence: float = 0.0
    method: str
    runner_up: str | None = None


class NeighbourOut(BaseModel):
    clause_id: UUID
    body: str
    status: str
    author_ref: str
    similarity: float
    same_binding: bool


class CapabilityOut(BaseModel):
    """The house three-state shape. `entitled` is hardcoded True until Phase 3."""

    enabled: bool
    entitled: bool = True
    upgrade_url: str | None = None


class PreviewRequest(BaseModel):
    prose: str = Field(min_length=1, max_length=4_000)
    context: dict[str, Any] = Field(default_factory=dict)


class PreviewResponse(BaseModel):
    capability: CapabilityOut
    draft: DraftOut | None = None
    classification: ClassificationOut | None = None
    neighbours: list[NeighbourOut] = Field(default_factory=list)


class DeclareRequest(BaseModel):
    draft: DraftIn
    actor_ref: str = Field(min_length=1, max_length=200)
    source_ref: dict[str, Any] = Field(default_factory=dict)
    situation_id: UUID | None = None
    classification: dict[str, Any] | None = None
    relation: Relation = Relation.NEW
    related_clause_id: UUID | None = None
    #: Make it visible to the team now, on this author's authority, instead of
    #: waiting for a second human to independently agree. Defaults False and
    #: must keep doing so: the two-human rule is what stops a private habit
    #: becoming false policy, and most declarations really are one person's note.
    publish: bool = False

    @model_validator(mode="after")
    def _relation_needs_a_counterpart(self) -> DeclareRequest:
        """Caught here as well as in the write path, so a malformed request is a
        422 with a field name rather than a 400 from three layers down."""
        if self.relation is not Relation.NEW and self.related_clause_id is None:
            raise ValueError(f"relation={self.relation.value} requires related_clause_id")
        return self


class DeclareResponse(BaseModel):
    capability: CapabilityOut
    clause_id: UUID | None = None
    #: False when a merge folded this into an existing clause. A client that
    #: reported "rule created" on a merge would be lying about the one outcome
    #: the author most needs to see.
    created: bool = False
    evidence_id: UUID | None = None
    situation_id: UUID | None = None
    #: `situation_id` is the `misc` bucket because nothing fit, not a situation
    #: anybody chose. Reported rather than swallowed: the fallback stops a
    #: clause being unreachable, it does not make it correctly filed, and only
    #: the author can move it somewhere real.
    situation_fallback: bool = False
    #: Who published it, or None if it is waiting on a second human. The client
    #: needs this to tell the author which of two very different things just
    #: happened -- "your rule is live for the team" versus "saved, and private
    #: until somebody agrees" -- and guessing wrong in either direction is bad.
    shared_by: str | None = None
    refused: str | None = None


class PublishRequest(BaseModel):
    clause_id: UUID
    actor_ref: str = Field(min_length=1, max_length=200)
    #: False withdraws a publication. The undo exists because publishing is
    #: unilateral: without it, the only ways to walk back a rule the team turns
    #: out to disagree with are deleting the clause -- destroying its evidence
    #: and history -- or leaving false policy in front of everybody.
    published: bool = True


class PublishResponse(BaseModel):
    capability: CapabilityOut
    clause_id: UUID | None = None
    shared_by: str | None = None
    changed: bool = False
    refused: str | None = None


class QueryRequest(BaseModel):
    actor_ref: str = Field(min_length=1, max_length=200)
    query: str | None = None
    situation_slug: str | None = None
    session_id: str | None = None
    limit: int = DEFAULT_LIMIT


class ClauseOut(BaseModel):
    id: UUID
    kind: str
    body: str
    status: str
    author_ref: str
    version: int
    binding: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    #: Label, never gate. A rule one person declared and published themselves is
    #: a different claim from one the team demonstrably follows, and a surface
    #: that drops these two fields renders both identically -- reaching, via the
    #: feature meant to work around the two-human guard, exactly the failure
    #: that guard exists to prevent.
    shared_by: str | None = None
    human_backers: int = 0
    #: Served out of the `misc` bucket because the requested situation had no
    #: rules. The surface must SAY so -- an unfiled rule rendered as a match is
    #: confident irrelevant advice, which is how an agent learns to ignore this
    #: whole surface.
    from_fallback: bool = False


class QueryResponse(BaseModel):
    capability: CapabilityOut
    clauses: list[ClauseOut] = Field(default_factory=list)
    serve_id: int | None = None
    classification: ClassificationOut | None = None
    #: Set when the tenant has no situation vocabulary at all. Carried as its own
    #: field rather than folded into an empty `clauses` list because those two
    #: states are otherwise identical on the wire, and the difference is
    #: "nothing matched" versus "this feature was never finished being turned
    #: on". Somebody has to be able to see the second one.
    no_situations_configured: bool = False


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@procedures_router.post("/procedures/preview", response_model=PreviewResponse)
async def preview(
    req: PreviewRequest,
    customer_id: str = Depends(authenticate_query),
) -> PreviewResponse:
    """Structure, classify and look for neighbours. Writes nothing."""
    capability = await _envelope(customer_id, _DECLARED_KEY)
    if not capability.enabled:
        return PreviewResponse(capability=capability)

    try:
        draft = await structure(req.prose, req.context)
    except StructuringFailed as exc:
        # 502, not 500: the failure is upstream of us and the caller's retry is
        # the right response. The detail is the model's shortcoming, never the
        # prose -- a declared rule is exactly the text the secret scanner exists
        # for, and an error body is the last place it should be echoed.
        log.warning("wfmem_procedures.structuring_failed", customer=customer_id, reason=str(exc))
        raise _bad_gateway(str(exc)) from exc

    classification = await classify(customer_id, draft.body)
    neighbours = await find_neighbours(customer_id, draft)

    return PreviewResponse(
        capability=capability,
        draft=_draft_out(draft),
        classification=_classification_out(classification),
        neighbours=[_neighbour_out(n) for n in neighbours],
    )


@procedures_router.post("/procedures/declare", response_model=DeclareResponse)
async def declare_rule(
    req: DeclareRequest,
    customer_id: str = Depends(authenticate_query),
) -> DeclareResponse:
    """Write the confirmed rule."""
    capability = await _envelope(customer_id, _DECLARED_KEY)
    if not capability.enabled:
        return DeclareResponse(capability=capability)

    draft = ClauseDraft(
        kind=req.draft.kind,
        body=req.draft.body,
        semantic_action=req.draft.semantic_action,
        binding=req.draft.binding,
        scope=req.draft.scope,
    )

    try:
        result = await declare(
            customer_id,
            draft,
            actor_ref=req.actor_ref,
            source_ref=req.source_ref,
            situation_id=req.situation_id,
            classification=req.classification,
            relation=req.relation,
            related_clause_id=req.related_clause_id,
            publish=req.publish,
        )
    except SecretDetected as exc:
        # 422 with the DETECTOR NAMES, never the matched text. The author needs
        # to know which shape tripped it so they can redact and retry; echoing
        # the match back would put the credential in a response body, a log line
        # and probably a terminal scrollback.
        log.warning("wfmem_procedures.secret_refused", customer=customer_id)
        raise _unprocessable(
            "that rule looks like it contains a credential "
            f"({', '.join(exc.args[0]) if exc.args else 'unknown'}); "
            "remove it and declare the rule without the secret"
        ) from exc
    except DeclarationRefused as exc:
        return DeclareResponse(capability=capability, refused=str(exc))

    return DeclareResponse(
        capability=capability,
        clause_id=result.clause_id,
        created=result.created,
        evidence_id=result.evidence_id,
        situation_id=result.situation_id,
        shared_by=result.shared_by,
        situation_fallback=result.situation_fallback,
    )


@procedures_router.post("/procedures/publish", response_model=PublishResponse)
async def publish_rule(
    req: PublishRequest,
    customer_id: str = Depends(authenticate_query),
) -> PublishResponse:
    """Publish an existing clause to the team, or withdraw a publication.

    Separate from `/declare` because these are different decisions made at
    different times: somebody writes a note for themselves, and weeks later
    decides the team should follow it. Routing that through `/declare` would
    mean retyping the rule, which produces a SECOND clause rather than
    publishing the first.

    Gated on the declared-input capability rather than the retrieval one --
    publishing changes what is written about a rule, not how it is served.
    """
    capability = await _envelope(customer_id, _DECLARED_KEY)
    if not capability.enabled:
        return PublishResponse(capability=capability)

    try:
        if req.published:
            shared_by = await publish_clause(customer_id, req.clause_id, actor_ref=req.actor_ref)
            if shared_by is None:
                return PublishResponse(
                    capability=capability,
                    refused="that rule does not exist in this workspace",
                )
            return PublishResponse(
                capability=capability,
                clause_id=req.clause_id,
                shared_by=shared_by,
                changed=True,
            )

        changed = await unpublish_clause(customer_id, req.clause_id)
        return PublishResponse(
            capability=capability, clause_id=req.clause_id, changed=changed
        )
    except DeclarationRefused as exc:
        return PublishResponse(capability=capability, refused=str(exc))


@procedures_router.post("/procedures/query", response_model=QueryResponse)
async def query_procedures(
    req: QueryRequest,
    customer_id: str = Depends(authenticate_query),
) -> QueryResponse:
    """Serve the rules this actor may see.

    Three ways to ask, and they mean different things:

    * `situation_slug` names the situation directly -- the caller already knows.
    * `query` gets classified, and an `unknown` classification serves ZERO
      clauses rather than falling back to everything. Serving a broadly-scoped
      rule into a situation nobody could identify is the failure the escape
      hatch exists to prevent.
    * Neither serves everything the actor may see, which is `/pull-rules`
      asking what the team has captured.
    """
    capability = await _envelope(customer_id, _RETRIEVAL_KEY)
    if not capability.enabled:
        return QueryResponse(capability=capability)

    situation_id: UUID | None = None
    classification_out: ClassificationOut | None = None

    if req.situation_slug:
        situation_id = await _situation_id_for_slug(customer_id, req.situation_slug)
        if situation_id is None:
            return QueryResponse(capability=capability)
    elif req.query:
        classification = await classify(customer_id, req.query)
        classification_out = _classification_out(classification)
        if classification.outcome is Outcome.NO_VOCABULARY:
            return QueryResponse(
                capability=capability,
                classification=classification_out,
                no_situations_configured=True,
            )
        if classification.outcome is not Outcome.MATCHED:
            return QueryResponse(capability=capability, classification=classification_out)
        situation_id = classification.situation_id

    served = await serve_clauses(
        customer_id,
        actor_ref=req.actor_ref,
        situation_id=situation_id,
        session_id=req.session_id,
        limit=req.limit,
    )

    return QueryResponse(
        capability=capability,
        clauses=[_clause_out(c) for c in served.clauses],
        serve_id=served.serve_id,
        classification=classification_out,
    )


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


async def _envelope(customer_id: str, key: str) -> CapabilityOut:
    return CapabilityOut(**await capability_envelope(customer_id, key))  # type: ignore[arg-type]


async def _situation_id_for_slug(customer_id: str, slug: str) -> UUID | None:
    async with with_tenant(customer_id) as conn:
        return await conn.fetchval(
            "SELECT id FROM situations WHERE customer_id = $1 AND slug = $2",
            customer_id,
            slug,
        )


def _draft_out(draft: ClauseDraft) -> DraftOut:
    return DraftOut(
        kind=draft.kind,
        body=draft.body,
        semantic_action=draft.semantic_action,
        binding=draft.binding,
        scope=draft.scope,
    )


def _classification_out(classification: Any) -> ClassificationOut:
    return ClassificationOut(
        outcome=str(classification.outcome),
        slug=classification.slug,
        situation_id=classification.situation_id,
        confidence=classification.confidence,
        method=classification.method,
        runner_up=classification.runner_up,
    )


def _neighbour_out(neighbour: Neighbour) -> NeighbourOut:
    return NeighbourOut(
        clause_id=neighbour.clause_id,
        body=neighbour.body,
        status=neighbour.status,
        author_ref=neighbour.author_ref,
        similarity=neighbour.similarity,
        same_binding=neighbour.same_binding,
    )


def _clause_out(clause: ServedClause) -> ClauseOut:
    return ClauseOut(
        id=clause.id,
        kind=clause.kind,
        body=clause.body,
        status=clause.status,
        author_ref=clause.author_ref,
        version=clause.version,
        binding=clause.binding,
        scope=clause.scope,
        shared_by=clause.shared_by,
        human_backers=clause.human_backers,
        from_fallback=clause.from_fallback,
    )


def _bad_gateway(detail: str) -> HTTPException:
    return HTTPException(status_code=502, detail=detail)


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


async def is_declared_input_enabled(customer_id: str) -> bool:
    """Exported for callers that need the gate without an HTTP round trip."""
    return await is_capability_enabled(customer_id, _DECLARED_KEY)

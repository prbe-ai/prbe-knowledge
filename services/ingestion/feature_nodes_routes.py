"""Internal endpoint: POST /internal/feature-nodes/upsert

Mints a FEATURE GraphNode + its standard edge fan-out (OWNS → PR
Document, AUTHORED → Person, DOCUMENTS → evidence Documents,
TOUCHES → Repo) in one atomic write inside `with_tenant()`. Used
by the new Applications Plane (prbe-apps) when a PR with an
approved rationale merges — see memory project-apps-plane-approved.

Trust boundary: X-Internal-Knowledge-Key (same gate as
admin_routes.py). Tenant scope via X-Prbe-Customer.

All edges land with confidence='EXTRACTED' because the why text was
human-approved on the PR — that's the strongest provenance signal
in our model. The Phase-1a entity-clusters alias-resolution layer
(PR #265) automatically rewrites edge endpoints if any canonical_id
in the input has been aliased into a primary, so callers don't have
to know about merges.

NodeLabel.FEATURE already exists in shared/constants.py — this
endpoint does NOT add new label kinds. Edge types are
OWNS/AUTHORED/DOCUMENTS/TOUCHES/DESCRIBES, all pre-existing.

Canonical_id format (chosen by caller, recorded on the row):
  feature:gh:{owner}/{repo}#{pr_number}

PR-scoped initially; future manual entity-cluster merges can
collapse multiple FEATURE nodes that represent the same product
feature into one primary.

PR-side endpoints (race-prone — STUB-upsert).
``pull_request closed && merged=true`` fans out from prbe-backend
to BOTH knowledge ingest AND the apps plane in parallel. Apps plane
finalize lands first (one HTTP call, no chunking) — the PR Document
/ Person / Repo nodes that handlers/github.py creates for the PR
webhook may not exist yet when we attach edges. So we stub-upsert
every PR-side structural endpoint ourselves alongside FEATURE; the
shallow JSONB merge from
``ON CONFLICT DO UPDATE SET properties = graph_nodes.properties ||
EXCLUDED.properties`` means our minimal stubs survive idempotently
when the heavy ingest writes the same canonical_id later.

Evidence Documents (lookup-only).
The Slack threads / Notion pages / Linear tickets / etc. cited by
the rationale are sourced from retrieval results that ran inside
this tenant's RLS scope, so the cited docs almost always exist in
``graph_nodes`` by the time the PR is approved + merged. We don't
stub them: stubbing would require trusting the canonical_id's
embedded customer-id segment for sources like ``custom_ingest:`` /
``claude_code:`` / ``manual_upload:``, and the late-webhook miss
rate for genuinely external sources (slack/notion/linear) is
negligible relative to the data-quality cost of creating
phantom rows. Edges to missing evidence drop silently (with a
warning log) — the writer's standard "missing endpoint → skip"
semantic.

Rationale Document (search-surfacing).
The approved ``why`` text is also persisted as a standalone Document
(doc_type=FEATURE_RATIONALE, source_system=GITHUB) via the standard
typed-writeback path (``Normalizer.persist_single_document``) so the
rationale text lands in BM25 + vector indexes — searchable by any
phrase from the why. The Document is connected via
``rationaleDoc --DESCRIBES--> FEATURE`` (EdgeType.DESCRIBES,
confidence=EXTRACTED). The DESCRIBES edge resolves because the main
with_tenant block stub-upserts the rationale's graph_nodes row in
the same transaction as the edge (persist_single_document writes
documents+chunks only, not graph_nodes).

Sequencing (three transactions, in order):
  1. Pre-stub the rationale's DOCUMENT graph_nodes row. This is its
     own short with_tenant block. Required because step (2) below
     enqueues inferred_edges_queue, which the side-worker can claim
     immediately — and the worker's 1-hop graph-walk anchors on
     graph_nodes. Without a pre-stubbed anchor, the worker builds an
     empty bundle, the extractor runs on degraded context, the queue
     row is marked done, and content_hash idempotency means
     same-content retries never re-enqueue. Lost inferred edges
     become permanent.
  2. ``persist_single_document`` commits the Document + chunks +
     embedding in its OWN transaction, then enqueues
     wiki_synthesis_queue + inferred_edges_queue.
  3. Main with_tenant block: upserts FEATURE + PR/Repo/Person stubs +
     (idempotent re-upsert of the rationale's stub from step 1) +
     all edges including DESCRIBES.

Atomicity: three separate transactions. If step (3) fails after
steps (1) and (2) committed, a searchable rationale Doc exists with
no FEATURE pointing at it. Self-healing on retry: deterministic
canonical_ids + content_hash idempotency mean a redelivered webhook
re-runs the full flow cleanly (the graph_nodes stub re-upserts
idempotently, the Document no-ops on matching content_hash, the
FEATURE + edges retry succeeds). Same eventual-consistency story as
wiki_synthesis_queue + inferred_edges_queue, which by design fire
AFTER doc commit.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import UTC, datetime

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import AwareDatetime, BaseModel, Field, field_validator

from services.ingestion.chunker import count_tokens
from services.ingestion.graph_writer import upsert_edges, upsert_nodes
from shared.config import get_settings
from shared.constants import (
    DocClass,
    DocType,
    DocumentKind,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    make_document,
)

log = get_logger(__name__)

router = APIRouter(prefix="/internal/feature-nodes", tags=["feature-nodes"])


# Match the PR number out of `https://github.com/<owner>/<repo>/pull/<N>`
# regardless of trailing slash, query string, or `#issuecomment-...`
# fragment. A bare ``rsplit('/', 1)[-1]`` would silently produce
# ``""`` on a trailing slash and never collide with the GitHub webhook
# ingest's canonical_id — leaving an orphan FEATURE → phantom-doc edge
# pointing at nothing forever.
_PR_NUMBER_RE = re.compile(r"/pull/(\d+)")

# Cap evidence list length defensively. A rationale-LLM hallucination
# that emits hundreds of citations shouldn't be able to fan out into
# hundreds of edge upserts in one transaction.
MAX_EVIDENCE_DOC_IDS = 100

# Cap rationale body length. Bullet-point rationales from prbe-apps are
# typically 1-5KB; 8K is a generous ceiling that keeps the chunker
# bounded (one or two 512-token windows) and the inline embedding
# round-trip well within the apps-plane finalize webhook's 60s budget.
RATIONALE_WHY_MAX_LENGTH = 8_000


def _verify_internal_key(request: Request) -> None:
    """Constant-time check of X-Internal-Knowledge-Key. Matches the
    gate already on admin_routes.py + entity_clusters_routes.py."""
    # Settings field is SecretStr | None — unwrap before hmac.compare_digest
    # (which requires both args be the same type, str or bytes).
    expected_secret = get_settings().internal_knowledge_api_key
    expected = expected_secret.get_secret_value() if expected_secret else ""
    provided = request.headers.get("X-Internal-Knowledge-Key", "")
    if not expected or not hmac.compare_digest(expected, provided):
        raise HTTPException(
            status_code=401,
            detail="disabled — set INTERNAL_KNOWLEDGE_API_KEY",
        )


def _verify_customer_header(x_prbe_customer: str | None) -> str:
    if not x_prbe_customer:
        raise HTTPException(
            status_code=400, detail="missing X-Prbe-Customer header"
        )
    return x_prbe_customer


def _extract_pr_number(source_pr_url: str) -> str:
    m = _PR_NUMBER_RE.search(source_pr_url)
    if m is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"source_pr_url must contain /pull/<number>: "
                f"{source_pr_url!r}"
            ),
        )
    return m.group(1)


def _build_rationale_document(
    *,
    customer_id: str,
    rationale_doc_id: str,
    pr_doc_id: str,
    body: FeatureNodeUpsertRequest,
) -> Document:
    """Shape a FEATURE_RATIONALE Document for the typed-writeback path.

    Mirrors services/ingestion/investigation_writeback_routes.py for
    consistency with the other agent-artifact writeback path.
    content_hash is keyed on (canonical_id, why) so /probe regenerate
    bumps version + re-chunks, while a redelivered webhook with the
    same approved why no-ops cleanly inside _upsert_document.
    """
    now = datetime.now(UTC)
    why = body.why
    content_hash = hashlib.sha256(
        f"{body.canonical_id}|{why}".encode()
    ).hexdigest()
    acl = ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=customer_id,
                permission=Permission.READ,
            ),
        ],
        captured_at=now,
    )
    return Document(
        doc_id=rationale_doc_id,
        customer_id=customer_id,
        source_system=SourceSystem.GITHUB,
        source_id=rationale_doc_id,
        source_url=body.source_pr_url,
        doc_class=DocClass.AGENT_ARTIFACT,
        doc_type=DocType.FEATURE_RATIONALE,
        content_type="text/markdown",
        content_hash=content_hash,
        title=body.title,
        body=why,
        body_preview=why[:280],
        body_size_bytes=len(why.encode("utf-8")),
        body_token_count=count_tokens(why),
        parent_doc_id=pr_doc_id,
        created_at=body.merged_at,
        updated_at=now,
        valid_from=body.merged_at,
        ingested_at=now,
        acl=acl,
        metadata={
            "feature_canonical_id": body.canonical_id,
            "merge_sha": body.merge_sha,
            "merged_at": body.merged_at.isoformat(),
        },
    )


class FeatureNodeUpsertRequest(BaseModel):
    canonical_id: str = Field(
        ...,
        description="Apps plane chooses this; format `feature:gh:<owner>/<repo>#<n>`.",
    )
    title: str = Field(..., description="PR title at merge time.")
    why: str = Field(
        ...,
        min_length=1,
        max_length=RATIONALE_WHY_MAX_LENGTH,
        description=(
            "Approved rationale text. The final_why if edited, otherwise "
            "the proposed_why. Persisted as a FEATURE_RATIONALE Document "
            "via Normalizer.persist_single_document so the text lands in "
            f"search indexes. Capped at {RATIONALE_WHY_MAX_LENGTH} chars."
        ),
    )
    source_pr_url: str = Field(
        ..., description="https://github.com/<owner>/<repo>/pull/<n>"
    )
    # AwareDatetime: Pydantic rejects naive datetime strings. Without
    # this, a caller that sends "2026-05-20T00:00:00" (no tz) would land
    # in a TIMESTAMPTZ column with an asyncpg coercion that may either
    # 500 or silently shift to local time. Apps-plane already serializes
    # ISO 8601 with "+00:00" so this is a defensive guard.
    merged_at: AwareDatetime
    merge_sha: str

    @field_validator("why")
    @classmethod
    def _why_not_blank(cls, v: str) -> str:
        # min_length=1 above accepts "   \n  ". Reject whitespace-only
        # rationales — they'd land as approved + searchable + queued
        # docs with no useful text.
        if not v.strip():
            raise ValueError("why must contain non-whitespace text")
        return v
    evidence_doc_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_DOC_IDS,
        description=(
            "doc_ids of Documents the rationale cited (Slack threads, "
            "Notion pages, Linear tickets, etc.). The shape matches "
            "shared.models.Document.doc_id (`<source>:<...>`). "
            f"Capped at {MAX_EVIDENCE_DOC_IDS} entries."
        ),
    )
    author_id: str | None = Field(
        default=None,
        description=(
            "Person canonical_id of the PR author, if resolved. Apps plane "
            "may pass None — we still create the FEATURE node + non-AUTHOR "
            "edges; AUTHORED edge is skipped."
        ),
    )
    repo_full_name: str = Field(
        ..., description="`<owner>/<repo>` — used to derive the Repo edge endpoint."
    )


class FeatureNodeUpsertResponse(BaseModel):
    canonical_id: str
    node_id: int | None = Field(
        default=None,
        description="DB node_id of the upserted FEATURE row; informational.",
    )
    edges_created: int


@router.post("/upsert", response_model=FeatureNodeUpsertResponse)
async def upsert_feature_node(
    body: FeatureNodeUpsertRequest,
    request: Request,
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> FeatureNodeUpsertResponse:
    _verify_internal_key(request)
    customer_id = _verify_customer_header(x_prbe_customer)

    pr_number = _extract_pr_number(body.source_pr_url)
    pr_doc_id = f"github:{body.repo_full_name}:pr:{pr_number}"
    # Sibling-of-PR-Doc shape so dashboard / doc_type_resolver filters
    # that scope by source_system='github' include the rationale.
    rationale_doc_id = (
        f"github:{body.repo_full_name}:feature_rationale:{pr_number}"
    )

    # Order-preserving dedupe — if the LLM cites the same Slack thread
    # twice, we don't want two FEATURE→DOCUMENTS edges or a `[doc, doc]`
    # array sitting on the FEATURE node forever.
    evidence_doc_ids = list(dict.fromkeys(body.evidence_doc_ids))

    # ---- Pre-stub the rationale Doc's graph_nodes row. persist_single_document
    # below enqueues inferred_edges_queue right after it commits the Document;
    # the side-worker can claim that row IMMEDIATELY and run a 1-hop graph
    # walk anchored on the rationale's doc_id. If the graph_nodes anchor
    # isn't already present, the bundle builder finds zero graph neighbors,
    # the extractor runs on a degraded bundle, and the queue row is marked
    # done — same-content retries no-op via content_hash idempotency, so
    # the lost inferred edges are permanent. Writing the stub first means
    # the worker sees a proper anchor and produces a full bundle.
    rationale_node_stub = GraphNodeSpec(
        label=NodeLabel.DOCUMENT,
        canonical_id=rationale_doc_id,
        properties={"doc_type": DocType.FEATURE_RATIONALE.value},
    )
    async with with_tenant(customer_id) as conn:
        await upsert_nodes(
            conn,
            nodes=[rationale_node_stub],
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB.value,
        )

    # ---- Persist the rationale Document via the standard typed-writeback
    # path so its text is chunked + embedded + BM25-indexed. Commits in
    # its OWN transaction (separate from the pre-stub above AND the
    # FEATURE+edges block below). See module docstring on atomicity +
    # retry semantics.
    rationale_doc = _build_rationale_document(
        customer_id=customer_id,
        rationale_doc_id=rationale_doc_id,
        pr_doc_id=pr_doc_id,
        body=body,
    )
    await request.app.state.normalizer.persist_single_document(
        customer_id, rationale_doc
    )

    # PR-side structural endpoints stub-upsert with source_system=GITHUB.
    # ON CONFLICT shallow-JSONB-merge means the parallel GitHub webhook
    # ingest enriches these rows in place rather than creating duplicates.
    github_nodes: list[GraphNodeSpec] = [
        GraphNodeSpec(
            label=NodeLabel.FEATURE,
            canonical_id=body.canonical_id,
            properties={
                "title": body.title,
                "why": body.why,
                "source_pr_url": body.source_pr_url,
                "merged_at": body.merged_at.isoformat(),
                "merge_sha": body.merge_sha,
                "evidence_doc_ids": evidence_doc_ids,
            },
        ),
        GraphNodeSpec(
            label=NodeLabel.DOCUMENT,
            canonical_id=pr_doc_id,
            properties={"doc_type": DocType.GITHUB_PULL_REQUEST.value},
        ),
        # Stub the rationale Doc's graph_nodes row so the DESCRIBES edge
        # below resolves. persist_single_document writes documents +
        # chunks rows but NOT graph_nodes — without this stub the edge
        # would silently drop via the missing-endpoint behavior in
        # graph_writer.upsert_edges. Shallow JSONB merge means this
        # coexists idempotently with whatever properties the documents
        # row carries; the graph_nodes properties just record the
        # doc_type for graph-side filters.
        GraphNodeSpec(
            label=NodeLabel.DOCUMENT,
            canonical_id=rationale_doc_id,
            properties={"doc_type": DocType.FEATURE_RATIONALE.value},
        ),
        make_document(
            canonical_id=body.repo_full_name,
            kind=DocumentKind.REPO,
        ),
    ]
    if body.author_id:
        github_nodes.append(
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=body.author_id,
                properties={"source_system": SourceSystem.GITHUB.value},
            )
        )

    edges: list[GraphEdgeSpec] = [
        GraphEdgeSpec(
            from_label=NodeLabel.FEATURE,
            from_canonical_id=body.canonical_id,
            edge_type=EdgeType.OWNS,
            to_label=NodeLabel.DOCUMENT,
            to_canonical_id=pr_doc_id,
            confidence="EXTRACTED",
        ),
        GraphEdgeSpec(
            from_label=NodeLabel.FEATURE,
            from_canonical_id=body.canonical_id,
            edge_type=EdgeType.TOUCHES,
            to_label=NodeLabel.DOCUMENT,
            to_canonical_id=body.repo_full_name,
            confidence="EXTRACTED",
        ),
        # Rationale Doc describes the FEATURE. Direction follows the
        # natural-language reading ("the rationale describes the
        # feature") and keeps EdgeType.DOCUMENTS reserved for the
        # FEATURE→evidence-doc semantic (the rationale CITES evidence;
        # it doesn't describe it).
        GraphEdgeSpec(
            from_label=NodeLabel.DOCUMENT,
            from_canonical_id=rationale_doc_id,
            edge_type=EdgeType.DESCRIBES,
            to_label=NodeLabel.FEATURE,
            to_canonical_id=body.canonical_id,
            confidence="EXTRACTED",
        ),
    ]
    if body.author_id:
        edges.append(
            GraphEdgeSpec(
                from_label=NodeLabel.FEATURE,
                from_canonical_id=body.canonical_id,
                edge_type=EdgeType.AUTHORED,
                to_label=NodeLabel.PERSON,
                to_canonical_id=body.author_id,
                confidence="EXTRACTED",
            )
        )
    for doc_id in evidence_doc_ids:
        edges.append(
            GraphEdgeSpec(
                from_label=NodeLabel.FEATURE,
                from_canonical_id=body.canonical_id,
                edge_type=EdgeType.DOCUMENTS,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                confidence="EXTRACTED",
            )
        )

    # with_tenant acquires its own conn + transaction and sets the RLS
    # GUC. PR-side stubs + evidence-doc lookup + edges all commit on the
    # same connection in one transaction.
    async with with_tenant(customer_id) as conn:
        node_ids = await upsert_nodes(
            conn,
            nodes=github_nodes,
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB.value,
        )

        # Lookup-only for evidence docs. Run the query inside this same
        # with_tenant block so RLS scopes the result set to this tenant —
        # which means a canonical_id from another tenant's namespace can
        # never satisfy the lookup, even if the caller forwarded a bogus
        # one. The standard "missing endpoint → silent edge skip"
        # behavior in graph_writer.upsert_edges does the rest.
        if evidence_doc_ids:
            rows = await conn.fetch(
                """
                SELECT canonical_id, node_id FROM graph_nodes
                WHERE customer_id = $1
                  AND label = 'Document'
                  AND canonical_id = ANY($2::text[])
                """,
                customer_id,
                evidence_doc_ids,
            )
            found_ids = {r["canonical_id"] for r in rows}
            for r in rows:
                node_ids[(NodeLabel.DOCUMENT.value, r["canonical_id"])] = (
                    r["node_id"]
                )
            missing = [d for d in evidence_doc_ids if d not in found_ids]
            if missing:
                log.warning(
                    "feature_node.evidence_docs_not_found",
                    customer_id=customer_id,
                    canonical_id=body.canonical_id,
                    missing_doc_ids=missing,
                )

        edges_created = await upsert_edges(
            conn,
            edges=edges,
            node_ids=node_ids,
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB.value,
        )

    feature_node_id = node_ids.get(
        (NodeLabel.FEATURE.value, body.canonical_id)
    )
    log.info(
        "feature_node.upserted",
        customer_id=customer_id,
        canonical_id=body.canonical_id,
        rationale_doc_id=rationale_doc_id,
        edges_created=edges_created,
        evidence_doc_count=len(evidence_doc_ids),
    )
    return FeatureNodeUpsertResponse(
        canonical_id=body.canonical_id,
        node_id=feature_node_id,
        edges_created=edges_created,
    )

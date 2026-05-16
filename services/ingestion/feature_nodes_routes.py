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
OWNS/AUTHORED/DOCUMENTS/TOUCHES, all pre-existing.

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
"""

from __future__ import annotations

import hmac
import re
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from services.ingestion.graph_writer import upsert_edges, upsert_nodes
from shared.config import get_settings
from shared.constants import DocType, EdgeType, NodeLabel, SourceSystem
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import GraphEdgeSpec, GraphNodeSpec

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


def _verify_internal_key(request: Request) -> None:
    """Constant-time check of X-Internal-Knowledge-Key. Matches the
    gate already on admin_routes.py + entity_clusters_routes.py."""
    expected = get_settings().internal_knowledge_api_key or ""
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


class FeatureNodeUpsertRequest(BaseModel):
    canonical_id: str = Field(
        ...,
        description="Apps plane chooses this; format `feature:gh:<owner>/<repo>#<n>`.",
    )
    title: str = Field(..., description="PR title at merge time.")
    why: str = Field(
        ...,
        description=(
            "Approved rationale text. The final_why if edited, otherwise "
            "the proposed_why."
        ),
    )
    source_pr_url: str = Field(
        ..., description="https://github.com/<owner>/<repo>/pull/<n>"
    )
    merged_at: datetime
    merge_sha: str
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

    # Order-preserving dedupe — if the LLM cites the same Slack thread
    # twice, we don't want two FEATURE→DOCUMENTS edges or a `[doc, doc]`
    # array sitting on the FEATURE node forever.
    evidence_doc_ids = list(dict.fromkeys(body.evidence_doc_ids))

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
        GraphNodeSpec(
            label=NodeLabel.REPO,
            canonical_id=body.repo_full_name,
            properties={},
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
            to_label=NodeLabel.REPO,
            to_canonical_id=body.repo_full_name,
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
        edges_created=edges_created,
        evidence_doc_count=len(evidence_doc_ids),
    )
    return FeatureNodeUpsertResponse(
        canonical_id=body.canonical_id,
        node_id=feature_node_id,
        edges_created=edges_created,
    )

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

NodeLabel.FEATURE already exists in shared/constants.py:127 — this
endpoint does NOT add new label kinds. Edge types are
OWNS/AUTHORED/DOCUMENTS/TOUCHES, all pre-existing.

Canonical_id format (chosen by caller, recorded on the row):
  feature:gh:{owner}/{repo}#{pr_number}

PR-scoped initially; future manual entity-cluster merges (the same
endpoint surface from PR #266) can collapse multiple FEATURE nodes
that represent the same product feature into one primary.
"""

from __future__ import annotations

import hmac
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from shared.config import get_settings
from shared.constants import EdgeConfidence, EdgeType, NodeLabel, SourceSystem
from shared.db import raw_conn, with_tenant
from shared.logging import get_logger
from shared.models import GraphEdgeSpec, GraphNodeSpec

log = get_logger(__name__)

router = APIRouter(prefix="/internal/feature-nodes", tags=["feature-nodes"])


def _verify_internal_key(request: Request) -> None:
    """Constant-time check of X-Internal-Knowledge-Key. Matches the
    gate already on admin_routes.py + entity_clusters_routes.py."""
    settings = get_settings()
    expected = getattr(settings, "internal_knowledge_key", None) or ""
    provided = request.headers.get("X-Internal-Knowledge-Key", "")
    if not expected or not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="invalid_internal_key")


def _verify_customer_header(x_prbe_customer: str | None) -> str:
    if not x_prbe_customer:
        raise HTTPException(
            status_code=400, detail="missing X-Prbe-Customer header"
        )
    return x_prbe_customer


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
        description=(
            "doc_ids of Documents the rationale cited (Slack threads, "
            "Notion pages, Linear tickets, etc.). The shape matches "
            "shared.models.Document.doc_id (`<source>:<...>`)."
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
        description="DB node_id when graph_writer returns one; informational.",
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

    # Build the NormalizationResult-equivalent fan-out by hand — we only
    # need one node + a handful of edges; the full normalizer machinery
    # is overkill (it batches across many documents).
    feature_node = GraphNodeSpec(
        label=NodeLabel.FEATURE,
        canonical_id=body.canonical_id,
        properties={
            "title": body.title,
            "why": body.why,
            "source_pr_url": body.source_pr_url,
            "merged_at": body.merged_at.isoformat(),
            "merge_sha": body.merge_sha,
            "evidence_doc_ids": body.evidence_doc_ids,
        },
    )

    edges: list[GraphEdgeSpec] = []

    # FEATURE -[OWNS]-> PR Document. Doc_id format matches the existing
    # GitHub PR doc_id convention (see prbe-knowledge feedback memory
    # documents_source_id_format).
    pr_doc_id = (
        f"github:{body.repo_full_name}:pr:"
        f"{body.source_pr_url.rsplit('/', 1)[-1]}"
    )
    edges.append(
        GraphEdgeSpec(
            from_label=NodeLabel.FEATURE,
            from_canonical_id=body.canonical_id,
            edge_type=EdgeType.OWNS,
            to_label=NodeLabel.DOCUMENT,
            to_canonical_id=pr_doc_id,
            confidence=EdgeConfidence.EXTRACTED,
        )
    )

    if body.author_id:
        edges.append(
            GraphEdgeSpec(
                from_label=NodeLabel.FEATURE,
                from_canonical_id=body.canonical_id,
                edge_type=EdgeType.AUTHORED,
                to_label=NodeLabel.PERSON,
                to_canonical_id=body.author_id,
                confidence=EdgeConfidence.EXTRACTED,
            )
        )

    for doc_id in body.evidence_doc_ids:
        edges.append(
            GraphEdgeSpec(
                from_label=NodeLabel.FEATURE,
                from_canonical_id=body.canonical_id,
                edge_type=EdgeType.DOCUMENTS,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                confidence=EdgeConfidence.EXTRACTED,
            )
        )

    edges.append(
        GraphEdgeSpec(
            from_label=NodeLabel.FEATURE,
            from_canonical_id=body.canonical_id,
            edge_type=EdgeType.TOUCHES,
            to_label=NodeLabel.REPO,
            to_canonical_id=body.repo_full_name,
            confidence=EdgeConfidence.EXTRACTED,
        )
    )

    # Persist inside with_tenant() so RLS GUC fires + entity-clusters
    # alias resolution (PR #265 / shared/graph_writer.upsert_*) rewrites
    # any aliased endpoints transparently.
    async with raw_conn() as conn, with_tenant(conn, customer_id):
        from shared.graph_writer import upsert_edges, upsert_nodes  # local import

        node_ids = await upsert_nodes(
            conn,
            nodes=[feature_node],
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB,
        )
        await upsert_edges(
            conn,
            edges=edges,
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB,
        )

    node_id: int | None = None
    if isinstance(node_ids, dict):
        node_id = node_ids.get((NodeLabel.FEATURE, body.canonical_id))
    elif isinstance(node_ids, list) and node_ids:
        # graph_writer.upsert_nodes returns ids in input order in some
        # variants of the API; accept either shape.
        node_id = node_ids[0] if isinstance(node_ids[0], int) else None

    log.info(
        "feature_node.upserted",
        customer_id=customer_id,
        canonical_id=body.canonical_id,
        edges=len(edges),
    )
    return FeatureNodeUpsertResponse(
        canonical_id=body.canonical_id,
        node_id=node_id,
        edges_created=len(edges),
    )

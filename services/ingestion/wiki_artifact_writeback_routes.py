"""POST /api/wiki-artifacts writeback route.

The orchestrator's Pass 2 wiki authors (postmortem / knowledge_page /
correction) call this after producing a draft body. The route:

1. Validates kind/target consistency (Python check ahead of the DB
   constraint so the failure is a typed 422 with a clear message).
2. For corrections, validates that ``target_doc_id`` exists and is
   readable (``visibility='approved'``) for the customer.
3. Computes a stable ``artifact_doc_id`` from the incident source +
   artifact kind + version (+ target-hash for corrections so multiple
   corrections from the same incident targeting different docs don't
   collide on the doc_id).
4. Idempotency probe: an existing documents row for the same doc_id
   short-circuits to a duplicate=True response.
5. Builds the Document with ``visibility=DRAFT``, persists via
   ``Normalizer.persist_single_document``, then upserts the
   wiki_review_queue row at ``pending_review`` (or
   ``failed_pending_review`` for stub-mode artifacts).

Mirrors investigation_writeback_routes.py's style: bypasses the
ingestion queue, routes through the same Normalizer for retrieval
parity, idempotent on (customer_id, doc_id), local import of
``_verify_internal_key`` to avoid a circular with main.py.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request

from services.ingestion.chunker import count_tokens
from services.post_approval.wiki_review_state import (
    _validate_kind_target_consistency,
    get_detail,
    upsert_pending_review,
)
from shared.constants import (
    DocClass,
    DocType,
    Permission,
    PrincipalType,
    SourceSystem,
    Visibility,
)
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import ACLPrincipal, ACLSnapshot, Document
from shared.schemas.wiki_artifact import (
    ArtifactState,
    WikiArtifactWritebackRequest,
    WikiArtifactWritebackResponse,
)

log = get_logger(__name__)

router = APIRouter(
    prefix="/api/wiki-artifacts",
    tags=["wiki-artifacts"],
)


_KIND_TO_DOC_TYPE: dict[str, str] = {
    "postmortem": DocType.WIKI_POSTMORTEM,
    "knowledge_page": DocType.WIKI_KNOWLEDGE_PAGE,
    "correction": DocType.WIKI_CORRECTION,
}


def _source_prefix(incident_doc_id: str) -> str:
    """Pull the source-system prefix off the Plan 1 incident_doc_id.

    PD writes ``pd:incident:...``; incident.io writes ``iio:incident:...``.
    Anything else is a caller bug and surfaces as a 422 upstream.
    """
    if incident_doc_id.startswith("pd:"):
        return "pd"
    if incident_doc_id.startswith("iio:"):
        return "iio"
    raise HTTPException(
        status_code=422,
        detail=(
            f"incident_doc_id '{incident_doc_id}' must start with 'pd:' or 'iio:'"
        ),
    )


def _incident_suffix(incident_doc_id: str) -> str:
    """The provider-side incident id (last colon-segment of the Plan 1 doc_id)."""
    if ":" not in incident_doc_id:
        return incident_doc_id
    return incident_doc_id.split(":")[-1]


def _next_version(prior_artifact_doc_id: str | None) -> int:
    """Parse the ``:vN`` suffix off the prior artifact_doc_id (if any).

    Returns 1 when ``prior_artifact_doc_id`` is None. The writeback
    caller (orchestrator) re-uses the same artifact_kind across re-runs
    on reject, so incrementing on top of the prior version is the
    correct lineage.

    Raises ``ValueError`` when ``prior_artifact_doc_id`` is supplied
    but malformed (no ``:vN`` suffix, or ``N`` is not an int). Silently
    returning 1 in those cases would collide with the original v1
    artifact_doc_id and let the idempotency probe return
    ``duplicate=True`` against the wrong row — the writeback route
    catches this and surfaces it as 422 so the orchestrator fails loud.
    """
    if not prior_artifact_doc_id:
        return 1
    tail = prior_artifact_doc_id.rsplit(":v", 1)
    if len(tail) != 2:
        raise ValueError(
            f"prior_artifact_doc_id '{prior_artifact_doc_id}' "
            "is malformed (missing ':vN' suffix)"
        )
    try:
        return int(tail[1]) + 1
    except ValueError as exc:
        raise ValueError(
            f"prior_artifact_doc_id '{prior_artifact_doc_id}' "
            "has non-integer version suffix"
        ) from exc


def _compose_artifact_doc_id(
    *,
    incident_doc_id: str,
    artifact_kind: str,
    target_doc_id: str | None,
    version: int,
) -> str:
    """Build the stable artifact_doc_id.

    Examples:
    - pd:wiki.postmortem:PD-INC-001:v1
    - iio:wiki.knowledge_page:01ABC...:v1
    - pd:wiki.correction:PD-INC-001:abc12345:v1
      (where abc12345 = sha256(target_doc_id)[:8])

    The target-hash segment for corrections lets the orchestrator emit
    multiple correction drafts from the same incident — one per
    targeted doc — without colliding on the artifact_doc_id. Eight
    hex chars (32 bits) is enough for the orders-of-magnitude few
    targets per incident; collision probability is dominated by the
    incident grain anyway.
    """
    prefix = _source_prefix(incident_doc_id)
    incident_id = _incident_suffix(incident_doc_id)
    doc_type = _KIND_TO_DOC_TYPE[artifact_kind]
    if artifact_kind == "correction" and target_doc_id is not None:
        target_hash = hashlib.sha256(
            target_doc_id.encode("utf-8")
        ).hexdigest()[:8]
        return f"{prefix}:{doc_type}:{incident_id}:{target_hash}:v{version}"
    return f"{prefix}:{doc_type}:{incident_id}:v{version}"


def _content_hash(payload: WikiArtifactWritebackRequest) -> str:
    parts = (
        payload.incident_doc_id,
        payload.artifact_kind,
        payload.target_doc_id or "",
        payload.body_markdown,
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _parent_doc_id_for_kind(
    artifact_kind: str,
    incident_doc_id: str,
    target_doc_id: str | None,
) -> str | None:
    """Parent-doc lineage:
    - postmortem -> the incident doc (the new artifact is a child
      synthesis of the incident).
    - correction -> the doc being corrected (so the wiki UI can show
      the correction inline on the target's page).
    - knowledge_page -> standalone, no parent.
    """
    if artifact_kind == "postmortem":
        return incident_doc_id
    if artifact_kind == "correction":
        return target_doc_id
    return None


async def _target_doc_is_readable(
    customer_id: str, target_doc_id: str,
) -> bool:
    """Validate a correction's target_doc_id is approved + live.

    Mirrors template_resolver.upsert_override's pre-write check —
    surfacing the misconfiguration at writeback time rather than at
    render time.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchval(
            "SELECT 1 FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 "
            "  AND valid_to IS NULL AND visibility = 'approved' "
            "LIMIT 1",
            customer_id, target_doc_id,
        )
    return row is not None


async def _existing_artifact_doc_id(customer_id: str, doc_id: str) -> bool:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM documents "
            "WHERE doc_id = $1 AND customer_id = $2 AND valid_to IS NULL",
            doc_id, customer_id,
        )
    return row is not None


@router.post("", response_model=WikiArtifactWritebackResponse)
async def writeback(
    payload: WikiArtifactWritebackRequest,
    request: Request,
) -> WikiArtifactWritebackResponse:
    # Trust boundary.
    from services.ingestion.main import _verify_internal_key
    _verify_internal_key(request)

    # Python-side kind/target consistency check — gives a typed 422
    # with a clear message rather than letting asyncpg surface the
    # check constraint violation deeper in the call.
    try:
        _validate_kind_target_consistency(
            payload.artifact_kind, payload.target_doc_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Correction-only: validate target readability up front. Catching
    # this here keeps the failure mode loud (the orchestrator gets a
    # typed 422 instead of writing a draft against a phantom target).
    if (
        payload.artifact_kind == "correction"
        and payload.target_doc_id is not None
        and not await _target_doc_is_readable(
            payload.customer_id, payload.target_doc_id,
        )
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"target_doc_id '{payload.target_doc_id}' does not exist "
                "or is not readable"
            ),
        )

    try:
        version = _next_version(payload.metadata.prior_artifact_doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    artifact_doc_id = _compose_artifact_doc_id(
        incident_doc_id=payload.incident_doc_id,
        artifact_kind=payload.artifact_kind,
        target_doc_id=payload.target_doc_id,
        version=version,
    )

    # Idempotency probe — if the document was already persisted (e.g.
    # the orchestrator retried a redelivery), surface the existing
    # queue row's state and short-circuit. The state-row read uses
    # ``get_detail`` so the response carries the lineage-aware state.
    if await _existing_artifact_doc_id(payload.customer_id, artifact_doc_id):
        detail = await get_detail(payload.customer_id, artifact_doc_id)
        # Fallback to pending_review if the queue row hasn't materialized
        # yet — narrow race between document write and queue upsert.
        current_state: ArtifactState = (
            detail.state if detail else "pending_review"
        )
        return WikiArtifactWritebackResponse(
            artifact_doc_id=artifact_doc_id,
            state=current_state,
            duplicate=True,
        )

    now = datetime.now(UTC)
    source_system = (
        SourceSystem.PAGERDUTY
        if payload.incident_doc_id.startswith("pd:")
        else SourceSystem.INCIDENT_IO
    )

    # Merge writeback metadata + the four routing keys the dashboard
    # / orchestrator round-trip will need. Pydantic's exclude_none keeps
    # the metadata jsonb tight (no { "reviewer_feedback": null } noise).
    merged_metadata = payload.metadata.model_dump(exclude_none=True)
    merged_metadata.update(
        {
            "incident_doc_id": payload.incident_doc_id,
            "investigation_doc_id": payload.investigation_doc_id,
            "artifact_kind": payload.artifact_kind,
            "target_doc_id": payload.target_doc_id,
        }
    )

    acl = ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=payload.customer_id,
                permission=Permission.READ,
            ),
        ],
        captured_at=now,
    )
    doc = Document(
        doc_id=artifact_doc_id,
        customer_id=payload.customer_id,
        source_system=source_system,
        source_id=artifact_doc_id,
        source_url="",
        doc_class=DocClass.AGENT_ARTIFACT,
        doc_type=_KIND_TO_DOC_TYPE[payload.artifact_kind],
        content_type="text/markdown",
        content_hash=_content_hash(payload),
        title=payload.title,
        body=payload.body_markdown,
        body_preview=payload.body_markdown[:280],
        body_size_bytes=len(payload.body_markdown.encode("utf-8")),
        body_token_count=count_tokens(payload.body_markdown),
        parent_doc_id=_parent_doc_id_for_kind(
            payload.artifact_kind,
            payload.incident_doc_id,
            payload.target_doc_id,
        ),
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=acl,
        metadata=merged_metadata,
        visibility=Visibility.DRAFT,
    )

    normalizer = request.app.state.normalizer
    await normalizer.persist_single_document(payload.customer_id, doc)

    # Stub-mode artifacts land in ``failed_pending_review`` so the
    # dashboard's review UI surfaces them distinctly from the happy-
    # path drafts.
    initial_state: ArtifactState = (
        "failed_pending_review"
        if payload.metadata.mode == "stub"
        else "pending_review"
    )

    detail = await upsert_pending_review(
        customer_id=payload.customer_id,
        artifact_doc_id=artifact_doc_id,
        incident_doc_id=payload.incident_doc_id,
        artifact_kind=payload.artifact_kind,
        target_doc_id=payload.target_doc_id,
        parent_artifact_doc_id=payload.metadata.prior_artifact_doc_id,
        initial_state=initial_state,
        metadata={
            "mode": payload.metadata.mode,
            "tool_trace_run_id": payload.metadata.tool_trace_run_id,
        },
    )

    log.info(
        "wiki_artifact.written",
        customer_id=payload.customer_id,
        artifact_doc_id=artifact_doc_id,
        artifact_kind=payload.artifact_kind,
        incident_doc_id=payload.incident_doc_id,
        mode=payload.metadata.mode,
        state=detail.state,
        duplicate=False,
    )

    return WikiArtifactWritebackResponse(
        artifact_doc_id=artifact_doc_id,
        state=detail.state,
        duplicate=False,
    )

"""Pydantic request/response schemas for the incident investigation surface.

These types are exchanged across the trust boundary between
prbe-orchestrator (which produces the investigation report) and
prbe-knowledge (which persists it as an INCIDENT_INVESTIGATION
document and tracks reviewer lifecycle). They are also used by the
dashboard BFF for the review endpoints.

The persistence side (Document write, ACL, embedding) lives in the
writeback route (Plan 2 Task 5). The reviewer lifecycle table is
`incident_investigations` (migration 0080).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.constants import SourceSystem


InvestigationMode = Literal["full", "playbook_only", "stub"]
InvestigationState = Literal[
    "pending_dispatch", "running", "pending_review",
    "approved", "rejected", "failed_pending_review",
]


class EvidenceSection(BaseModel):
    """One section of the investigation's structured evidence list.

    Stored in the report Document's `metadata.evidence` JSONB. The body
    markdown also renders these sections, but the LLM-facing surface is
    the human-readable body; structured access goes through metadata.
    """

    source: str
    query: str
    result_summary: str
    linked_doc_ids: list[str] = Field(default_factory=list)


class InvestigationWritebackRequest(BaseModel):
    """Payload from the orchestrator's investigation agent into knowledge.

    POST /api/incident-investigations. Idempotency on
    (customer_id, source_event_id, version) is enforced by the route.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    incident_doc_id: str
    source_system: SourceSystem
    source_event_id: str
    version: int = Field(ge=1)
    mode: InvestigationMode
    title: str = Field(min_length=1, max_length=512)
    body_markdown: str = Field(min_length=1)
    evidence: list[EvidenceSection] = Field(default_factory=list)
    narrative: str | None = None
    tool_trace_run_id: str | None = None
    prior_report_doc_id: str | None = None
    reviewer_feedback: str | None = None

    @field_validator("source_system")
    @classmethod
    def _source_must_be_incident_source(cls, v: SourceSystem) -> SourceSystem:
        if v not in (SourceSystem.PAGERDUTY, SourceSystem.INCIDENT_IO):
            raise ValueError(
                f"source_system must be PAGERDUTY or INCIDENT_IO; got {v.value}"
            )
        return v


class InvestigationWritebackResponse(BaseModel):
    report_doc_id: str
    state: InvestigationState
    duplicate: bool


class InvestigationVersionEntry(BaseModel):
    """One entry in `incident_investigations.versions` JSONB."""

    version: int
    doc_id: str
    mode: InvestigationMode
    created_at: datetime
    decision: Literal["approved", "rejected", "pending"] = "pending"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    feedback: str | None = None


class InvestigationDetail(BaseModel):
    """Row shape returned by GET /api/incident-investigations/{id}."""

    customer_id: str
    incident_doc_id: str
    current_report_doc_id: str | None
    state: InvestigationState
    versions: list[InvestigationVersionEntry]
    reviewer_id: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class InvestigationListItem(BaseModel):
    """Row shape returned by GET /api/incident-investigations (list)."""

    incident_doc_id: str
    current_report_doc_id: str | None
    state: InvestigationState
    updated_at: datetime


class RejectRequest(BaseModel):
    feedback: str = Field(min_length=1)
    reviewer_id: str


class ApproveRequest(BaseModel):
    reviewer_id: str

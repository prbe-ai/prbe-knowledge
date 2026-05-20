"""Pydantic models for wiki artifact writeback + review."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ArtifactKind = Literal["postmortem", "knowledge_page", "correction"]
ArtifactMode = Literal["full", "playbook_only", "stub"]
ArtifactState = Literal[
    "pending_writeback", "pending_review", "approved",
    "rejected", "failed_pending_review",
]


class WikiArtifactMetadata(BaseModel):
    mode: ArtifactMode
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""
    tool_trace_run_id: str | None = None
    prior_artifact_doc_id: str | None = None
    reviewer_feedback: str | None = None
    slot_data: dict[str, Any] | None = None  # postmortem-only: structured slot values
    target_excerpt: str | None = None  # correction-only: text being replaced


class WikiArtifactWritebackRequest(BaseModel):
    customer_id: str
    incident_doc_id: str
    investigation_doc_id: str
    artifact_kind: ArtifactKind
    target_doc_id: str | None = None  # required when artifact_kind='correction'
    title: str = Field(min_length=1, max_length=240)
    body_markdown: str
    metadata: WikiArtifactMetadata


class WikiArtifactWritebackResponse(BaseModel):
    artifact_doc_id: str
    state: ArtifactState
    duplicate: bool


class WikiArtifactListItem(BaseModel):
    artifact_doc_id: str
    incident_doc_id: str
    artifact_kind: ArtifactKind
    target_doc_id: str | None
    state: ArtifactState
    parent_artifact_doc_id: str | None
    updated_at: datetime


class WikiArtifactVersionEntry(BaseModel):
    artifact_doc_id: str
    decision: Literal["pending", "approved", "rejected"]
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    feedback: str | None = None
    created_at: datetime


class WikiArtifactDetail(BaseModel):
    customer_id: str
    artifact_doc_id: str
    incident_doc_id: str
    artifact_kind: ArtifactKind
    target_doc_id: str | None
    parent_artifact_doc_id: str | None
    state: ArtifactState
    versions: list[WikiArtifactVersionEntry]
    reviewer_id: str | None
    reviewed_at: datetime | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ApproveRequest(BaseModel):
    reviewer_id: str = Field(min_length=1)


class RejectRequest(BaseModel):
    reviewer_id: str = Field(min_length=1)
    feedback: str = Field(min_length=1)

"""Pydantic models for evidence pack writeback (Pass 1 -> knowledge)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    at: str  # ISO8601
    event: str
    source: str  # 'pd' | 'iio' | 'slack' | 'deploy' | 'alert' | etc.


class ResolutionAction(BaseModel):
    at: str
    actor: str | None = None
    action: str
    source: str


class RecoverySignal(BaseModel):
    at: str
    metric: str
    value: str
    source: str


class DeployRef(BaseModel):
    deploy_id: str
    deployed_at: str
    pr_url: str | None = None
    service: str | None = None


class EvidencePack(BaseModel):
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    resolution_actions: list[ResolutionAction] = Field(default_factory=list)
    recovery_signals: list[RecoverySignal] = Field(default_factory=list)
    post_resolution_discussion: list[str] = Field(default_factory=list)
    related_doc_ids: list[str] = Field(default_factory=list)
    deploys_in_window: list[DeployRef] = Field(default_factory=list)
    similar_past_incidents: list[str] = Field(default_factory=list)
    free_form_findings: str = ""
    mode: Literal["full", "seed_only", "empty"]


class EvidencePackWritebackRequest(BaseModel):
    customer_id: str
    incident_doc_id: str
    evidence_pack: EvidencePack


class EvidencePackWritebackResponse(BaseModel):
    incident_doc_id: str
    duplicate: bool

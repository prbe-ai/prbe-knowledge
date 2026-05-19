"""Pydantic models for per-customer postmortem template storage."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TemplateMode = Literal["inline", "doc_ref"]


class TemplateUpsertRequest(BaseModel):
    customer_id: str
    mode: TemplateMode
    body_markdown: str | None = None
    ref_doc_id: str | None = None

    @model_validator(mode="after")
    def _mode_consistency(self) -> TemplateUpsertRequest:
        if self.mode == "inline" and not self.body_markdown:
            raise ValueError("inline mode requires body_markdown")
        if self.mode == "doc_ref" and not self.ref_doc_id:
            raise ValueError("doc_ref mode requires ref_doc_id")
        if self.mode == "inline" and self.ref_doc_id:
            raise ValueError("ref_doc_id must be null in inline mode")
        if self.mode == "doc_ref" and self.body_markdown:
            raise ValueError("body_markdown must be null in doc_ref mode")
        return self


class TemplateRow(BaseModel):
    customer_id: str
    mode: TemplateMode
    body_markdown: str | None
    ref_doc_id: str | None
    updated_at: datetime


class TemplateEffectiveResponse(BaseModel):
    """The resolved template body the agent will use."""
    body_markdown: str = Field(min_length=1)
    source: Literal["inline_override", "doc_ref_override", "default"]
    resolved_ref_doc_id: str | None = None

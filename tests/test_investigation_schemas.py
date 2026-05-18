"""Tests for shared.investigation_schemas — Pydantic shape, defaults, validation."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from shared.constants import SourceSystem
from shared.investigation_schemas import (
    ApproveRequest,
    EvidenceSection,
    InvestigationDetail,
    InvestigationListItem,
    InvestigationVersionEntry,
    InvestigationWritebackRequest,
    InvestigationWritebackResponse,
    RejectRequest,
)


def test_evidence_section_default_linked_doc_ids_empty() -> None:
    e = EvidenceSection(source="knowledge", query="q", result_summary="r")
    assert e.linked_doc_ids == []


def test_writeback_request_rejects_version_zero() -> None:
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(
            customer_id="c", incident_doc_id="d", source_system=SourceSystem.PAGERDUTY,
            source_event_id="e", version=0, mode="full",
            title="t", body_markdown="b",
        )


def test_writeback_request_accepts_minimal_payload() -> None:
    req = InvestigationWritebackRequest(
        customer_id="c", incident_doc_id="d", source_system=SourceSystem.PAGERDUTY,
        source_event_id="e", version=1, mode="full",
        title="t", body_markdown="b",
    )
    assert req.evidence == []
    assert req.narrative is None
    assert req.prior_report_doc_id is None
    assert req.reviewer_feedback is None


def test_writeback_request_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(
            customer_id="c", incident_doc_id="d", source_system=SourceSystem.PAGERDUTY,
            source_event_id="e", version=1, mode="bogus",  # type: ignore[arg-type]
            title="t", body_markdown="b",
        )


def test_writeback_request_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(
            customer_id="c", incident_doc_id="d", source_system="opsgenie",  # type: ignore[arg-type]
            source_event_id="e", version=1, mode="full",
            title="t", body_markdown="b",
        )


def test_writeback_request_rejects_non_incident_source_system() -> None:
    # SLACK is a real SourceSystem value but not in the incident allowlist.
    with pytest.raises(ValidationError, match="PAGERDUTY or INCIDENT_IO"):
        InvestigationWritebackRequest(
            customer_id="c", incident_doc_id="d",
            source_system=SourceSystem.SLACK,
            source_event_id="e", version=1, mode="full",
            title="t", body_markdown="b",
        )


def test_writeback_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(
            customer_id="c", incident_doc_id="d",
            source_system=SourceSystem.PAGERDUTY,
            source_event_id="e", version=1, mode="full",
            title="t", body_markdown="b",
            extra_unexpected_field="boom",  # type: ignore[call-arg]
        )


def test_writeback_request_rejects_empty_title_or_body() -> None:
    base = {
        "customer_id": "c", "incident_doc_id": "d",
        "source_system": SourceSystem.PAGERDUTY,
        "source_event_id": "e", "version": 1, "mode": "full",
    }
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(**base, title="", body_markdown="b")
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(**base, title="t", body_markdown="")


def test_writeback_request_rejects_title_over_240_chars() -> None:
    base = {
        "customer_id": "c", "incident_doc_id": "d",
        "source_system": SourceSystem.PAGERDUTY,
        "source_event_id": "e", "version": 1, "mode": "full",
        "body_markdown": "b",
    }
    with pytest.raises(ValidationError):
        InvestigationWritebackRequest(**base, title="x" * 241)


def test_writeback_response_round_trip() -> None:
    resp = InvestigationWritebackResponse(
        report_doc_id="r", state="pending_review", duplicate=False,
    )
    assert resp.model_dump() == {
        "report_doc_id": "r", "state": "pending_review", "duplicate": False,
    }


def test_reject_request_requires_nonempty_feedback() -> None:
    with pytest.raises(ValidationError):
        RejectRequest(feedback="", reviewer_id="u")


def test_reject_request_accepts_nonempty_feedback() -> None:
    r = RejectRequest(feedback="missed the deploy", reviewer_id="u")
    assert r.feedback == "missed the deploy"


def test_approve_request_requires_reviewer_id() -> None:
    with pytest.raises(ValidationError) as exc:
        ApproveRequest()  # type: ignore[call-arg]
    assert any(err["loc"] == ("reviewer_id",) for err in exc.value.errors())


def test_detail_round_trip() -> None:
    now = datetime.now(UTC)
    d = InvestigationDetail(
        customer_id="c", incident_doc_id="d", current_report_doc_id=None,
        state="pending_review", versions=[], reviewer_id=None, reviewed_at=None,
        created_at=now, updated_at=now,
    )
    assert d.state == "pending_review"


def test_list_item_minimal() -> None:
    now = datetime.now(UTC)
    li = InvestigationListItem(
        incident_doc_id="d", current_report_doc_id=None,
        state="approved", updated_at=now,
    )
    assert li.state == "approved"


def test_version_entry_defaults() -> None:
    v = InvestigationVersionEntry(
        version=1, doc_id="d", mode="full", created_at=datetime.now(UTC),
    )
    assert v.decision == "pending"
    assert v.reviewed_by is None
    assert v.feedback is None


def test_writeback_request_rejects_body_over_256kib() -> None:
    base = {
        "customer_id": "c", "incident_doc_id": "d",
        "source_system": SourceSystem.PAGERDUTY,
        "source_event_id": "e", "version": 1, "mode": "full",
        "title": "t",
    }
    with pytest.raises(ValidationError):
        # 262145 chars > 256 KiB
        InvestigationWritebackRequest(**base, body_markdown="x" * 262145)


def test_writeback_request_accepts_body_at_256kib_boundary() -> None:
    base = {
        "customer_id": "c", "incident_doc_id": "d",
        "source_system": SourceSystem.PAGERDUTY,
        "source_event_id": "e", "version": 1, "mode": "full",
        "title": "t",
    }
    req = InvestigationWritebackRequest(**base, body_markdown="x" * 262144)
    assert len(req.body_markdown) == 262144

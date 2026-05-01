import pytest

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector
from shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    PrincipalType,
    SourceSystem,
)
from shared.models import WebhookEvent


def _event(customer_id: str = "cust-1", session_id: str = "s-1") -> WebhookEvent:
    from datetime import UTC, datetime
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.CLAUDE_CODE,
        source_event_id=f"{session_id}:0",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/claude_code/cust-1/s-1/0.jsonl",
        raw_payload={
            "device_id": "dev-1",
            "session_id": session_id,
            "batch_seq": 0,
            "cwd": "/tmp/p",
            "events": [],
            "employee_id": "emp-1",
        },
        headers={},
    )


@pytest.mark.asyncio
async def test_normalize_incomplete_emits_bare_session_doc_only() -> None:
    c = ClaudeCodeConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [{"line_no": 0, "raw": {"role": "user", "content": "hi"}}],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_event(), hydrated)

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_type == DocType.CLAUDE_CODE_SESSION
    assert doc.doc_class == DocClass.RAW_SOURCE
    assert doc.source_id == "s-1"
    assert doc.author_id == "emp-1"
    # ACL: one principal = the employee
    assert len(doc.acl.principals) == 1
    assert doc.acl.principals[0].principal_type == PrincipalType.USER
    assert doc.acl.principals[0].principal_id == "emp-1"
    # Graph: at least one AUTHORED edge
    assert any(
        getattr(e, "edge_type", None) == EdgeType.AUTHORED for e in result.graph_edges
    )
    # No unit docs while incomplete
    assert all(d.doc_type == DocType.CLAUDE_CODE_SESSION for d in result.documents)


@pytest.mark.asyncio
async def test_normalize_complete_emits_session_plus_units(monkeypatch) -> None:
    """When session_complete is True the connector calls extract_units_from_session
    and emits one Document per returned unit, each with parent_doc_id pointing at
    the session doc.
    """
    import services.ingestion.handlers.claude_code as cc_mod
    ext_mod = cc_mod._ext

    bundle = ext_mod.UnitBundle(
        qa=[ext_mod.QA(prompt="Why?", outcome="Because.", tags=["x"])],
        code_change=[ext_mod.CodeChange(file="a.py", before="x", after="y", intent="z")],
        decision=[ext_mod.Decision(question="?", options_considered=["a", "b"], chosen="b", rationale="r")],
        file_ref=[ext_mod.FileRef(files=["a.py"], context="ctx")],
    )

    async def fake_extract(*a, **k):
        return bundle

    # The connector calls await _ext.extract_units_from_session(...) where _ext
    # is `import shared.claude_code_extraction as _ext`. Patch the function on the
    # aliased module directly so future import-style refactors don't silently
    # break the test.
    monkeypatch.setattr(cc_mod._ext, "extract_units_from_session", fake_extract)

    c = ClaudeCodeConnector(make_default_context())
    hydrated = {
        "session_id": "s-2",
        "events": [{"line_no": 0, "raw": {}}],
        "session_complete": True,
        "cwd": "/tmp/p",
    }
    ev = _event(session_id="s-2")
    result = await c.normalize(ev, hydrated)

    by_type: dict = {}
    for d in result.documents:
        by_type.setdefault(d.doc_type, []).append(d)
    assert len(by_type[DocType.CLAUDE_CODE_SESSION]) == 1
    assert len(by_type[DocType.CLAUDE_CODE_QA]) == 1
    assert len(by_type[DocType.CLAUDE_CODE_CODE_CHANGE]) == 1
    assert len(by_type[DocType.CLAUDE_CODE_DECISION]) == 1
    assert len(by_type[DocType.CLAUDE_CODE_FILE_REF]) == 1

    session_doc_id = by_type[DocType.CLAUDE_CODE_SESSION][0].doc_id
    for unit_type in (DocType.CLAUDE_CODE_QA, DocType.CLAUDE_CODE_CODE_CHANGE,
                      DocType.CLAUDE_CODE_DECISION, DocType.CLAUDE_CODE_FILE_REF):
        for d in by_type[unit_type]:
            assert d.parent_doc_id == session_doc_id
            assert d.author_id == "emp-1"


@pytest.mark.asyncio
async def test_normalize_raises_on_missing_employee_id() -> None:
    """employee_id is required to populate Document.author_id and the ACL.
    Missing field must raise InvalidWebhookPayload, not silently default."""
    from shared.exceptions import InvalidWebhookPayload

    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload.pop("employee_id")
    with pytest.raises(InvalidWebhookPayload, match="employee_id"):
        await c.normalize(
            ev,
            {"session_id": "s-x", "events": [], "session_complete": False, "cwd": None},
        )


# ---- Person node properties: employee_name + employee_email ----------------
#
# Gateway (prbe-backend PR #67) injects employee_name/employee_email into the
# webhook body alongside employee_id. The handler must merge them onto the
# Person GraphNodeSpec.properties so the LOWER(properties->>'name') graph
# index resolves name-keyed searches. Absence of the fields means "no value"
# (NEVER null in the wire format) — empty string is treated the same.


def _person_props(result) -> dict:
    person_specs = [
        n for n in result.graph_nodes if n.canonical_id == "emp-1"
    ]
    assert len(person_specs) == 1, f"expected one Person node, got {person_specs}"
    return person_specs[0].properties


@pytest.mark.asyncio
async def test_normalize_writes_name_and_email_on_person_when_present() -> None:
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_name"] = "Richard Roe"
    ev.raw_payload["employee_email"] = "richard@example.com"

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert props == {
        "employee_id": "emp-1",
        "name": "Richard Roe",
        "email": "richard@example.com",
    }


@pytest.mark.asyncio
async def test_normalize_omits_name_when_payload_missing_name() -> None:
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_email"] = "richard@example.com"

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert "name" not in props
    assert props["email"] == "richard@example.com"
    assert props["employee_id"] == "emp-1"


@pytest.mark.asyncio
async def test_normalize_omits_email_when_payload_missing_email() -> None:
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_name"] = "Richard Roe"

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert "email" not in props
    assert props["name"] == "Richard Roe"
    assert props["employee_id"] == "emp-1"


@pytest.mark.asyncio
async def test_normalize_omits_both_when_payload_has_only_id() -> None:
    """Regression: pre-Lane-A daemons / failed enrichment paths send only
    employee_id. Person properties must remain exactly {"employee_id": ...}
    so the existing wire contract keeps working."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    # _event() already sets only employee_id by default.

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert props == {"employee_id": "emp-1"}


@pytest.mark.asyncio
async def test_normalize_pulls_name_from_merged_events_on_finalize() -> None:
    """Finalize events have no per-event identity in event.raw_payload; the
    fallback walks merged_events for the first record carrying the field.
    Mirrors _employee_id_from_event's existing finalize behaviour."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    # Simulate finalize-event shape: raw_payload has session/finalize but no
    # per-event identity fields beyond employee_id.
    ev.raw_payload.pop("employee_name", None)
    ev.raw_payload.pop("employee_email", None)
    hydrated = {
        "session_id": "s-1",
        "events": [
            {
                "line_no": 0,
                "raw": {},
                "employee_name": "Richard Roe",
                "employee_email": "richard@example.com",
            }
        ],
        "session_complete": False,
        "cwd": None,
    }
    result = await c.normalize(ev, hydrated)
    props = _person_props(result)
    assert props["name"] == "Richard Roe"
    assert props["email"] == "richard@example.com"


@pytest.mark.asyncio
async def test_normalize_treats_empty_string_name_as_absent() -> None:
    """Empty strings must not land in properties — the
    LOWER(properties->>'name') index would otherwise hold a useless ''
    entry per employee, defeating the index."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_name"] = ""
    ev.raw_payload["employee_email"] = ""

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert "name" not in props
    assert "email" not in props
    assert props == {"employee_id": "emp-1"}

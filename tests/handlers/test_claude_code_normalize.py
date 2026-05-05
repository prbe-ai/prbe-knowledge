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


# ---- Title format + identity metadata + Person hostname --------------------
#
# The handler renders the session Document title via _format_session_title
# so it carries the human's identity. The same routine drives unit-doc
# titles. Identity fields also land on documents.metadata and Person
# graph_nodes.properties so retrieval has multiple ranking surfaces.


_HOSTNAME = "Richards-Macbook-Pro"


def _session_doc(result):
    from shared.constants import DocType

    docs = [d for d in result.documents if d.doc_type == DocType.CLAUDE_CODE_SESSION]
    assert len(docs) == 1
    return docs[0]


@pytest.mark.parametrize(
    "name,email,hostname,expected_title",
    [
        # All 8 combinations of (name, email, hostname) presence.
        (None, None, None, "Claude Code session s-1"),
        ("Richard Wei", None, None, "Richard Wei's Claude Code session s-1"),
        (None, "richard@prbe.ai", None, "(richard@prbe.ai) Claude Code session s-1"),
        (None, None, _HOSTNAME, f"Claude Code session s-1 ({_HOSTNAME})"),
        (
            "Richard Wei",
            "richard@prbe.ai",
            None,
            "Richard Wei's (richard@prbe.ai) Claude Code session s-1",
        ),
        (
            "Richard Wei",
            None,
            _HOSTNAME,
            f"Richard Wei's Claude Code session s-1 ({_HOSTNAME})",
        ),
        (
            None,
            "richard@prbe.ai",
            _HOSTNAME,
            f"(richard@prbe.ai) Claude Code session s-1 ({_HOSTNAME})",
        ),
        (
            "Richard Wei",
            "richard@prbe.ai",
            _HOSTNAME,
            f"Richard Wei's (richard@prbe.ai) Claude Code session s-1 ({_HOSTNAME})",
        ),
    ],
)
def test_format_session_title_all_combinations(
    name: str | None,
    email: str | None,
    hostname: str | None,
    expected_title: str,
) -> None:
    """Direct unit test on the title formatter — exercises every presence
    combination so future changes to the format are explicit."""
    from services.ingestion.handlers.claude_code import _format_session_title

    assert (
        _format_session_title("s-1", name, email, hostname) == expected_title
    )


def test_format_session_title_uses_kind_for_unit_docs() -> None:
    from services.ingestion.handlers.claude_code import _format_session_title

    title = _format_session_title("abcd1234", "Ada", None, None, kind="decision")
    assert title == "Ada's decision abcd1234"


@pytest.mark.asyncio
async def test_normalize_session_title_full_identity() -> None:
    """Session doc title carries name + email + hostname when all present."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event(session_id="82861aa0XXXX")
    ev.raw_payload["employee_name"] = "Richard Wei"
    ev.raw_payload["employee_email"] = "richard@prbe.ai"
    ev.raw_payload["employee_hostname"] = _HOSTNAME

    result = await c.normalize(
        ev,
        {"session_id": "82861aa0XXXX", "events": [], "session_complete": False, "cwd": None},
    )
    doc = _session_doc(result)
    assert (
        doc.title
        == f"Richard Wei's (richard@prbe.ai) Claude Code session 82861aa0 ({_HOSTNAME})"
    )


@pytest.mark.asyncio
async def test_normalize_session_title_id_only_regression() -> None:
    """Regression: a payload with no identity fields keeps the pre-Lane-B
    title shape ("Claude Code session XXXXXXXX")."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event(session_id="82861aa0XXXX")

    result = await c.normalize(
        ev,
        {"session_id": "82861aa0XXXX", "events": [], "session_complete": False, "cwd": None},
    )
    doc = _session_doc(result)
    assert doc.title == "Claude Code session 82861aa0"


@pytest.mark.asyncio
async def test_normalize_metadata_includes_identity_when_present() -> None:
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_name"] = "Richard Wei"
    ev.raw_payload["employee_email"] = "richard@prbe.ai"
    ev.raw_payload["employee_hostname"] = _HOSTNAME

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    md = _session_doc(result).metadata
    assert md["employee_name"] == "Richard Wei"
    assert md["employee_email"] == "richard@prbe.ai"
    assert md["employee_hostname"] == _HOSTNAME


@pytest.mark.asyncio
async def test_normalize_metadata_omits_identity_when_absent() -> None:
    """Absent identity fields must not appear on metadata — keeps JSONB
    null-free and matches the rest of the handler's omit-when-absent
    convention."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    md = _session_doc(result).metadata
    assert "employee_name" not in md
    assert "employee_email" not in md
    assert "employee_hostname" not in md


@pytest.mark.asyncio
async def test_normalize_person_props_includes_hostname_when_present() -> None:
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_name"] = "Richard Wei"
    ev.raw_payload["employee_email"] = "richard@prbe.ai"
    ev.raw_payload["employee_hostname"] = _HOSTNAME

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert props["hostname"] == _HOSTNAME
    # Existing fields still stamped.
    assert props["name"] == "Richard Wei"
    assert props["email"] == "richard@prbe.ai"


@pytest.mark.asyncio
async def test_normalize_person_props_omits_hostname_when_absent() -> None:
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload["employee_name"] = "Richard Wei"
    ev.raw_payload["employee_email"] = "richard@prbe.ai"

    result = await c.normalize(
        ev,
        {"session_id": "s-1", "events": [], "session_complete": False, "cwd": None},
    )
    props = _person_props(result)
    assert "hostname" not in props


@pytest.mark.asyncio
async def test_normalize_unit_doc_title_uses_identity(monkeypatch) -> None:
    """Unit docs (qa/code_change/decision/file_ref) reuse the same
    identity-bearing title shape, with their unit_kind in place of the
    session prefix."""
    import services.ingestion.handlers.claude_code as cc_mod

    ext_mod = cc_mod._ext
    bundle = ext_mod.UnitBundle(
        qa=[],
        code_change=[],
        decision=[
            ext_mod.Decision(
                question="?",
                options_considered=["a", "b"],
                chosen="b",
                rationale="r",
            )
        ],
        file_ref=[],
    )

    async def fake_extract(*a, **k):
        return bundle

    monkeypatch.setattr(cc_mod._ext, "extract_units_from_session", fake_extract)

    c = ClaudeCodeConnector(make_default_context())
    ev = _event(session_id="82861aa0XXXX")
    ev.raw_payload["employee_name"] = "Richard Wei"
    ev.raw_payload["employee_email"] = "richard@prbe.ai"
    ev.raw_payload["employee_hostname"] = _HOSTNAME

    result = await c.normalize(
        ev,
        {
            "session_id": "82861aa0XXXX",
            "events": [],
            "session_complete": True,
            "cwd": None,
        },
    )

    from shared.constants import DocType

    decisions = [d for d in result.documents if d.doc_type == DocType.CLAUDE_CODE_DECISION]
    assert len(decisions) == 1
    assert (
        decisions[0].title
        == f"Richard Wei's (richard@prbe.ai) decision 82861aa0 ({_HOSTNAME})"
    )
    # Unit metadata also carries identity.
    md = decisions[0].metadata
    assert md["employee_name"] == "Richard Wei"
    assert md["employee_email"] == "richard@prbe.ai"
    assert md["employee_hostname"] == _HOSTNAME


@pytest.mark.asyncio
async def test_normalize_pulls_hostname_from_merged_events_on_finalize() -> None:
    """Mirrors the existing _employee_name finalize-fallback test —
    finalize events have no per-event identity in raw_payload."""
    c = ClaudeCodeConnector(make_default_context())
    ev = _event()
    ev.raw_payload.pop("employee_hostname", None)
    hydrated = {
        "session_id": "s-1",
        "events": [
            {"line_no": 0, "raw": {}, "employee_hostname": _HOSTNAME}
        ],
        "session_complete": False,
        "cwd": None,
    }
    result = await c.normalize(ev, hydrated)
    props = _person_props(result)
    assert props["hostname"] == _HOSTNAME

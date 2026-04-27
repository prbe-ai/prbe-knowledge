from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector


def _make() -> ClaudeCodeConnector:
    return ClaudeCodeConnector(make_default_context())


def test_extract_external_id_returns_device_id() -> None:
    c = _make()
    out = c.extract_external_id_from_payload(
        headers={},
        raw_payload={"device_id": "dev-uuid-1", "session_id": "s1", "events": []},
    )
    assert out == "dev-uuid-1"


def test_extract_external_id_missing_returns_none() -> None:
    c = _make()
    assert c.extract_external_id_from_payload(headers={}, raw_payload={}) is None


def test_extract_external_id_non_string_returns_none() -> None:
    c = _make()
    assert c.extract_external_id_from_payload(headers={}, raw_payload={"device_id": 123}) is None

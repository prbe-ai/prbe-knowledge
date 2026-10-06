"""The ingest pass and the session's ATIF trajectory (engine/ingest/atif).

What must hold, whatever the tenant's render mode:
  * the indexed text is exactly the event path's text;
  * `trajectory.json` is written on a completing pass, never on a live one,
    scrubbed of credentials, in the session's own folder (so deletion owns it);
  * nothing about the trajectory -- building, comparing, writing -- can fail a
    pass.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import orjson
import pytest

from engine.ingest.atif import store as atif_store
from engine.ingest.atif.mode import RenderMode, render_mode
from engine.ingest.handlers.base import make_default_context
from engine.shared.config import Settings
from engine.shared.constants import SourceSystem
from engine.shared.models import WebhookEvent
from engine.shared.session_suppression import is_own_folder_key, session_folder
from engine.shared.storage import StorageNotFound, StorageUnavailable
from engine.shared.transcript_render import lines_from_events, render_lines
from kb.handlers import claude_code as cc_mod
from kb.handlers.claude_code import ClaudeCodeConnector, CodexConnector

# A shape the shared stdlib scrub always catches (an AWS access key id).
SECRET = "AKIAIOSFODNN7EXAMPLE"

EVENTS = [
    {"line_no": 0, "raw": {"type": "user", "message": {"content": f"deploy with {SECRET}"}}},
    {"line_no": 1, "raw": {"type": "assistant", "inference_id": "m1", "message": {
        "model": "claude-x",
        "content": [{"type": "thinking", "thinking": "check"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "summary": "make deploy"}]}}},
    {"line_no": 2, "raw": {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "result_bytes": 9}]}}},
    {"line_no": 3, "raw": {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "it failed"}], "stop_reason": "max_tokens"}}},
]


class FakeStore:
    def __init__(self, *, fail_put: bool = False) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.fail_put = fail_put

    async def bucket_for(self, customer_id: str) -> str:
        return f"bucket-{customer_id}"

    async def put(self, bucket: str, key: str, body: bytes, content_type: str = "") -> None:
        if self.fail_put:
            raise StorageUnavailable("put_object failed: test")
        self.objects[(bucket, key)] = body

    async def get(self, bucket: str, key: str) -> bytes:
        try:
            return self.objects[(bucket, key)]
        except KeyError:
            raise StorageNotFound(key) from None


def _event(customer: str = "cust-1", session: str = "s-1",
           source: SourceSystem = SourceSystem.CLAUDE_CODE) -> WebhookEvent:
    return WebhookEvent(
        customer_id=customer,
        source_system=source,
        source_event_id=f"{session}:0",
        received_at=datetime.now(UTC),
        payload_s3_key=f"raw/{source.value}/{customer}/s/0.json",
        raw_payload={"session_id": session, "batch_seq": 0, "events": [], "employee_id": "emp-1"},
        headers={},
    )


def _settings(**kw: str) -> Settings:
    return Settings(**{f"session_render_{k}": v for k, v in kw.items()})


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(cc_mod, "get_store", lambda: fake)
    return fake


@pytest.fixture
def mined(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What extraction was handed, per call; it returns an empty, non-authoritative bundle."""
    calls: list[dict[str, Any]] = []

    async def fake_extract(**kw: Any) -> Any:
        calls.append(kw)
        return cc_mod._ext.UnitBundle(authoritative=False)

    monkeypatch.setattr(cc_mod._ext, "extract_units_from_session", fake_extract)
    return calls


def _use_mode(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(cc_mod, "render_mode", lambda customer: render_mode(customer, settings))


async def _normalize(complete: bool, event: WebhookEvent | None = None,
                     connector: type[ClaudeCodeConnector] = ClaudeCodeConnector) -> Any:
    c = connector(make_default_context())
    return await c.normalize(
        event or _event(),
        {"session_id": "s-1", "events": EVENTS, "session_complete": complete, "cwd": "/p"},
    )


# -- render mode ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "customer", "expected"),
    [
        (_settings(), "c", RenderMode.LEGACY),
        (_settings(default="shadow"), "c", RenderMode.SHADOW),
        (_settings(default=" ATIF "), "c", RenderMode.ATIF),
        (_settings(default="atfi"), "c", RenderMode.LEGACY),
        (_settings(shadow_customers="a, c"), "c", RenderMode.SHADOW),
        (_settings(shadow_customers="c", atif_customers="c"), "c", RenderMode.ATIF),
        (_settings(default="atif", shadow_customers="c"), "c", RenderMode.SHADOW),
        (_settings(atif_customers="x"), "c", RenderMode.LEGACY),
    ],
)
def test_render_mode(settings: Settings, customer: str, expected: RenderMode) -> None:
    assert render_mode(customer, settings) is expected


# -- what the pass indexes -------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "shadow", "atif"])
@pytest.mark.parametrize("complete", [False, True])
async def test_indexed_text_is_the_event_text_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, mode: str, complete: bool
) -> None:
    _use_mode(monkeypatch, _settings(default=mode))
    result = await _normalize(complete)
    assert result.documents[0].body == render_lines(lines_from_events(EVENTS))


async def test_a_live_legacy_pass_builds_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore
) -> None:
    def must_not_build(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("built on a live legacy pass")

    monkeypatch.setattr(cc_mod, "build_trajectory", must_not_build)
    _use_mode(monkeypatch, _settings())
    await _normalize(complete=False)
    assert store.objects == {}


async def test_atif_mode_serves_the_trajectory_lines_when_they_match(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    served: list[Any] = []
    real = cc_mod.lines_from_trajectory

    def spy(trajectory: dict[str, Any]) -> Any:
        served.append(real(trajectory))
        return served[-1]

    monkeypatch.setattr(cc_mod, "lines_from_trajectory", spy)
    _use_mode(monkeypatch, _settings(default="atif"))
    await _normalize(complete=True)
    assert mined[0]["lines"] is served[0], "extraction must read the trajectory's lines"


async def test_atif_mode_falls_back_to_events_when_they_differ(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    def tampered(trajectory: dict[str, Any]) -> Any:
        lines = cc_mod.lines_from_events(EVENTS)
        return [*lines[:-1], type(lines[-1])(line_no=3, text="ASSISTANT: something else")]

    monkeypatch.setattr(cc_mod, "lines_from_trajectory", tampered)
    _use_mode(monkeypatch, _settings(default="atif"))
    result = await _normalize(complete=True)
    assert result.documents[0].body == render_lines(lines_from_events(EVENTS))
    assert "lines" not in mined[0], "extraction renders the events itself"


async def test_atif_mode_does_not_serve_a_trajectory_with_an_unparsed_event(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    real_build = cc_mod.build_trajectory

    def degraded(*a: Any, **k: Any) -> Any:
        built = real_build(*a, **k)
        built.unparsed = 1
        return built

    monkeypatch.setattr(cc_mod, "build_trajectory", degraded)
    sentinel: list[Any] = []
    real_lines = cc_mod.lines_from_trajectory
    monkeypatch.setattr(cc_mod, "lines_from_trajectory",
                        lambda t: sentinel.append(real_lines(t)) or sentinel[-1])
    _use_mode(monkeypatch, _settings(default="atif"))
    await _normalize(complete=True)
    assert sentinel, "the trajectory was rendered and compared"
    assert "lines" not in mined[0], "a degraded trajectory is never served"


# -- trajectory.json -----------------------------------------------------------------


async def test_a_completing_pass_stores_a_scrubbed_trajectory_in_the_session_folder(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    _use_mode(monkeypatch, _settings())
    await _normalize(complete=True, connector=CodexConnector,
                     event=_event(source=SourceSystem.CODEX))
    [(bucket, key)] = list(store.objects)
    assert bucket == "bucket-cust-1"
    assert key == "raw/codex/cust-1/s-1/trajectory.json"
    assert is_own_folder_key(key, session_folder("codex", "cust-1", "s-1"))
    stored = orjson.loads(store.objects[(bucket, key)])
    assert SECRET not in orjson.dumps(stored).decode()
    assert stored["agent"]["name"] == "codex"
    assert "validation_error" not in stored.get("extra", {})
    # Search reads the same scrub through the chunk backstop; the trajectory's
    # text still rebuilds once scrubbed.
    assert [s["source"] for s in stored["steps"]] == ["user", "agent", "agent"]


async def test_a_failed_write_never_fails_the_pass(
    monkeypatch: pytest.MonkeyPatch, mined: list
) -> None:
    monkeypatch.setattr(cc_mod, "get_store", lambda: FakeStore(fail_put=True))
    _use_mode(monkeypatch, _settings(default="atif"))
    result = await _normalize(complete=True)
    assert result.documents[0].body == render_lines(lines_from_events(EVENTS))


async def test_a_failed_build_never_fails_the_pass(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(cc_mod, "build_trajectory", boom)
    _use_mode(monkeypatch, _settings(default="atif"))
    result = await _normalize(complete=True)
    assert result.documents[0].body == render_lines(lines_from_events(EVENTS))
    assert store.objects == {}


async def test_an_invalid_trajectory_is_written_with_its_error() -> None:
    fake = FakeStore()
    invalid = {"schema_version": "ATIF-v1.8", "agent": {"name": "x", "version": "1"}, "steps": []}
    size, error = await atif_store.write_trajectory(fake, "b", "k", invalid)
    assert error is not None and size > 0
    assert orjson.loads(fake.objects[("b", "k")])["extra"]["validation_error"] == error


async def test_read_trajectory_missing_is_none() -> None:
    assert await atif_store.read_trajectory(FakeStore(), "b", "nope") is None


# -- ownership ---------------------------------------------------------------------


def test_the_trajectory_is_its_own_sessions_object_only() -> None:
    folder = session_folder("claude_code", "c", "X")
    assert is_own_folder_key(f"{folder}trajectory.json", folder)
    # Protocol 1 allows `/` in a session id: session `X/y`'s trajectory sits
    # inside session `X`'s folder but is not session `X`'s.
    assert not is_own_folder_key(f"{folder}y/trajectory.json", folder)
    assert not is_own_folder_key(f"{folder}trajectory.json.bak", folder)


async def test_a_trajectory_past_the_limit_is_skipped_before_any_scan(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    scanned: list[Any] = []

    async def no_scan(value: Any) -> Any:
        scanned.append(value)
        return value

    monkeypatch.setattr(atif_store, "redact_payload_async", no_scan)
    _use_mode(monkeypatch, _settings())
    monkeypatch.setattr(cc_mod, "get_settings", lambda: Settings(session_trajectory_max_bytes=10))
    result = await _normalize(complete=True)
    assert store.objects == {} and scanned == []
    assert result.documents[0].body == render_lines(lines_from_events(EVENTS))

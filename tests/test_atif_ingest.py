"""The ingest pass and the session's ATIF trajectory (engine/ingest/atif).

What must hold, whatever the tenant's render mode:
  * the indexed text is exactly what the event renderer produced before this
    module existed (pinned below as a literal);
  * a completing pass writes the final `trajectory.json` and a live pass a
    throttled live copy (`extra.session_ended` says which); either is written
    without the engine's render provenance, scrubbed the way the index is, in
    the session's own folder (so deletion owns it);
  * a session that resumes, or whose newer trajectory cannot be written, has
    its old one removed, so a reader never gets a stale document;
  * nothing about the trajectory -- building, comparing, writing -- can fail a
    pass.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import orjson
import pytest
from structlog.testing import capture_logs

from engine.ingest.atif import store as atif_store
from engine.ingest.atif.mode import RenderMode, render_mode
from engine.ingest.handlers.base import make_default_context
from engine.retrieval.middleware import _should_log
from engine.retrieval.usage import EVENT_TYPE_GET_SOURCE, event_type_for
from engine.shared.config import Settings
from engine.shared.constants import SourceSystem
from engine.shared.models import WebhookEvent
from engine.shared.session_suppression import is_own_folder_key, session_folder
from engine.shared.storage import StorageNotFound, StorageUnavailable
from engine.shared.transcript_render import Line, lines_from_events, render_lines
from kb.handlers import claude_code as cc_mod
from kb.handlers.claude_code import ClaudeCodeConnector, CodexConnector

# A shape the shared stdlib scrub always catches (an AWS access key id).
SECRET = "AKIAIOSFODNN7EXAMPLE"

EVENTS = [
    {"line_no": 0, "raw": {"type": "user", "message": {"content": f"deploy with {SECRET}"}}},
    {"line_no": 1, "raw": {"type": "assistant", "inference_id": "m1", "message": {
        "model": "claude-x",
        "content": [{"type": "thinking", "thinking": f"plan\nkey {SECRET} here\nthen run"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "summary": "make deploy"}]}}},
    {"line_no": 2, "raw": {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "result_bytes": 9}]}}},
    {"line_no": 3, "raw": {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "it failed"}], "stop_reason": "max_tokens"}}},
]

#: What the event renderer produced for EVENTS on prbe-knowledge main before
#: this module existed. The indexed body must equal it in every mode.
EXPECTED_BODY = (
    f"USER: deploy with {SECRET}"
    f"\n\nASSISTANT (thinking): plan\nkey {SECRET} here\nthen run\nTOOL_USE: Bash — make deploy"
    "\n\nTOOL_RESULT (t1): error (9 bytes)"
    "\n\nASSISTANT: it failed\n[stop: max_tokens]"
)

KEY = "raw/claude_code/cust-1/s-1/trajectory.json"


class FakeStore:
    def __init__(self, *, fail_put: bool = False) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[str] = []
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

    async def delete(self, bucket: str, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop((bucket, key), None)


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


def _settings(**kw: Any) -> Settings:
    return Settings(**{f"session_{k}": v for k, v in kw.items()})


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


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Every build_and_render call's result."""
    seen: list[tuple[Any, ...]] = []
    real = cc_mod.build_and_render

    def spy(*args: Any) -> Any:
        result = real(*args)
        seen.append(result)
        return result

    monkeypatch.setattr(cc_mod, "build_and_render", spy)
    return seen


@pytest.fixture(autouse=True)
def _fresh_live_throttle() -> Any:
    for table in (cc_mod._LIVE_WRITES, cc_mod._LIVE_TOO_LARGE, cc_mod._LIVE_TURNS):
        table.clear()
    yield
    for table in (cc_mod._LIVE_WRITES, cc_mod._LIVE_TOO_LARGE, cc_mod._LIVE_TURNS):
        table.clear()


def _use_mode(monkeypatch: pytest.MonkeyPatch, mode: str = "legacy", **kw: Any) -> None:
    settings = _settings(render_default=mode, **kw)
    monkeypatch.setattr(cc_mod, "render_mode", lambda customer: render_mode(customer, settings))
    monkeypatch.setattr(cc_mod, "get_settings", lambda: settings)


async def _normalize(complete: bool, event: WebhookEvent | None = None,
                     connector: type[ClaudeCodeConnector] = ClaudeCodeConnector,
                     **hydrated: Any) -> Any:
    c = connector(make_default_context())
    return await c.normalize(
        event or _event(),
        {"session_id": "s-1", "events": EVENTS, "session_complete": complete, "cwd": "/p",
         **hydrated},
    )


# -- render mode ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "customer", "expected"),
    [
        (_settings(), "c", RenderMode.LEGACY),
        (_settings(render_default="shadow"), "c", RenderMode.SHADOW),
        (_settings(render_default=" ATIF "), "c", RenderMode.ATIF),
        (_settings(render_default="atfi"), "c", RenderMode.LEGACY),
        (_settings(render_shadow_customers="a, c"), "c", RenderMode.SHADOW),
        (_settings(render_shadow_customers="c", render_atif_customers="c"), "c", RenderMode.ATIF),
        (_settings(render_default="atif", render_shadow_customers="c"), "c", RenderMode.SHADOW),
        (_settings(render_atif_customers="x"), "c", RenderMode.LEGACY),
    ],
)
def test_render_mode(settings: Settings, customer: str, expected: RenderMode) -> None:
    assert render_mode(customer, settings) is expected


# -- what the pass indexes -------------------------------------------------------


def test_the_event_renderer_still_produces_main_s_body() -> None:
    assert render_lines(lines_from_events(EVENTS)) == EXPECTED_BODY


@pytest.mark.parametrize("mode", ["legacy", "shadow", "atif"])
@pytest.mark.parametrize("complete", [False, True])
async def test_indexed_text_is_main_s_text_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, mode: str, complete: bool
) -> None:
    _use_mode(monkeypatch, mode)
    result = await _normalize(complete)
    assert result.documents[0].body == EXPECTED_BODY


@pytest.mark.parametrize("mode", ["legacy", "shadow", "atif"])
async def test_with_live_copies_off_a_live_pass_builds_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, builds: list, mode: str
) -> None:
    _use_mode(monkeypatch, mode, trajectory_live_interval_s=0)
    await _normalize(complete=False)
    assert builds == [] and store.objects == {} and store.deleted == []


@pytest.mark.parametrize("mode", ["legacy", "shadow", "atif"])
async def test_a_live_pass_keeps_a_live_copy_and_serves_the_events_text(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list, mode: str
) -> None:
    _use_mode(monkeypatch, mode)
    with capture_logs() as logs:
        result = await _normalize(complete=False)
    assert result.documents[0].body == EXPECTED_BODY
    assert not any(e["event"] == "session_render.compared" for e in logs), "only at the end"
    stored = orjson.loads(store.objects[("bucket-cust-1", KEY)])
    assert stored["extra"] == {"session_ended": False}
    assert len(builds) == 1


async def test_live_copies_are_throttled_per_session_and_the_final_one_never_is(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    _use_mode(monkeypatch, trajectory_live_interval_s=3600)
    await _normalize(complete=False)
    await _normalize(complete=False)
    assert len(builds) == 1, "the second live pass inside the interval builds nothing"
    await _normalize(complete=True)
    assert len(builds) == 2
    stored = orjson.loads(store.objects[("bucket-cust-1", KEY)])
    assert stored["extra"] == {"session_ended": True}


async def test_atif_mode_serves_the_trajectory_lines_when_they_match(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    _use_mode(monkeypatch, "atif")
    await _normalize(complete=True)
    [(_built, atif_lines, _error)] = builds
    assert mined[0]["lines"] is atif_lines, "extraction must read the trajectory's lines"


async def test_shadow_logs_agreement_and_serves_the_events(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    _use_mode(monkeypatch, "shadow")
    with capture_logs() as logs:
        await _normalize(complete=True)
    [compared] = [e for e in logs if e["event"] == "session_render.compared"]
    assert (compared["same"], compared["first_diff"], compared["mode"]) == (True, None, "shadow")
    assert "lines" not in mined[0], "shadow never serves the trajectory's lines"


@pytest.mark.parametrize("mode", ["shadow", "atif"])
async def test_a_disagreement_is_logged_and_the_events_are_served(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, mode: str
) -> None:
    real = cc_mod.build_and_render

    def tampered(*args: Any) -> Any:
        built, lines, error = real(*args)
        return built, [*lines[:-1], Line(line_no=3, text="ASSISTANT: something else")], error

    monkeypatch.setattr(cc_mod, "build_and_render", tampered)
    _use_mode(monkeypatch, mode)
    with capture_logs() as logs:
        result = await _normalize(complete=True)
    [compared] = [e for e in logs if e["event"] == "session_render.compared"]
    assert (compared["same"], compared["first_diff"]) == (False, 3)
    assert compared["log_level"] == "warning"
    assert result.documents[0].body == EXPECTED_BODY
    assert "lines" not in mined[0]


async def test_an_unrenderable_trajectory_is_logged_and_the_events_are_served(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    real = cc_mod.build_and_render
    monkeypatch.setattr(
        cc_mod, "build_and_render", lambda *a: (real(*a)[0], None, "UnrenderableTrajectory")
    )
    _use_mode(monkeypatch, "atif")
    with capture_logs() as logs:
        result = await _normalize(complete=True)
    assert any(e["event"] == "session_render.unrenderable" for e in logs)
    assert result.documents[0].body == EXPECTED_BODY
    assert "lines" not in mined[0]


async def test_atif_mode_does_not_serve_a_trajectory_with_an_unparsed_event(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    real = cc_mod.build_and_render

    def degraded(*args: Any) -> Any:
        built, lines, error = real(*args)
        built.unparsed = 1
        return built, lines, error

    monkeypatch.setattr(cc_mod, "build_and_render", degraded)
    _use_mode(monkeypatch, "atif")
    await _normalize(complete=True)
    assert "lines" not in mined[0], "a degraded trajectory is never served"


# -- trajectory.json -----------------------------------------------------------------


async def test_a_completing_pass_stores_a_scrubbed_trajectory_in_the_session_folder(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    _use_mode(monkeypatch)
    await _normalize(complete=True, connector=CodexConnector,
                     event=_event(source=SourceSystem.CODEX))
    [(bucket, key)] = list(store.objects)
    assert bucket == "bucket-cust-1"
    assert key == "raw/codex/cust-1/s-1/trajectory.json"
    assert is_own_folder_key(key, session_folder("codex", "cust-1", "s-1"))
    stored = orjson.loads(store.objects[(bucket, key)])
    assert SECRET not in orjson.dumps(stored).decode()
    assert stored["agent"]["name"] == "codex"
    assert stored["extra"] == {"session_ended": True}, "render provenance is engine-internal"
    assert all("probe_parts" not in (s.get("extra") or {}) for s in stored["steps"])
    assert [s["source"] for s in stored["steps"]] == ["user", "agent", "agent"]
    # Scrubbed the way the index is: the finding costs its line, never the field.
    reasoning = stored["steps"][1]["reasoning_content"]
    assert reasoning.startswith("plan\n") and reasoning.endswith("\nthen run")
    assert stored["steps"][0]["message"].startswith("deploy with ")


async def test_a_trajectory_past_the_limit_is_skipped_before_any_scan_and_the_old_one_removed(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    scanned: list[Any] = []

    async def no_scan(value: Any) -> Any:
        scanned.append(value)
        return value

    monkeypatch.setattr(atif_store, "scrub_trajectory", no_scan)
    store.objects[("bucket-cust-1", KEY)] = b'{"old": true}'
    _use_mode(monkeypatch, trajectory_max_bytes=10)
    result = await _normalize(complete=True)
    assert store.objects == {} and scanned == [] and store.deleted == [KEY]
    assert result.documents[0].body == EXPECTED_BODY


async def test_a_failed_write_never_fails_the_pass(
    monkeypatch: pytest.MonkeyPatch, mined: list
) -> None:
    fake = FakeStore(fail_put=True)
    monkeypatch.setattr(cc_mod, "get_store", lambda: fake)
    _use_mode(monkeypatch, "atif")
    result = await _normalize(complete=True)
    assert result.documents[0].body == EXPECTED_BODY
    assert fake.deleted == [KEY], "the stale document goes when its replacement cannot be written"


async def test_a_failed_build_never_fails_the_pass_and_removes_the_old_trajectory(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    def boom(*_a: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(cc_mod, "build_and_render", boom)
    store.objects[("bucket-cust-1", KEY)] = b'{"old": true}'
    _use_mode(monkeypatch, "atif")
    result = await _normalize(complete=True)
    assert result.documents[0].body == EXPECTED_BODY
    assert store.objects == {} and store.deleted == [KEY]


async def test_with_live_copies_off_a_resumed_session_s_old_trajectory_is_removed(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, builds: list
) -> None:
    store.objects[("bucket-cust-1", KEY)] = b'{"old": true}'
    _use_mode(monkeypatch, trajectory_live_interval_s=0)
    await _normalize(complete=False, ended_before=True)
    assert builds == [] and store.objects == {} and store.deleted == [KEY]


async def test_a_resumed_session_s_final_copy_is_replaced_by_a_live_one(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    store.objects[("bucket-cust-1", KEY)] = b'{"old": true}'
    _use_mode(monkeypatch)
    await _normalize(complete=False, ended_before=True)
    stored = orjson.loads(store.objects[("bucket-cust-1", KEY)])
    assert stored["extra"] == {"session_ended": False} and len(builds) == 1


async def test_an_invalid_trajectory_is_written_with_its_error() -> None:
    fake = FakeStore()
    invalid = {"schema_version": "ATIF-v1.8", "agent": {"name": "x", "version": "1"}, "steps": []}
    size, error = await atif_store.write_trajectory(fake, "b", "k", invalid)
    assert error is not None and size > 0
    assert orjson.loads(fake.objects[("b", "k")])["extra"]["validation_error"] == error


async def test_read_trajectory_missing_or_unreadable_is_none() -> None:
    fake = FakeStore()
    assert await atif_store.read_trajectory(fake, "b", "nope") is None
    fake.objects[("b", "bad")] = b"not json"
    assert await atif_store.read_trajectory(fake, "b", "bad") is None


# -- ownership and telemetry -------------------------------------------------------


def test_the_trajectory_is_its_own_sessions_object_only() -> None:
    folder = session_folder("claude_code", "c", "X")
    assert is_own_folder_key(f"{folder}trajectory.json", folder)
    # Protocol 1 allows `/` in a session id: session `X/y`'s trajectory sits
    # inside session `X`'s folder but is not session `X`'s.
    assert not is_own_folder_key(f"{folder}y/trajectory.json", folder)
    assert not is_own_folder_key(f"{folder}trajectory.json.bak", folder)


def test_trajectory_reads_are_logged_as_source_reads() -> None:
    assert _should_log("/trajectory/claude_code:c:s")
    assert event_type_for("/trajectory/claude_code:c:s") == EVENT_TYPE_GET_SOURCE


async def test_a_pool_outage_retries_the_row_and_keeps_the_trajectory(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    from engine.shared.exceptions import CpuPoolUnavailable

    async def down(*_a: Any, **_k: Any) -> Any:
        raise CpuPoolUnavailable("pool restarting")

    monkeypatch.setattr(cc_mod.cpu_pool, "run_cpu", down)
    store.objects[("bucket-cust-1", KEY)] = b'{"old": true}'
    _use_mode(monkeypatch, "atif")
    with pytest.raises(CpuPoolUnavailable):
        await _normalize(complete=True)
    assert store.deleted == [], "a transient outage never deletes an ended session's trajectory"


async def test_unparsed_events_are_logged_by_the_pass(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    real = cc_mod.build_and_render

    def degraded(*args: Any) -> Any:
        built, lines, error = real(*args)
        built.unparsed, built.unparsed_events = 1, [(2, "user", "KeyError")]
        return built, lines, error

    monkeypatch.setattr(cc_mod, "build_and_render", degraded)
    _use_mode(monkeypatch)
    with capture_logs() as logs:
        await _normalize(complete=True)
    [entry] = [e for e in logs if e["event"] == "atif.event_unparsed"]
    assert (entry["session_id"], entry["line_no"], entry["error"]) == ("s-1", 2, "KeyError")


@pytest.mark.parametrize("mode", ["legacy", "shadow"])
async def test_the_off_switch_writes_nothing_and_removes_the_old_trajectory(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list, mode: str
) -> None:
    store.objects[("bucket-cust-1", KEY)] = b'{"old": true}'
    _use_mode(monkeypatch, mode, trajectory_store=False)
    result = await _normalize(complete=True)
    assert result.documents[0].body == EXPECTED_BODY
    assert store.objects == {} and store.deleted == [KEY]
    assert len(builds) == (0 if mode == "legacy" else 1), "legacy skips the build entirely"


async def _normalize_events(events: list[dict[str, Any]], complete: bool, **hydrated: Any) -> Any:
    c = ClaudeCodeConnector(make_default_context())
    return await c.normalize(_event(), {"session_id": "s-1", "events": events,
                                        "session_complete": complete, "cwd": "/p", **hydrated})


async def test_a_reply_that_ends_its_turn_is_written_live_inside_the_interval(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    _use_mode(monkeypatch, trajectory_live_interval_s=3600)
    await _normalize(complete=False)
    ended_turn = [*EVENTS[:3], {"line_no": 3, "raw": {"type": "assistant", "message": {
        "content": [{"type": "text", "text": "it failed"}], "stop_reason": "end_turn"}}},
        {"line_no": 4, "raw": {"type": "queue-operation", "content": "x"}}]
    # A sixth of the interval after the last live write (at least 10 s), a NEW
    # turn-ending reply is written, even behind trailing bookkeeping; the same
    # reply again is not.
    for key in cc_mod._LIVE_WRITES:
        cc_mod._LIVE_WRITES[key] -= 601
    await _normalize_events(ended_turn, complete=False)
    assert len(builds) == 2, "the newest answer is not held back by the throttle"
    for key in cc_mod._LIVE_WRITES:
        cc_mod._LIVE_WRITES[key] -= 601
    await _normalize_events(ended_turn, complete=False)
    assert len(builds) == 2, "the same reply is not rewritten"


async def test_a_turn_ending_reply_right_after_a_live_write_waits(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    _use_mode(monkeypatch, trajectory_live_interval_s=3600)
    await _normalize(complete=False)
    ended_turn = [*EVENTS[:3], {"line_no": 3, "raw": {"type": "assistant", "message": {
        "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}}}]
    await _normalize_events(ended_turn, complete=False)
    assert len(builds) == 1


async def test_a_completing_pass_resets_the_throttle_so_a_resume_is_live_at_once(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    _use_mode(monkeypatch, trajectory_live_interval_s=3600)
    await _normalize(complete=False)
    await _normalize(complete=True)
    await _normalize(complete=False, ended_before=True)
    stored = orjson.loads(store.objects[("bucket-cust-1", KEY)])
    assert stored["extra"] == {"session_ended": False} and len(builds) == 3


async def test_a_deleted_session_gets_no_trajectory(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    async def deleted(*_a: Any) -> bool:
        return True

    monkeypatch.setattr(cc_mod, "is_session_deleted", deleted)
    _use_mode(monkeypatch)
    await _normalize(complete=True)
    await _normalize(complete=False)
    assert store.objects == {}


async def test_a_pool_outage_on_a_live_pass_keeps_the_previous_copy(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list
) -> None:
    from engine.shared.exceptions import CpuPoolUnavailable

    async def down(*_a: Any, **_k: Any) -> Any:
        raise CpuPoolUnavailable("pool restarting")

    store.objects[("bucket-cust-1", KEY)] = b'{"previous": true}'
    monkeypatch.setattr(cc_mod.cpu_pool, "run_cpu", down)
    _use_mode(monkeypatch)
    result = await _normalize(complete=False)
    assert result.documents[0].body == EXPECTED_BODY
    assert store.objects[("bucket-cust-1", KEY)] == b'{"previous": true}' and store.deleted == []
    assert cc_mod._LIVE_WRITES == {}, "the next pass retries"


async def test_a_failed_live_write_keeps_the_previous_copy(
    monkeypatch: pytest.MonkeyPatch, mined: list
) -> None:
    fake = FakeStore(fail_put=True)
    fake.objects[("bucket-cust-1", KEY)] = b'{"previous": true}'
    monkeypatch.setattr(cc_mod, "get_store", lambda: fake)
    _use_mode(monkeypatch)
    await _normalize(complete=False)
    assert fake.deleted == [] and fake.objects[("bucket-cust-1", KEY)] == b'{"previous": true}'


async def test_a_live_copy_past_its_cap_is_dropped_and_not_rebuilt_until_the_end(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    store.objects[("bucket-cust-1", KEY)] = b'{"previous": true}'
    _use_mode(monkeypatch, trajectory_live_max_bytes=10, trajectory_live_interval_s=1)
    await _normalize(complete=False)
    assert store.objects == {} and store.deleted == [KEY]
    cc_mod._LIVE_WRITES.clear()
    await _normalize(complete=False)
    assert len(builds) == 1, "remembered as too large: no rebuild every interval"
    await _normalize(complete=True)
    assert orjson.loads(store.objects[("bucket-cust-1", KEY)])["extra"] == {"session_ended": True}


async def test_a_resume_known_too_large_removes_the_old_final_copy(
    monkeypatch: pytest.MonkeyPatch, store: FakeStore, mined: list, builds: list
) -> None:
    """Another process found the live copy too large; this one sees the resume
    after the old final copy was put back: the stale final copy goes, as it
    did before live copies existed."""
    store.objects[("bucket-cust-1", KEY)] = b'{"final": true}'
    _use_mode(monkeypatch)
    cc_mod._LIVE_TOO_LARGE[("cust-1", "claude_code", "s-1")] = 0.0
    await _normalize(complete=False, ended_before=True)
    assert builds == [] and store.deleted == [KEY]


@pytest.mark.parametrize(("raw", "ends"), [
    ({"type": "assistant", "message": {"stop_reason": "end_turn"}}, True),
    ({"type": "assistant", "message": {"stop_reason": "tool_use"}}, False),
    ({"type": "assistant", "_pi_extras": {"stop_reason": "stop"}}, True),
    ({"type": "assistant", "_codex_extras": {"phase": "final_answer"}}, True),
    ({"type": "assistant", "_codex_extras": {"phase": "commentary"}}, False),
    ({"type": "user", "message": {"content": "next"}}, False),
])
def test_which_replies_end_a_turn(raw: dict[str, Any], ends: bool) -> None:
    events = [{"line_no": 7, "raw": raw}, {"line_no": 8, "raw": {"type": "system"}}]
    assert cc_mod._turn_end_line(events) == (7 if ends else None)

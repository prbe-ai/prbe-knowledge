"""The per-segment extraction cache (engine/shared/extraction_cache.py).

A session re-ended with some segments unchanged used to pay a model call for
every segment again: 49 % of segment calls on research, 2026-09-24. These pin
what the cache may and may not do:

  * an unchanged segment of the same session costs no call, and yields the same
    units as the call it replaces;
  * any change to what the model is asked (text, model, revision, prompts,
    schema, max_tokens, cwd) is a different key;
  * only an answer that built and grounded is ever stored;
  * no storage failure fails, or stalls, a pass.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

import pytest

from engine.shared import claude_code_extraction as ext
from engine.shared import extraction_cache as cache_mod
from engine.shared.exceptions import StorageNotFound, StorageUnavailable
from engine.shared.llm_tools import ToolCallParseError

TEXT = "Why does /ingest return 422 for a list of dicts?"
QUOTE = "return 422 for a list of dicts"


class Store:
    """In-memory stand-in for ObjectStore: records every operation."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets = 0
        self.puts = 0
        self.fail_get: BaseException | None = None
        self.fail_put: BaseException | None = None
        self.get_delay = 0.0

    async def bucket_for(self, customer_id: str) -> str:
        return f"bucket-{customer_id}"

    async def get(self, bucket: str, key: str) -> bytes:
        self.gets += 1
        if self.get_delay:
            await asyncio.sleep(self.get_delay)
        if self.fail_get is not None:
            raise self.fail_get
        try:
            return self.objects[f"{bucket}/{key}"]
        except KeyError:
            raise StorageNotFound(key) from None

    async def put(self, bucket: str, key: str, body: bytes) -> None:
        self.puts += 1
        if self.fail_put is not None:
            raise self.fail_put
        self.objects[f"{bucket}/{key}"] = body


class Model:
    """Stand-in for forced_tool_call: counts calls per tool, answers per tool."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.answers: dict[str, Any] = {
            "emit_units": {
                "qa": [{"prompt": "Why 422?", "outcome": "list[dict] coercion", "evidence": QUOTE}]
            },
            "emit_supersessions": {"links": [{"earlier": 0, "later": 1, "reason": "reversed"}]},
        }
        self.declines = False

    async def __call__(self, *, tool_name: str, **_kwargs: Any):
        self.calls[tool_name] = self.calls.get(tool_name, 0) + 1
        if self.declines:
            raise ToolCallParseError("model answered in prose")
        return json.loads(json.dumps(self.answers[tool_name])), None


def _event(line_no: int, text: str) -> dict[str, Any]:
    return {
        "line_no": line_no,
        "raw": {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        },
    }


def _compaction(line_no: int) -> dict[str, Any]:
    return {"line_no": line_no, "raw": {"type": "system", "subtype": "compact_boundary"}}


@pytest.fixture
def model(monkeypatch) -> Model:
    fake = Model()
    monkeypatch.setattr(ext, "forced_tool_call", fake)
    return fake


@pytest.fixture
def store() -> Store:
    return Store()


def _cache(store: Store, session: str = "s1") -> cache_mod.SegmentCache:
    return cache_mod.SegmentCache(store, "bucket-c1", f"raw/claude_code/c1/{session}/extraction-cache/")


async def _mine(events, store: Store, *, cwd: str = "/work", session: str = "s1"):
    return await ext.extract_units_from_session(
        session_id=session, events=events, cwd=cwd, cache=_cache(store, session)
    )


def _units(bundle: ext.UnitBundle) -> list[dict[str, Any]]:
    return [asdict(u) for u in ext.all_units(bundle)]


@pytest.mark.asyncio
async def test_an_unchanged_segment_costs_no_second_call(model, store):
    first = await _mine([_event(0, TEXT)], store)
    second = await _mine([_event(0, TEXT)], store)
    assert model.calls["emit_units"] == 1
    assert (first.calls, first.cache_hits) == (1, 0)
    assert (second.calls, second.cache_hits) == (0, 1)
    assert _units(second) == _units(first)
    assert _units(second)[0]["evidence_verified"] is True  # re-grounded, not copied
    assert second.segment_hashes == first.segment_hashes and second.authoritative


@pytest.mark.asyncio
async def test_changed_text_is_mined_again(model, store):
    await _mine([_event(0, TEXT)], store)
    await _mine([_event(0, TEXT + " And for tuples?")], store)
    assert model.calls["emit_units"] == 2


@pytest.mark.asyncio
async def test_only_the_changed_tail_segment_is_mined_after_a_resume(model, store):
    """Segmentation is prefix-stable: appending events leaves every earlier
    segment's text, and so its key, unchanged."""
    session = [_event(0, TEXT), _compaction(1), _event(2, "Then we switched to msgspec.")]
    await _mine(session, store)
    assert model.calls["emit_units"] == 2
    resumed = [*session, _event(3, "And benchmarked it.")]
    bundle = await _mine(resumed, store)
    assert model.calls["emit_units"] == 3
    assert (bundle.calls, bundle.cache_hits, bundle.segments) == (1, 1, 2)


def _patch_question(monkeypatch, part: str) -> None:
    if part == "model":
        monkeypatch.setattr(ext, "_model_and_transport", lambda: ("another-model", {}))
    elif part == "revision":
        real = ext.get_settings()
        monkeypatch.setattr(
            ext, "get_settings",
            lambda: real.model_copy(update={"claude_code_extraction_cache_revision": "2"}),
        )
    elif part == "system":
        monkeypatch.setattr(ext, "_SYSTEM_TEMPLATE", ext._SYSTEM_TEMPLATE + " Be brief.")
    elif part == "user_template":
        monkeypatch.setattr(ext, "_USER_TEMPLATE", "Mine this.\n" + ext._USER_TEMPLATE)
    elif part == "schema":
        schema = json.loads(json.dumps(ext._TOOL_PARAMETERS))
        schema["properties"]["extra"] = {"type": "string"}
        monkeypatch.setattr(ext, "_TOOL_PARAMETERS", schema)
    elif part == "max_tokens":
        monkeypatch.setattr(ext, "_EXTRACT_MAX_TOKENS", 4000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part", ["model", "revision", "system", "user_template", "schema", "max_tokens"]
)
async def test_any_change_to_the_question_is_a_new_key(model, store, monkeypatch, part):
    await _mine([_event(0, TEXT)], store)
    _patch_question(monkeypatch, part)
    await _mine([_event(0, TEXT)], store)
    assert model.calls["emit_units"] == 2, f"{part} change reused an answer"


@pytest.mark.asyncio
async def test_a_different_cwd_is_a_new_key(model, store):
    await _mine([_event(0, TEXT)], store, cwd="/work")
    await _mine([_event(0, TEXT)], store, cwd="/elsewhere")
    assert model.calls["emit_units"] == 2


@pytest.mark.asyncio
async def test_the_part_number_is_not_part_of_the_key(model, store):
    """A session that grows from 1 to 2 segments renumbers nothing it already had."""
    kwargs = dict(session_id="s1", events=[_event(0, TEXT)], cwd="/w", agent="claude_code",
                  drop_summaries=True, cache=_cache(store))
    await ext._extract_one(part=(1, 1), **kwargs)
    hit = await ext._extract_one(part=(1, 2), **kwargs)
    assert model.calls["emit_units"] == 1 and hit.cache_hits == 1


@pytest.mark.asyncio
async def test_a_session_never_reads_another_sessions_answers(model, store):
    await _mine([_event(0, TEXT)], store, session="s1")
    await _mine([_event(0, TEXT)], store, session="s2")
    assert model.calls["emit_units"] == 2


@pytest.mark.asyncio
async def test_a_declined_answer_is_never_stored(model, store):
    model.declines = True
    declined = await _mine([_event(0, TEXT)], store)
    assert not declined.authoritative and store.puts == 0
    model.declines = False
    again = await _mine([_event(0, TEXT)], store)
    assert again.calls == 1 and again.cache_hits == 0


@pytest.mark.asyncio
async def test_an_answer_that_does_not_build_is_never_stored(model, store):
    """`forced_tool_call` only checks for a dict; construction can still fail.
    Stored first, it would fail every later pass too."""
    model.answers["emit_units"] = {"qa": [{}]}
    failed = await _mine([_event(0, TEXT)], store)
    assert ext.ExtractionProblem.SEGMENT_FAILED in failed.problems
    assert store.puts == 0


@pytest.mark.asyncio
async def test_a_stored_answer_that_no_longer_builds_is_mined_fresh(model, store):
    await _mine([_event(0, TEXT)], store)
    (path,) = store.objects
    body = json.loads(store.objects[path])
    body["answer"] = {"qa": [{"unknown_required_shape": 1}]}
    store.objects[path] = json.dumps(body).encode()
    bundle = await _mine([_event(0, TEXT)], store)
    assert model.calls["emit_units"] == 2
    assert bundle.calls == 1 and bundle.cache_hits == 0 and _units(bundle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [b"not json", json.dumps({"v": 99, "answer": {}}).encode(),
     json.dumps({"v": 1, "fingerprint": "someone-else", "answer": {"qa": []}}).encode()],
)
async def test_an_unreadable_or_foreign_body_is_a_miss(model, store, body):
    await _mine([_event(0, TEXT)], store)
    (path,) = store.objects
    store.objects[path] = body
    await _mine([_event(0, TEXT)], store)
    assert model.calls["emit_units"] == 2


@pytest.mark.asyncio
async def test_a_failing_store_costs_one_attempt_per_pass(model, store):
    store.fail_get = StorageUnavailable("r2 down")
    session = [_event(0, TEXT), _compaction(1), _event(2, "second"), _compaction(3),
               _event(4, "third")]
    handle = _cache(store)
    bundle = await ext.extract_units_from_session(
        session_id="s1", events=session, cwd="/w", cache=handle
    )
    assert bundle.calls == 3 and bundle.authoritative
    assert store.gets == 1 and store.puts == 0 and handle.enabled is False


@pytest.mark.asyncio
async def test_a_slow_store_is_bounded(model, store, monkeypatch):
    monkeypatch.setattr(cache_mod, "_OP_DEADLINE_S", 0.05)
    store.get_delay = 5
    started = asyncio.get_running_loop().time()
    bundle = await _mine([_event(0, TEXT)], store)
    assert asyncio.get_running_loop().time() - started < 2
    assert bundle.calls == 1


@pytest.mark.asyncio
async def test_a_failing_write_does_not_fail_the_pass(model, store):
    store.fail_put = StorageUnavailable("r2 down")
    bundle = await _mine([_event(0, TEXT)], store)
    assert bundle.calls == 1 and bundle.authoritative and _units(bundle)


@pytest.mark.asyncio
async def test_the_switch_turns_the_cache_off(monkeypatch):
    real = cache_mod.get_settings()
    monkeypatch.setattr(
        cache_mod, "get_settings",
        lambda: real.model_copy(update={"claude_code_extraction_segment_cache": False}),
    )
    assert await cache_mod.SegmentCache.for_session(
        customer_id="c1", source="claude_code", session_id="s1"
    ) is None


@pytest.mark.asyncio
async def test_an_unknown_bucket_means_no_cache(monkeypatch):
    class NoBucket(Store):
        async def bucket_for(self, customer_id):
            raise StorageUnavailable("no bucket")

    monkeypatch.setattr(cache_mod, "_cache_store", lambda: NoBucket())
    assert await cache_mod.SegmentCache.for_session(
        customer_id="c1", source="claude_code", session_id="s1"
    ) is None


@pytest.mark.asyncio
async def test_the_cache_lives_under_the_sessions_raw_prefix(monkeypatch):
    """The source purge deletes raw/<source>/<customer>/: the cache goes with it."""
    store = Store()
    monkeypatch.setattr(cache_mod, "_cache_store", lambda: store)
    handle = await cache_mod.SegmentCache.for_session(
        customer_id="c1", source="codex", session_id="s9"
    )
    await handle.save("k", "fp", {"qa": []})
    assert list(store.objects) == ["bucket-c1/raw/codex/c1/s9/extraction-cache/k.json"]


def _decisions_session() -> list[dict[str, Any]]:
    return [_event(0, "We chose REST, then switched to gRPC.")]


@pytest.mark.asyncio
async def test_an_unchanged_decision_list_reuses_its_supersession_answer(model, store):
    model.answers["emit_units"] = {
        "decision": [
            {"question": "API?", "options_considered": ["REST", "gRPC"], "chosen": "REST",
             "rationale": "simple"},
            {"question": "API?", "options_considered": ["REST", "gRPC"], "chosen": "gRPC",
             "rationale": "streaming"},
        ]
    }
    first = await _mine(_decisions_session(), store)
    second = await _mine(_decisions_session(), store)
    assert model.calls["emit_supersessions"] == 1
    assert (first.calls, second.calls) == (2, 0)
    assert second.supersede_cached and not first.supersede_cached
    assert second.decision[1].supersedes == 0 and second.decision[0].superseded_by == 1


@pytest.mark.asyncio
async def test_a_changed_decision_list_asks_again(model, store):
    decision = {"question": "API?", "options_considered": ["REST", "gRPC"], "rationale": "r"}
    model.answers["emit_units"] = {
        "decision": [{**decision, "chosen": "REST"}, {**decision, "chosen": "gRPC"}]
    }
    await _mine(_decisions_session(), store)
    model.answers["emit_units"] = {
        "decision": [{**decision, "chosen": "REST"}, {**decision, "chosen": "GraphQL"}]
    }
    await _mine([_event(0, "We chose REST, then switched to GraphQL.")], store)
    assert model.calls["emit_supersessions"] == 2


@pytest.mark.asyncio
async def test_without_a_cache_nothing_changes(model):
    """The regression contract: no cache handle, exactly the calls there were before."""
    for _ in range(2):
        bundle = await ext.extract_units_from_session(
            session_id="s1", events=[_event(0, TEXT)], cwd="/w"
        )
        assert (bundle.calls, bundle.cache_hits) == (1, 0)
    assert model.calls["emit_units"] == 2

"""A session's events -> ATIF trajectory -> the same text the events render.

Two contracts:
  * the trajectory is valid ATIF v1.8 (Harbor's own models, vendored);
  * the Lines rebuilt from it equal the Lines the events render, line for line
    (text, line_no and the segmentation facts). That equality is what lets the
    ingest pass serve the trajectory's text without changing one byte of the
    index, the evidence spans or the extraction cache.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from engine.ingest.atif.build import build_trajectory
from engine.ingest.atif.lines import UnrenderableTrajectory, lines_from_trajectory
from engine.ingest.atif.models import Trajectory
from engine.shared.transcript_render import lines_from_events, render_lines_indexed

FIXTURES = Path(__file__).parent / "fixtures" / "session_protocol_v2"


def _ev(raw: Any, line_no: int | None) -> dict[str, Any]:
    ev: dict[str, Any] = {"raw": raw}
    if line_no is not None:
        ev["line_no"] = line_no
    return ev


def _user(content: Any, **extra: Any) -> dict[str, Any]:
    return {"type": "user", "message": {"role": "user", "content": content}, **extra}


def _assistant(content: Any, **extra: Any) -> dict[str, Any]:
    msg = {"role": "assistant", "content": content}
    msg.update(extra.pop("msg", {}))
    return {"type": "assistant", "message": msg, **extra}


def _round_trip(events: list[dict[str, Any]], agent: str = "claude_code") -> dict[str, Any]:
    built = build_trajectory(events, session_id="s-1", agent_name=agent)
    assert built.unparsed == 0
    Trajectory.model_validate(built.trajectory)
    assert lines_from_trajectory(built.trajectory) == lines_from_events(events)
    # Text and spans follow from equal Lines; asserted anyway because they are
    # what the index and the evidence check consume.
    assert render_lines_indexed(lines_from_trajectory(built.trajectory)) == render_lines_indexed(
        lines_from_events(events)
    )
    # It must survive the JSON round trip R2 puts it through.
    reloaded = json.loads(json.dumps(built.trajectory))
    assert lines_from_trajectory(reloaded) == lines_from_events(events)
    return built.trajectory


CASES: dict[str, list[dict[str, Any]]] = {
    "user string": [_ev(_user("how does auth work?"), 0)],
    "user string harness-only renders nothing": [
        _ev(_user("<system-reminder>x</system-reminder>"), 0)
    ],
    "user list text and an error result in one event": [
        _ev(_assistant([{"type": "tool_use", "id": "t1", "name": "Bash", "summary": "ls"}]), 0),
        _ev(
            _user(
                [
                    {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "result_bytes": 12},
                    {"type": "text", "text": "and then?"},
                ]
            ),
            1,
        ),
    ],
    "successful results render nothing but stay a user turn": [
        _ev(_assistant([{"type": "tool_use", "id": "t1", "name": "Read"}]), 0),
        _ev(_user([{"type": "tool_result", "tool_use_id": "t1"}]), 1),
    ],
    "orphan result": [
        _ev(_user([{"type": "tool_result", "tool_use_id": "gone", "is_error": True}]), 0),
    ],
    "result id shapes": [
        _ev(_user([{"type": "tool_result", "tool_use_id": "", "is_error": True}]), 0),
        _ev(_user([{"type": "tool_result", "is_error": True, "result_bytes": 0}]), 1),
        _ev(_user([{"type": "tool_result", "tool_use_id": 7, "is_error": 1, "result_bytes": True}]), 2),
    ],
    "assistant thinking text tool and a stop reason": [
        _ev(
            _assistant(
                [
                    {"type": "thinking", "thinking": "think\n\nhard"},
                    {"type": "thinking", "thinking": "   "},
                    {"type": "text", "text": "doing it"},
                    {"type": "tool_use", "id": "t9", "name": "Edit", "summary": "a.py",
                     "stats": {"added_lines": 3, "removed_lines": 1, "replace_all": True}},
                    {"type": "text", "text": "after the call"},
                ],
                msg={"stop_reason": "max_tokens", "model": "claude-opus"},
            ),
            0,
        ),
    ],
    "stop reason with nothing rendered is not noted": [
        _ev(_assistant([{"type": "thinking", "thinking": ""}], msg={"stop_reason": "refusal"}), 0),
    ],
    "assistant string content ignores stop": [
        _ev(_assistant("plain", msg={"stop_reason": "max_tokens"}), 0)
    ],
    "tool names and summaries of odd types": [
        _ev(_assistant([{"type": "tool_use", "name": ""}, {"type": "tool_use", "name": 5, "summary": 0},
                        {"type": "tool_use", "id": "x", "summary": {"k": 1}}]), 0),
    ],
    "consecutive assistant events form one step": [
        _ev(_assistant([{"type": "thinking", "thinking": "plan"}]), 0),
        _ev(_assistant([{"type": "text", "text": "answer"}]), 1),
        _ev(_assistant([{"type": "tool_use", "id": "a", "name": "Bash"}]), 2),
    ],
    "inference ids split and rejoin": [
        _ev(_assistant([{"type": "text", "text": "one"}], inference_id="m1"), 0),
        _ev(_assistant([{"type": "text", "text": "two"}], inference_id="m2"), 1),
        _ev({"type": "system", "subtype": "hook"}, 2),
        _ev(_assistant([{"type": "text", "text": "three"}], inference_id="m2"), 3),
    ],
    "system and other events": [
        _ev({"type": "system", "subtype": "compact_boundary"}, 0),
        _ev({"type": "system", "content": "note"}, 1),
        _ev({"type": "system", "subtype": "x", "content": "y"}, 2),
        _ev({"type": "system"}, 3),
        _ev({"type": "attachment", "attachment": {"type": "todo"}}, 4),
        _ev({"type": "queue-operation", "content": "queued text"}, 5),
        _ev({"content": "typeless"}, 6),
    ],
    "compaction summary in both shapes": [
        _ev(_user("the summary", isCompactSummary=True), 0),
        _ev(_user([{"type": "text", "text": "listed summary"}], isCompactSummary=True), 1),
    ],
    "ansi survives to the shared strip": [
        _ev(_user("\x1b[31mred\x1b[0m"), 0),
        _ev(_assistant([{"type": "text", "text": "\x1b\x1b[31m[0m"}]), 1),
    ],
    "events without line numbers and non-dict raw": [
        _ev(_user("first"), None),
        _ev("not a dict", 4),
        {"line_no": 5},
        _ev(_user("last"), None),
    ],
    "message not a dict": [
        _ev({"type": "user", "message": "x"}, 0),
        _ev({"type": "assistant", "message": None}, 1),
    ],
    "harness extras ride along": [
        _ev({"type": "system", "subtype": "session_start",
             "_codex_extras": {"cli_version": "0.99.0", "originator": "codex_cli"}}, 0),
        _ev(_user("hi", _pi_extras={"id": "e1", "parentId": None}), 1),
        _ev(_assistant([{"type": "text", "text": "yo"}], _kimi_extras={"step": 2}), 2),
    ],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip(name: str) -> None:
    _round_trip(CASES[name])


@pytest.mark.parametrize("source", ["claude_code", "codex", "pi"])
def test_protocol_v2_fixtures_round_trip(source: str) -> None:
    fixture = json.loads((FIXTURES / f"{source}.json").read_text())
    events = [e for batch in fixture["batches"] for e in batch.get("events") or []]
    assert events, "fixture carries no events"
    _round_trip(events, agent=source)


def test_structure_readers_see() -> None:
    trajectory = _round_trip(
        [
            _ev(_user("fix the test"), 0),
            _ev(_assistant([{"type": "thinking", "thinking": "look first"},
                            {"type": "tool_use", "id": "t1", "name": "Bash", "summary": "pytest"}],
                           msg={"model": "claude-x"}, inference_id="m1",
                           usage={"input_tokens": 10, "output_tokens": 5,
                                  "cache_read_input_tokens": 100,
                                  "cache_creation_input_tokens": 7}), 1),
            _ev(_user([{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                        "result_bytes": 42}]), 2),
            _ev(_assistant([{"type": "text", "text": "fixed"}], inference_id="m2"), 3),
        ]
    )
    user, call, answer = trajectory["steps"]
    assert (user["source"], user["message"]) == ("user", "fix the test")
    assert call["source"] == "agent" and call["model_name"] == "claude-x"
    assert call["reasoning_content"] == "look first"
    assert call["tool_calls"] == [
        {"tool_call_id": "t1", "function_name": "Bash", "arguments": {},
         "extra": {"summary": "pytest"}}
    ]
    assert call["observation"]["results"] == [
        {"source_call_id": "t1", "extra": {"is_error": True, "result_bytes": 42}}
    ]
    assert call["metrics"] == {"prompt_tokens": 117, "completion_tokens": 5,
                               "cached_tokens": 100,
                               "extra": {"cache_creation_input_tokens": 7}}
    assert answer["message"] == [{"type": "text", "text": "fixed"}]
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 117
    assert trajectory["agent"] == {"name": "claude_code", "version": "unknown",
                                   "model_name": "claude-x"}


def test_usage_counts_once_per_inference() -> None:
    usage = {"input_tokens": 1, "output_tokens": 2}
    trajectory = _round_trip(
        [
            _ev(_assistant([{"type": "text", "text": "a"}], inference_id="m", usage=usage), 0),
            _ev({"type": "system", "subtype": "hook"}, 1),
            _ev(_assistant([{"type": "text", "text": "b"}], inference_id="m",
                           usage={"input_tokens": 1, "output_tokens": 9}), 2),
        ]
    )
    first, _hook, second = trajectory["steps"]
    assert first["metrics"]["completion_tokens"] == 9, "last report wins, on the first step"
    assert "metrics" not in second and second["extra"]["continues_inference"] is True


def test_codex_cli_version_becomes_the_agent_version() -> None:
    trajectory = _round_trip(CASES["harness extras ride along"], agent="codex")
    assert trajectory["agent"]["version"] == "0.99.0"
    assert trajectory["steps"][0]["extra"]["codex_extras"][0]["originator"] == "codex_cli"


def test_an_event_that_cannot_map_costs_only_itself() -> None:
    events = [
        _ev(_user("before"), 0),
        # Legacy rendering raises on a non-string text block; the builder must not.
        _ev(_user([{"type": "text", "text": 5}]), 1),
        _ev(_user("after"), 2),
    ]
    built = build_trajectory(events, session_id="s", agent_name="claude_code")
    assert built.unparsed == 1
    Trajectory.model_validate(built.trajectory)
    subtypes = [s["extra"].get("subtype") for s in built.trajectory["steps"]]
    assert "unparsed" in subtypes
    texts = [line.text for line in lines_from_trajectory(built.trajectory)]
    assert texts[0] == "USER: before" and texts[2] == "USER: after"


def test_unknown_provenance_is_refused_not_guessed() -> None:
    built = build_trajectory([_ev(_user("x"), 0)], session_id="s", agent_name="claude_code")
    built.trajectory["extra"]["probe"]["format"] = "probe-render/999"
    with pytest.raises(UnrenderableTrajectory):
        lines_from_trajectory(built.trajectory)


# -- generated sessions ----------------------------------------------------------

_TEXTS = ["", " ", "hi", "a\n\nb", "<system-reminder>r</system-reminder> real",
          "Base directory for this skill: x", "\x1b[1mbold\x1b[0m", "<task-notification>t</task-notification>"]


def _random_block(rng: random.Random, role: str, ids: list[str]) -> Any:
    kind = rng.choice(["text", "thinking", "tool_use", "tool_result", "image", "weird", None])
    if kind == "text":
        return {"type": "text", "text": rng.choice(_TEXTS)}
    if kind == "thinking" and role == "assistant":
        return {"type": "thinking", "thinking": rng.choice(_TEXTS)}
    if kind == "tool_use" and role == "assistant":
        tid = rng.choice([f"t{rng.randint(0, 9)}", "", None])
        if tid:
            ids.append(tid)
        block = {"type": "tool_use", "name": rng.choice(["Bash", "", None, "Edit"])}
        if tid is not None:
            block["id"] = tid
        if rng.random() < 0.5:
            block["summary"] = rng.choice(["ls", "", "x — y"])
        if rng.random() < 0.3:
            block["stats"] = {"added_lines": rng.choice([1, -5, "3", 10**9]), "removed_lines": 2}
        return block
    if kind == "tool_result" and role == "user":
        block = {"type": "tool_result",
                 "tool_use_id": rng.choice((ids or ["zz"]) + ["", "missing", None])}
        if rng.random() < 0.5:
            block["is_error"] = rng.choice([True, False, 1, 0])
        if rng.random() < 0.5:
            block["result_bytes"] = rng.choice([0, 17, -1, None])
        return block
    if kind == "image":
        return {"type": "image"}
    if kind == "weird":
        return rng.choice(["str-block", 3, {"type": "unknown_block", "block_type": "x"}])
    return {"no": "type"}


def _random_event(rng: random.Random, ids: list[str]) -> Any:
    kind = rng.choice(["user", "user", "assistant", "assistant", "assistant", "system", "other"])
    extra: dict[str, Any] = {}
    if rng.random() < 0.2:
        extra["timestamp"] = rng.choice(["2026-10-01T00:00:00Z", "yesterday", 5])
    if kind == "user":
        if rng.random() < 0.3:
            return _user(rng.choice(_TEXTS), isCompactSummary=rng.random() < 0.2, **extra)
        blocks = [_random_block(rng, "user", ids) for _ in range(rng.randint(0, 4))]
        return _user(blocks, isCompactSummary=rng.random() < 0.1, **extra)
    if kind == "assistant":
        if rng.random() < 0.15:
            return _assistant(rng.choice(_TEXTS), **extra)
        blocks = [_random_block(rng, "assistant", ids) for _ in range(rng.randint(0, 4))]
        if rng.random() < 0.4:
            extra["inference_id"] = rng.choice(["m1", "m2", "m3"])
        msg = {"stop_reason": rng.choice([None, "end_turn", "tool_use", "max_tokens", "refusal"])}
        if rng.random() < 0.3:
            msg["model"] = "m"
        return _assistant(blocks, msg=msg, **extra)
    if kind == "system":
        return {"type": "system", "subtype": rng.choice([None, "", "compact_boundary", "hook"]),
                "content": rng.choice([None, "", "text", 3]), **extra}
    return {"type": rng.choice(["attachment", "queue-operation", None]),
            "content": rng.choice([None, "", "body"]), **extra}


@pytest.mark.parametrize("seed", range(300))
def test_generated_sessions_round_trip(seed: int) -> None:
    rng = random.Random(seed)
    ids: list[str] = []
    events = []
    for n in range(rng.randint(1, 30)):
        line_no = n if rng.random() < 0.95 else None
        events.append(_ev(_random_event(rng, ids), line_no))
    _round_trip(events)


# -- what a reader can be served ---------------------------------------------------


def test_client_supplied_values_reach_a_reader_only_as_short_scalars() -> None:
    """Upload is the trust boundary: a patched or compromised tap posts whatever
    it likes, and GET /trajectory serves these fields."""
    big = "x" * 5000
    events = [
        _ev({"type": "system", "subtype": {"nested": "subtype"},
             "_codex_extras": {"action": {"command": ["sh"], "env": {"TOKEN": "s3cr3t"}},
                               "cli_version": "0.99.0", "note": big, "n": 3, "ok": True}}, 0),
        _ev(_assistant([{"type": "tool_use", "id": "t", "name": "Edit", "summary": {"k": "v"},
                         "stats": {"added_lines": "3", "removed_lines": 2, "replace_all": "yes",
                                   "content": big}}],
                       msg={"stop_reason": {"why": "x"}}), 1),
        _ev(_user([{"type": "tool_result", "tool_use_id": ["a"], "is_error": True,
                    "result_bytes": {"n": 1}}]), 2),
    ]
    trajectory = _round_trip(events, agent="codex")
    system, agent, orphan = trajectory["steps"]
    [extras] = system["extra"]["codex_extras"]
    assert extras == {"cli_version": "0.99.0", "note": "x" * 256, "n": 3, "ok": True}
    assert isinstance(system["extra"]["subtype"], str)
    [call] = agent["tool_calls"]
    assert call["arguments"] == {}
    assert call["extra"] == {"summary": "{'k': 'v'}", "stats": {"removed_lines": 2}}
    assert isinstance(agent["extra"]["stop_reason"], str)
    [result] = orphan["observation"]["results"]
    assert result["extra"] == {"is_error": True, "tool_use_id": "['a']"}


def test_a_long_inference_builds_in_linear_time() -> None:
    import time

    events = [_ev(_assistant([{"type": "thinking", "thinking": "t" * 1000}]), i)
              for i in range(20_000)]
    started = time.perf_counter()
    built = build_trajectory(events, session_id="s", agent_name="claude_code")
    assert time.perf_counter() - started < 5
    [step] = built.trajectory["steps"]
    assert len(step["reasoning_content"]) == 20_000 * 1000 + 19_999 * 2


def test_a_long_unknown_event_type_renders_byte_identically() -> None:
    events = [_ev({"type": "Z" * 100, "content": "hello"}, 0)]
    built = build_trajectory(events, session_id="s", agent_name="claude_code")
    assert lines_from_trajectory(built.trajectory) == lines_from_events(events)


def test_unparsed_events_are_reported_to_the_caller_not_logged_by_the_build() -> None:
    from structlog.testing import capture_logs

    import engine.ingest.atif.build_reference as build_mod

    def broken(self: Any, line: int, raw: dict[str, Any]) -> None:
        raise KeyError("x")

    original = build_mod._Builder._map
    build_mod._Builder._map = broken
    try:
        with capture_logs() as logs:
            built = build_trajectory([_ev(_user("hi"), 0)], session_id="s", agent_name="claude_code")
    finally:
        build_mod._Builder._map = original
    assert built.unparsed == 1 and built.unparsed_events == [(0, "user", "KeyError")]
    assert logs == []


def test_a_command_the_researcher_typed_is_its_own_step_never_the_models() -> None:
    """pi's `!git status` (probe-events/1 `origin: user_shell`, tap 0.9.11): an
    assistant-shaped tool call with no inference id. It must not join the
    model's open step before it, and the model's next reply must not join it."""
    shell = {"origin": "user_shell"}
    events = [
        _ev(_user("look at the repo"), 0),
        _ev(_assistant([{"type": "text", "text": "sure"}], inference_id="m1"), 1),
        _ev(_assistant([{"type": "tool_use", "id": "bash-b1", "name": "bash",
                         "summary": "git status --short"}], **shell), 2),
        _ev(_user([{"type": "tool_result", "tool_use_id": "bash-b1", "result_bytes": 12}],
                  **shell), 3),
        _ev(_assistant([{"type": "text", "text": "clean tree"}]), 4),
    ]
    trajectory = _round_trip(events, agent="pi")
    agent_steps = [s for s in trajectory["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 3
    model, typed, reply = agent_steps
    assert "tool_calls" not in model and model["extra"].get("origin") is None
    assert typed["extra"]["origin"] == "user_shell"
    assert [c["function_name"] for c in typed["tool_calls"]] == ["bash"]
    [result] = typed["observation"]["results"]
    assert result["source_call_id"] == "bash-b1"
    assert reply["extra"].get("origin") is None and "tool_calls" not in reply


def test_a_kimi_shell_command_is_the_researchers_turn_marked_as_typed() -> None:
    """Kimi's shell mode (probe-events/1 `origin: user_shell`, tap 0.9.11) sends
    the command as a `user` event and its output as a `shell_output` system
    event carrying only its size."""
    events = [
        _ev(_user("check the history"), 0),
        _ev(_user([{"type": "text", "text": "<bash-input>git log --oneline -3</bash-input>"}],
                  origin="user_shell"), 1),
        _ev({"type": "system", "subtype": "shell_output", "_kimi_extras": {"result_bytes": 39}}, 2),
    ]
    trajectory = _round_trip(events, agent="kimi_code")
    prompt, typed = [s for s in trajectory["steps"] if s["source"] == "user"]
    assert prompt["extra"].get("origin") is None
    assert typed["extra"]["origin"] == "user_shell"

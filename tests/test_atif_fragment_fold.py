"""fragment + fold == the frozen builder (engine/ingest/atif).

`fold(map(fragment, events))` must build, for every fixture and case the
builder's own suite has, the document `build_reference.build_trajectory` builds
-- the same `BuildResult`, provenance included, down to the JSON bytes -- and
must do so for every prefix of the events (what a live pass folds mid-session).
A fragment must be a pure function of its event: JSON round-trip stable,
deterministic, carrying the event's index Line exactly as the event renderer
gives it, and importing nothing a client could not vendor. fold must treat
fragments as untrusted: wrong types, huge values, unknown keys, missing fields,
duplicates and junk are folded into a defined document, never an exception.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import orjson
import pytest
from structlog.testing import capture_logs

from engine.ingest.atif import build_reference
from engine.ingest.atif import fold as fold_mod
from engine.ingest.atif.build import Builder, build_and_render, build_trajectory, builder_for
from engine.ingest.atif.fold import Unreadable, fold, line_of
from engine.ingest.atif.fragment import (
    FRAGMENT_VERSION,
    Kind,
    PieceType,
    fragment,
    fragment_line,
)
from engine.ingest.atif.lines import lines_from_trajectory
from engine.ingest.atif.models import Trajectory
from engine.shared.transcript_render import line_from_event, lines_from_events
from tests.test_atif_build import CASES as BUILD_CASES
from tests.test_atif_build import FIXTURES as PROTOCOL_V2
from tests.test_atif_build import _assistant, _ev, _random_event, _user
from tests.test_atif_ingest import EVENTS as INGEST_EVENTS
from tests.test_probe_events_project import GOLDENS, _events

ROOT = Path(__file__).resolve().parents[1]

# -- every case the builder is tested on, plus the joins fold has to get right --

EXTRA_CASES: dict[str, list[dict[str, Any]]] = {
    "structure readers see": [
        _ev(_user("fix the test"), 0),
        _ev(
            _assistant(
                [
                    {"type": "thinking", "thinking": "look first"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "summary": "pytest"},
                ],
                msg={"model": "claude-x"},
                inference_id="m1",
                usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 7,
                },
            ),
            1,
        ),
        _ev(
            _user(
                [{"type": "tool_result", "tool_use_id": "t1", "is_error": True, "result_bytes": 42}]
            ),
            2,
        ),
        _ev(_assistant([{"type": "text", "text": "fixed"}], inference_id="m2"), 3),
    ],
    "usage counts once per inference, last report wins": [
        _ev(
            _assistant(
                [{"type": "text", "text": "a"}],
                inference_id="m",
                usage={"input_tokens": 1, "output_tokens": 2},
            ),
            0,
        ),
        _ev({"type": "system", "subtype": "hook"}, 1),
        _ev(
            _assistant(
                [{"type": "text", "text": "b"}],
                inference_id="m",
                usage={"input_tokens": 1, "output_tokens": 9},
            ),
            2,
        ),
    ],
    "usage first reported on the continuation": [
        _ev(_assistant([{"type": "text", "text": "a"}], inference_id="m"), 0),
        _ev({"type": "system", "subtype": "hook"}, 1),
        _ev(
            _assistant(
                [{"type": "text", "text": "b"}], inference_id="m", usage={"input_tokens": 3}
            ),
            2,
        ),
        _ev(
            _assistant(
                [{"type": "text", "text": "c"}],
                inference_id="m",
                usage={"input_tokens": 4, "output_tokens": 1},
            ),
            3,
        ),
    ],
    "message.usage when there is no top-level usage": [
        _ev(
            _assistant(
                [{"type": "text", "text": "a"}],
                msg={"usage": {"input_tokens": 5, "output_tokens": True}},
            ),
            0,
        ),
        _ev(
            _assistant(
                [{"type": "text", "text": "b"}],
                usage="nonsense",
                msg={"usage": {"output_tokens": 2}},
            ),
            1,
        ),
    ],
    "a late inference id is adopted by the open step": [
        _ev(_assistant([{"type": "thinking", "thinking": "plan"}]), 0),
        _ev(_assistant([{"type": "text", "text": "answer"}], inference_id="m1"), 1),
        _ev(_assistant([{"type": "text", "text": "more"}], inference_id="m1"), 2),
        _ev(_assistant([{"type": "text", "text": "next"}], inference_id="m2"), 3),
    ],
    "an assistant event without a message leaves the call open": [
        _ev(_assistant([{"type": "text", "text": "one"}], inference_id="m1"), 0),
        _ev({"type": "assistant", "message": "not an object"}, 1),
        _ev("not a dict", 2),
        _ev(_assistant([{"type": "text", "text": "two"}], inference_id="m1"), 3),
    ],
    "a user event without a message still ends the call": [
        _ev(_assistant([{"type": "text", "text": "one"}], inference_id="m1"), 0),
        _ev({"type": "user", "message": None}, 1),
        _ev(_assistant([{"type": "text", "text": "two"}], inference_id="m1"), 2),
    ],
    "results before, after and without their call": [
        _ev(_user([{"type": "tool_result", "tool_use_id": "late", "is_error": True}]), 0),
        _ev(
            _assistant(
                [
                    {"type": "tool_use", "id": "late", "name": "Bash"},
                    {"type": "tool_use", "name": "Read"},
                    {"type": "tool_use", "id": "dup", "name": "A"},
                ]
            ),
            1,
        ),
        _ev(_assistant([{"type": "tool_use", "id": "dup", "name": "B"}], inference_id="m9"), 2),
        _ev(
            _user(
                [
                    {"type": "tool_result", "tool_use_id": "late", "result_bytes": 3},
                    {"type": "tool_result", "tool_use_id": "dup", "is_error": True},
                    {"type": "tool_result", "tool_use_id": None, "is_error": True},
                    {"type": "tool_result", "tool_use_id": {"odd": 1}, "is_error": True},
                    {"type": "tool_result", "tool_use_id": "nowhere", "is_error": True},
                    {"type": "text", "text": "and?"},
                ]
            ),
            3,
        ),
    ],
    "a result in the same event as a second orphan reuses the orphan step": [
        _ev(
            _user(
                [
                    {"type": "tool_result", "tool_use_id": "a", "is_error": True},
                    {"type": "tool_result", "tool_use_id": "b", "is_error": True},
                ],
                timestamp="2026-10-01T00:00:00Z",
            ),
            0,
        ),
    ],
    "an event that cannot map keeps what it mapped before": [
        _ev(_assistant([{"type": "tool_use", "id": "t1", "name": "Bash"}], inference_id="m1"), 0),
        _ev(
            _user(
                [
                    {"type": "tool_result", "tool_use_id": "t1", "is_error": True},
                    {"type": "text", "text": "first"},
                    {"type": "image"},
                    {"type": "text", "text": 5},
                    {"type": "text", "text": "never"},
                ]
            ),
            1,
        ),
        _ev(_assistant([{"type": "text", "text": "after"}], inference_id="m1"), 2),
        _ev(_user([{"type": "text", "text": ["x"]}]), 3),
    ],
    "dropped blocks are noted only beside text": [
        _ev(
            _user(
                [
                    {"type": "image"},
                    {"type": "text", "text": "look"},
                    *({"type": f"block-{i}" + "x" * 80} for i in range(40)),
                ]
            ),
            0,
        ),
        _ev(_user([{"type": "image"}, {"type": "tool_result", "tool_use_id": "q"}]), 1),
    ],
    "client supplied values reach a reader only as short scalars": [
        _ev(
            {
                "type": "system",
                "subtype": {"nested": "subtype"},
                "_codex_extras": {
                    "action": {"command": ["sh"], "env": {"TOKEN": "s3cr3t"}},
                    "cli_version": "0.99.0",
                    "note": "x" * 5000,
                    "n": 3,
                    "ok": True,
                    "f": 1.5,
                    "": "empty",
                    **{f"k{i}": i for i in range(80)},
                },
            },
            0,
        ),
        _ev(
            _assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": "Edit",
                        "summary": {"k": "v"},
                        "stats": {
                            "added_lines": "3",
                            "removed_lines": 2,
                            "replace_all": "yes",
                            "content": "x" * 5000,
                        },
                    }
                ],
                msg={"stop_reason": {"why": "x"}, "model": "m" * 400},
                inference_id="i" * 400,
            ),
            1,
        ),
        _ev(
            _user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": ["a"],
                        "is_error": True,
                        "result_bytes": {"n": 1},
                        "content": "TOOL-OUTPUT",
                    }
                ]
            ),
            2,
        ),
        _ev({"type": "attachment", "attachment": {"type": "t" * 100}, "content": "c"}, 3),
        _ev({"type": "system", "_codex_extras": {"cli_version": "v" * 100}}, 4),
    ],
    "commands the researcher typed": [
        _ev(_user("look at the repo"), 0),
        _ev(_assistant([{"type": "text", "text": "sure"}], inference_id="m1"), 1),
        _ev(
            _assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "bash-b1",
                        "name": "bash",
                        "summary": "git status --short",
                    }
                ],
                origin="user_shell",
            ),
            2,
        ),
        _ev(
            _user(
                [{"type": "tool_result", "tool_use_id": "bash-b1", "result_bytes": 12}],
                origin="user_shell",
            ),
            3,
        ),
        _ev(_assistant([{"type": "text", "text": "clean tree"}]), 4),
        _ev(
            _user(
                [{"type": "text", "text": "<bash-input>git log</bash-input>"}], origin="user_shell"
            ),
            5,
        ),
        _ev(_user("a summary", isCompactSummary=True, origin="user_shell"), 6),
        _ev({"type": "system", "subtype": "shell_output", "_kimi_extras": {"result_bytes": 39}}, 7),
    ],
    "a long unknown event type": [_ev({"type": "Z" * 100, "content": "hello"}, 0)],
    "duplicate and missing ordinals": [
        _ev(_user("one"), 3),
        _ev(_user("one again"), 3),
        _ev(_assistant([{"type": "text", "text": "x"}]), True),
        _ev(_user("unnumbered"), None),
        {"raw": {"type": "system", "content": "no line_no key"}},
    ],
    "timestamps that do and do not parse": [
        _ev(_user("a", timestamp="2026-10-01T00:00:00Z"), 0),
        _ev(_assistant([{"type": "text", "text": "b"}], timestamp="yesterday"), 1),
        _ev(_assistant([{"type": "text", "text": "c"}], timestamp="2026-10-01T00:00:01+00:00"), 2),
        _ev({"type": "system", "timestamp": 5}, 3),
    ],
    "nothing readable at all": [_ev("x", 0), {"line_no": 1}],
    "empty": [],
}


def _generated(seed: int) -> list[dict[str, Any]]:
    """test_atif_build's generated session for `seed`."""
    rng = random.Random(seed)
    ids: list[str] = []
    events = []
    for n in range(rng.randint(1, 30)):
        line_no = n if rng.random() < 0.95 else None
        events.append(_ev(_random_event(rng, ids), line_no))
    return events


def _protocol_v2(source: str) -> list[dict[str, Any]]:
    fixture = json.loads((PROTOCOL_V2 / f"{source}.json").read_text())
    return [e for batch in fixture["batches"] for e in batch.get("events") or []]


def _golden(path: Path) -> list[dict[str, Any]]:
    """A tap's probe-events/1 golden as the engine stores it: event ordinals as line_no."""
    return [{"line_no": n, "raw": raw} for n, raw in enumerate(_events(path))]


ALL_CASES: dict[str, tuple[list[dict[str, Any]], str]] = {
    **{f"build:{name}": (events, "claude_code") for name, events in BUILD_CASES.items()},
    **{f"extra:{name}": (events, "claude_code") for name, events in EXTRA_CASES.items()},
    **{f"protocol_v2:{s}": (_protocol_v2(s), s) for s in ("claude_code", "codex", "pi")},
    **{f"golden:{p.name}": (_golden(p), p.name.split(".")[0]) for p in GOLDENS},
    "ingest:EVENTS": (INGEST_EVENTS, "claude_code"),
    **{f"generated:{seed}": (_generated(seed), "claude_code") for seed in range(300)},
}
NAMES = sorted(ALL_CASES)


def _data(built: Any) -> dict[str, Any]:
    return {
        "trajectory": built.trajectory,
        "unparsed": built.unparsed,
        "unparsed_events": [tuple(e) for e in built.unparsed_events],
    }


def _fragments(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [fragment(ev) for ev in events if isinstance(ev, dict)]


def _assert_same(events: list[dict[str, Any]], agent: str) -> None:
    reference = build_reference.build_trajectory(events, session_id="s-1", agent_name=agent)
    folded = fold(_fragments(events), session_id="s-1", agent_name=agent)
    assert _data(folded) == _data(reference)
    # Byte for byte, key order included: a stored copy must not change at all.
    assert json.dumps(folded.trajectory) == json.dumps(reference.trajectory)


def test_every_source_of_cases_is_present() -> None:
    sources = {name.split(":")[0] for name in ALL_CASES}
    assert sources == {"build", "extra", "protocol_v2", "golden", "ingest", "generated"}
    assert len(GOLDENS) == 4, "the four harness goldens (claude_code, codex, kimi_code, pi)"
    assert all(ALL_CASES[f"golden:{p.name}"][0] for p in GOLDENS)


# -- equality with the frozen builder -------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_fold_of_fragments_is_the_reference_document(name: str) -> None:
    events, agent = ALL_CASES[name]
    _assert_same(events, agent)


@pytest.mark.parametrize("name", NAMES)
def test_every_batch_prefix_folds_to_the_reference_prefix(name: str) -> None:
    """A live pass folds the fragments of the batches so far. Folding is a
    function of the fragment list alone, so the fold of the first k fragments
    -- however the batches fell -- is the reference build of the first k events."""
    events, agent = ALL_CASES[name]
    events = [ev for ev in events if isinstance(ev, dict)]
    fragments = _fragments(events)
    for k in range(len(events) + 1):
        reference = build_reference.build_trajectory(events[:k], session_id="s", agent_name=agent)
        assert _data(fold(fragments[:k], session_id="s", agent_name=agent)) == _data(reference), k


@pytest.mark.parametrize("name", NAMES)
def test_the_build_api_serves_either_builder(name: str) -> None:
    events, agent = ALL_CASES[name]
    by_builder = {
        b: _data(build_trajectory(events, session_id="s", agent_name=agent, builder=b))
        for b in (Builder.REFERENCE, Builder.FOLD, "fold", "reference")
    }
    assert len({json.dumps(v, default=str) for v in by_builder.values()}) == 1


# -- what a fragment is -------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_fragments_survive_json_and_are_deterministic(name: str) -> None:
    events, agent = ALL_CASES[name]
    events = [ev for ev in events if isinstance(ev, dict)]
    fragments = _fragments(events)
    for ev, frag in zip(events, fragments, strict=True):
        assert json.loads(json.dumps(frag)) == frag
        assert orjson.loads(orjson.dumps(frag)) == frag
        again = fragment(copy.deepcopy(ev))
        assert again == frag and json.dumps(again) == json.dumps(frag)
        assert frag["v"] == FRAGMENT_VERSION and frag["kind"] in {k.value for k in Kind}
    # What the server folds is what came over the wire.
    wire = orjson.loads(orjson.dumps(fragments))
    assert _data(fold(wire, session_id="s", agent_name=agent)) == _data(
        fold(fragments, session_id="s", agent_name=agent)
    )


@pytest.mark.parametrize("name", NAMES)
def test_fragment_lines_are_the_event_lines(name: str) -> None:
    events, _agent = ALL_CASES[name]
    events = [ev for ev in events if isinstance(ev, dict)]
    renderable = []
    for ev in events:
        frag = fragment(ev)
        try:
            expected = line_from_event(ev)
        except Exception:
            # The renderer's own gap: the flags survive, the text is empty and named.
            assert frag["line"]["text"] == "" and frag["line_error"]
            continue
        assert "line_error" not in frag
        assert fragment_line(frag) == expected
        assert line_of(frag) == expected, "fold's reader sees the same Line"
        renderable.append(ev)
    if len(renderable) == len(events):
        assert [fragment_line(f) for f in _fragments(events)] == lines_from_events(events)


def test_an_event_the_renderer_cannot_render_keeps_its_flags_and_says_so() -> None:
    ev = _ev(_user([{"type": "text", "text": "kept"}, {"type": "text", "text": 5}]), 7)
    with pytest.raises(TypeError):
        line_from_event(ev)
    frag = fragment(ev)
    assert frag["line"] == {
        "line_no": 7,
        "text": "",
        "user_turn": True,
        "compact_boundary": False,
        "compact_summary": False,
    }
    assert frag["line_error"] == "TypeError" and frag["error"] == "TypeError"
    assert frag["parts"] == [{"type": "text", "seq": 0, "text": "kept"}]


def test_a_fragment_carries_no_tool_input_or_output() -> None:
    ev = _ev(
        _assistant(
            [
                {
                    "type": "tool_use",
                    "id": "t",
                    "name": "Bash",
                    "summary": "ls",
                    "input": {"command": "cat SECRET-INPUT"},
                }
            ]
        ),
        0,
    )
    result = _ev(
        _user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": "SECRET-OUTPUT",
                    "result_bytes": 13,
                }
            ],
            toolUseResult={"stdout": "SECRET-OUTPUT"},
        ),
        1,
    )
    wire = json.dumps([fragment(ev), fragment(result)])
    assert "SECRET" not in wire


def _keys(fragments: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for frag in fragments:
        keys |= set(frag) | set(frag["line"])
        for piece in frag.get("parts") or []:
            keys |= set(piece) | set(piece.get("stats") or {})
        keys |= set(frag.get("stop") or {}) | set(frag.get("usage") or {})
        keys |= set(frag.get("extras") or {}) | set(frag.get("lineage") or {})
    return keys


def test_every_key_a_fragment_carries_is_documented() -> None:
    import engine.ingest.atif.fragment as fragment_mod

    emitted = set()
    for events, _agent in ALL_CASES.values():
        emitted |= _keys(_fragments(events))
    doc = fragment_mod.__doc__ or ""
    undocumented = sorted(k for k in emitted if k not in doc)
    assert not undocumented, f"document these in fragment.py's schema: {undocumented}"
    assert {p.value for p in PieceType} <= {
        p["type"]
        for events, _a in ALL_CASES.values()
        for f in _fragments(events)
        for p in f.get("parts") or []
    }


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_fragment_imports_only_what_a_client_can_vendor() -> None:
    """fragment.py goes into the tap with transcript_render.py: stdlib and that."""
    allowed = {"engine.shared.transcript_render"}
    for path in (
        ROOT / "engine/ingest/atif/fragment.py",
        ROOT / "engine/shared/transcript_render.py",
    ):
        foreign = {
            name
            for name in _imports(path)
            if name not in allowed and name.split(".")[0] not in sys.stdlib_module_names
        }
        assert not foreign, f"{path.name} imports {sorted(foreign)}"


# -- fold reads fragments as untrusted ------------------------------------------------


def _fold(fragments: list[Any]) -> Any:
    built = fold(fragments, session_id="s", agent_name="claude_code")
    # Whatever came in, the document is valid ATIF, renders, and serialises.
    Trajectory.model_validate(built.trajectory)
    lines_from_trajectory(built.trajectory)
    orjson.dumps(built.trajectory)
    return built


def _valid() -> list[dict[str, Any]]:
    return _fragments(EXTRA_CASES["structure readers see"])


def _unreadable_steps(built: Any) -> list[dict[str, Any]]:
    return [s for s in built.trajectory["steps"] if s["extra"].get("subtype") == "unparsed"]


@pytest.mark.parametrize(
    "junk",
    [None, "x", 5, [], [1], True, {}, {"v": 1}, {"line": {"line_no": 0, "text": "t"}}],
    ids=repr,
)
def test_a_fragment_fold_cannot_read_is_one_line_and_one_unparsed_step(junk: Any) -> None:
    good = _valid()
    built = _fold([good[0], junk, good[1]])
    assert built.unparsed == 1
    [unreadable] = _unreadable_steps(built)
    assert unreadable["extra"]["event_type"] is None
    assert len(built.trajectory["extra"]["probe"]["lines"]) == 3, "one line per fragment"
    [(_line_no, ev_type, error)] = built.unparsed_events
    assert ev_type is None and error == Unreadable.INVALID


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("kind",), "robot"),
        (("kind",), 3),
        (("kind",), ["user"]),
        (("v",), "1"),
        (("v",), True),
        (("line",), "USER: x"),
        (("line", "text"), 5),
        (("line", "line_no"), "0"),
        (("line", "user_turn"), "yes"),
        (("parts",), "text"),
        (("parts",), {"type": "text"}),
        (("parts", 0), "piece"),
        (("parts", 0, "type"), "image"),
        (("parts", 0, "type"), None),
        (("parts", 0, "seq"), -1),
        (("parts", 0, "seq"), "0"),
        (("parts", 0, "seq"), True),
        (("parts", 0, "seq"), 2**40),
        (("parts", 0, "text"), None),
        (("parts", 0, "text"), 5),
        (("parts", 1, "name"), None),
        (("parts", 1, "id"), 5),
        (("parts", 1, "summary"), ["x"]),
        (("stop",), "max_tokens"),
        (("stop",), {"reason": "max_tokens"}),
        (("inference_id",), 5),
        (("model",), {"m": 1}),
        (("error",), 5),
    ],
    ids=str,
)
def test_a_field_of_the_wrong_type_makes_the_fragment_unreadable(path: tuple, value: Any) -> None:
    fragments = _valid()
    target = copy.deepcopy(fragments[1])  # the assistant event: thinking + a tool call
    node = target
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    fragments[1] = target
    built = _fold(fragments)
    assert built.unparsed == 1 and len(_unreadable_steps(built)) == 1
    assert not any(
        s["source"] == "agent" and s["extra"].get("inference_id") == "m1"
        for s in built.trajectory["steps"]
    ), "nothing of it is mapped"


def test_an_unknown_version_is_unreadable_and_named() -> None:
    fragments = _valid()
    fragments[0] = {**fragments[0], "v": FRAGMENT_VERSION + 1}
    built = _fold(fragments)
    assert built.unparsed_events == [(0, None, Unreadable.VERSION)]


def test_unknown_keys_are_ignored_and_never_copied() -> None:
    clean = _valid()
    noisy = copy.deepcopy(clean)
    for frag in noisy:
        frag.update({"content": "SECRET-1", "arguments": {"cmd": "SECRET-2"}, "steps": [1]})
        frag["line"]["extra"] = "SECRET-3"
        for piece in frag.get("parts") or []:
            piece.update(
                {
                    "content": "SECRET-4",
                    "arguments": {"a": "SECRET-5"},
                    "input": "SECRET-6",
                    "source_call_id": "SECRET-7",
                }
            )
        if "usage" in frag:
            frag["usage"]["secret_tokens"] = "SECRET-8"
    noisy[0]["extras"] = {"other_extras": {"k": "SECRET-9"}}
    assert _data(_fold(noisy)) == _data(_fold(clean))
    assert "SECRET" not in json.dumps(_fold(noisy).trajectory)


def test_every_bound_the_builder_applies_is_applied_again() -> None:
    huge = "x" * 1_000_000
    hostile = [
        {
            "v": 1,
            "line": {"line_no": 0, "text": ""},
            "kind": "system",
            "subtype": "s",
            "agent_version": huge,
            "extras": {
                "codex_extras": {
                    "k" * 500: huge,
                    "nested": {"a": 1},
                    "list": [1],
                    **{f"k{i}": i for i in range(100)},
                }
            },
        },
        {
            "v": 1,
            "line": {"line_no": 1, "text": ""},
            "kind": "assistant",
            "inference_id": huge,
            "model": huge,
            "error": huge,
            "usage": {
                "input_tokens": -5,
                "output_tokens": "9",
                "cached": 3,
                "cache_read_input_tokens": True,
            },
            "parts": [
                {
                    "type": "tool_call",
                    "seq": 0,
                    "name": "Edit",
                    "id": "t",
                    "stats": {
                        "added_lines": "3",
                        "removed_lines": 2,
                        "replace_all": "yes",
                        "content": huge,
                    },
                }
            ],
        },
        {
            "v": 1,
            "line": {"line_no": 2, "text": ""},
            "kind": "user",
            "dropped_blocks": [huge] * 100,
            "parts": [
                {"type": "text", "seq": 0, "text": "hi"},
                {
                    "type": "tool_result",
                    "seq": 1,
                    "call_id": "t",
                    "is_error": "yes",
                    "result_bytes": "12",
                    "content": huge,
                },
            ],
        },
        {
            "v": 1,
            "line": {"line_no": 3, "text": ""},
            "kind": "other",
            "event_type": "e",
            "attachment_type": huge,
            "timestamp": "not a time",
        },
    ]
    built = _fold(hostile)
    system, agent, _unparsed, prompt, other = built.trajectory["steps"]
    [extras] = system["extra"]["codex_extras"]
    assert len(extras) == 64 and all(len(k) <= 64 for k in extras)
    assert all(len(v) <= 256 for v in extras.values() if isinstance(v, str))
    assert "nested" not in extras and "list" not in extras
    assert built.trajectory["agent"]["version"] == "x" * 64
    assert len(agent["extra"]["inference_id"]) == 256 and len(agent["model_name"]) == 256
    assert "metrics" not in agent, "no usable count"
    [call] = agent["tool_calls"]
    assert call["arguments"] == {} and call["extra"] == {"stats": {"removed_lines": 2}}
    [result] = agent["observation"]["results"]
    assert result == {"source_call_id": "t", "extra": {"is_error": True}}
    assert built.unparsed_events == [(1, "assistant", "x" * 64)]
    assert prompt["extra"]["dropped_blocks"] == ["x" * 64] * 32
    assert other["extra"]["attachment_type"] == "x" * 64 and "timestamp" not in other


def test_missing_fields_are_absent_values_not_errors() -> None:
    minimal = [
        {"v": 1, "line": {"line_no": 0, "text": ""}, "kind": "assistant"},
        {"v": 1, "line": {"line_no": 1, "text": ""}, "kind": "user"},
        {"v": 1, "line": {"line_no": 2, "text": ""}, "kind": "system"},
        {"v": 1, "line": {"line_no": 3, "text": ""}, "kind": "other"},
        {"v": 1, "line": {"line_no": 4, "text": ""}, "kind": "none"},
    ]
    built = _fold(minimal)
    assert built.unparsed == 0
    assert [s["source"] for s in built.trajectory["steps"]] == ["agent", "system", "system"]


def test_duplicate_ordinals_fold_as_the_builder_maps_duplicate_events() -> None:
    events = [_ev(_user("a"), 0), _ev(_assistant([{"type": "text", "text": "b"}]), 1)]
    doubled = [events[0], events[1], events[1], events[0]]
    _assert_same(doubled, "claude_code")
    lines = fold(_fragments(doubled), session_id="s", agent_name="a").trajectory["extra"]["probe"]
    assert [n for n, _flags in lines["lines"]] == [0, 1, 1, 0]


def test_a_stop_reason_the_renderer_would_not_print_is_dropped() -> None:
    frag = fragment(
        _ev(_assistant([{"type": "text", "text": "x"}], msg={"stop_reason": "max_tokens"}), 0)
    )
    frag["stop"]["reason"] = "end_turn"
    [step] = _fold([frag]).trajectory["steps"]
    assert "stop_reason" not in step["extra"]


_JUNK = [
    None,
    0,
    -1,
    2**62,
    1.5,
    True,
    False,
    "",
    "y" * 10_000,
    [],
    [1, "a", None],
    {},
    {"k": {"n": 1}},
    {"type": "text"},
    "user",
    "tool_call",
]


def _mutate(rng: random.Random, value: Any, depth: int = 0) -> Any:
    """`value` with one randomly chosen place replaced, removed or added to."""
    if isinstance(value, dict) and value and depth < 4 and rng.random() < 0.7:
        key = rng.choice(list(value))
        out = dict(value)
        roll = rng.random()
        if roll < 0.15:
            del out[key]
        elif roll < 0.25:
            out[rng.choice(["content", "arguments", "zz", "seq", "type"])] = rng.choice(_JUNK)
        else:
            out[key] = _mutate(rng, value[key], depth + 1)
        return out
    if isinstance(value, list) and value and depth < 4 and rng.random() < 0.7:
        out = list(value)
        i = rng.randrange(len(out))
        out[i] = _mutate(rng, out[i], depth + 1)
        if rng.random() < 0.1:
            out.append(rng.choice(_JUNK))
        return out
    return rng.choice(_JUNK)


@pytest.mark.parametrize("seed", range(200))
def test_fold_never_raises_on_mutated_fragments(seed: int) -> None:
    rng = random.Random(seed)
    events, _agent = ALL_CASES[NAMES[seed % len(NAMES)]]
    fragments = orjson.loads(orjson.dumps(_fragments(events))) or [{"v": 1}]
    for _ in range(rng.randint(1, 5)):
        i = rng.randrange(len(fragments))
        fragments[i] = _mutate(rng, fragments[i])
    if rng.random() < 0.3:
        fragments.insert(rng.randrange(len(fragments) + 1), rng.choice(_JUNK))
    built = _fold(fragments)
    assert len(built.trajectory["extra"]["probe"]["lines"]) == len(fragments)


# -- the rest of the contract ---------------------------------------------------------


def test_unparsed_events_are_reported_by_fold_not_logged() -> None:
    def broken(self: Any, line: int, ev: Any) -> None:
        raise KeyError("x")

    original = fold_mod._Folder._map
    fold_mod._Folder._map = broken
    try:
        with capture_logs() as logs:
            built = fold([fragment(_ev(_user("hi"), 0))], session_id="s", agent_name="a")
    finally:
        fold_mod._Folder._map = original
    assert built.unparsed == 1 and built.unparsed_events == [(0, "user", "KeyError")]
    assert logs == []


def test_a_long_inference_folds_in_linear_time() -> None:
    events = [
        _ev(_assistant([{"type": "thinking", "thinking": "t" * 1000}]), i) for i in range(20_000)
    ]
    started = time.perf_counter()
    built = build_trajectory(events, session_id="s", agent_name="a", builder=Builder.FOLD)
    assert time.perf_counter() - started < 5
    [step] = built.trajectory["steps"]
    assert len(step["reasoning_content"]) == 20_000 * 1000 + 19_999 * 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fold", Builder.FOLD),
        (" FOLD ", Builder.FOLD),
        ("reference", Builder.REFERENCE),
        (Builder.FOLD, Builder.FOLD),
        ("folds", Builder.REFERENCE),
        ("", Builder.REFERENCE),
        (None, Builder.REFERENCE),
    ],
)
def test_the_builder_a_setting_names(value: Any, expected: Builder) -> None:
    assert builder_for(value) is expected


def test_the_default_builder_is_the_reference() -> None:
    from engine.shared.config import Settings

    assert builder_for(Settings().session_atif_builder) is Builder.REFERENCE


@pytest.mark.parametrize("builder", ["reference", "fold"])
def test_build_and_render_runs_the_builder_it_is_given(
    monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    called: list[str] = []
    real_fold, real_reference = fold_mod.fold, build_reference.build_trajectory
    import engine.ingest.atif.build as build_mod

    monkeypatch.setattr(
        build_mod, "fold", lambda *a, **k: called.append("fold") or real_fold(*a, **k)
    )
    monkeypatch.setattr(
        build_reference,
        "build_trajectory",
        lambda *a, **k: called.append("reference") or real_reference(*a, **k),
    )
    built, lines, error = build_and_render(INGEST_EVENTS, "s", "claude_code", True, builder)
    assert called == [builder] and error is None
    assert lines == lines_from_events(INGEST_EVENTS)
    assert _data(built) == _data(
        build_reference.build_trajectory(INGEST_EVENTS, session_id="s", agent_name="claude_code")
    )

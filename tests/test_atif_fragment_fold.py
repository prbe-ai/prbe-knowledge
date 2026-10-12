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
import hashlib
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
    FRAGMENT_VERSIONS,
    IMAGE_MEDIA_TYPES,
    Kind,
    PieceType,
    fragment,
    fragment_line,
)
from engine.ingest.atif.lines import lines_from_trajectory
from engine.ingest.atif.models import Trajectory
from engine.ingest.atif.store import strip_provenance
from engine.ingest.atif.uploaded import as_version_1, fragment_lines, fragments_like, image_parts
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
    "events stored without an ordinal": [
        _ev(_user("one"), 3),
        _ev(_user("unnumbered"), None),
        {"raw": {"type": "system", "content": "no line_no key"}},
        _ev(_user("unnumbered again"), None),
    ],
    "timestamps that do and do not parse": [
        _ev(_user("a", timestamp="2026-10-01T00:00:00Z"), 0),
        _ev(_assistant([{"type": "text", "text": "b"}], timestamp="yesterday"), 1),
        _ev(_assistant([{"type": "text", "text": "c"}], timestamp="2026-10-01T00:00:01+00:00"), 2),
        _ev({"type": "system", "timestamp": 5}, 3),
        # Parses on 3.11+ only: a fragment carries it as written, fold decides.
        _ev({"type": "system", "timestamp": "2026-10-01T00:00:01.12Z"}, 4),
        _ev({"type": "system", "timestamp": "20261001T000001"}, 5),
        _ev({"type": "system", "timestamp": ""}, 6),
    ],
    "nothing readable at all": [_ev("x", 0), {"line_no": 1}],
    "empty": [],
}


#: Not an equality case: the frozen builder does not cap counts, so a sum past
#: 64 bits makes a document no JSON writer accepts. fold caps; no real count
#: comes near it.
HUGE_USAGE = [
    _ev(
        _assistant(
            [{"type": "text", "text": "a"}],
            inference_id="m",
            usage={
                "input_tokens": 2**63 - 1,
                "cache_read_input_tokens": 2**63 - 1,
                "output_tokens": 2**70,
            },
        ),
        0,
    ),
    _ev(
        _assistant(
            [{"type": "text", "text": "b"}],
            inference_id="n",
            usage={"input_tokens": 2**63 - 1, "output_tokens": 2**63 - 1},
        ),
        1,
    ),
]


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


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _pasted(seed: str, mime: str = "image/png", **extra: Any) -> dict[str, Any]:
    """A pasted image as a tap that uploaded it sends it (probe-events/1
    media_block with `sha256`)."""
    return {"type": "image", "mimeType": mime, "bytes": 1234, "sha256": _sha(seed), **extra}


#: Sessions whose researcher pasted images the tap uploaded. NOT equality cases:
#: version 2 carries the images, which the frozen builder cannot.
IMAGE_CASES: dict[str, list[dict[str, Any]]] = {
    "text then two images then text": [
        _ev(
            _user(
                [
                    {"type": "text", "text": "what is wrong here?"},
                    _pasted("a"),
                    _pasted("b", "image/jpeg"),
                    {"type": "text", "text": "and here"},
                ],
                timestamp="2026-10-12T00:00:00Z",
            ),
            0,
        ),
        _ev(_assistant([{"type": "text", "text": "the axis"}], inference_id="m1"), 1),
    ],
    "an image alone is a prompt": [
        _ev(_user([_pasted("c", "image/webp")]), 0),
        _ev(_assistant([{"type": "text", "text": "a cat"}], inference_id="m1"), 1),
    ],
    "an image beside a result and other dropped blocks": [
        _ev(_assistant([{"type": "tool_use", "id": "t1", "name": "Bash"}], inference_id="m1"), 0),
        _ev(
            _user(
                [
                    {"type": "tool_result", "tool_use_id": "t1", "is_error": True},
                    {"type": "document"},
                    {"type": "text", "text": "see"},
                    _pasted("d", "image/gif"),
                    {"type": "image", "mimeType": "image/png", "bytes": 9},
                ]
            ),
            1,
        ),
    ],
    "only some images carry a usable digest": [
        _ev(
            _user(
                [
                    {"type": "text", "text": "three"},
                    _pasted("e"),
                    {"type": "image", "mimeType": "image/png", "bytes": 5, "sha256": "ab" * 31},
                    {"type": "image", "mimeType": "image/png", "sha256": _sha("f").upper()},
                    {"type": "image", "mimeType": "image/svg+xml", "sha256": _sha("g")},
                    {"type": "image", "sha256": _sha("h")},
                ]
            ),
            0,
        ),
    ],
    "the same image twice": [
        _ev(_user([{"type": "text", "text": "again"}, _pasted("i"), _pasted("i")]), 0),
    ],
}


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

    every = [*ALL_CASES.values(), *((events, "claude_code") for events in IMAGE_CASES.values())]
    emitted = set()
    for events, _agent in every:
        emitted |= _keys(_fragments(events))
    doc = fragment_mod.__doc__ or ""
    undocumented = sorted(k for k in emitted if k not in doc)
    assert not undocumented, f"document these in fragment.py's schema: {undocumented}"
    # Keys the cases must exercise, not merely allow (the harness goldens end
    # turns with `end_turn` and `tool_use`, which no Line prints).
    assert {"stop", "stop_reason", "extras", "lineage", "media_type", "sha256"} <= emitted
    assert {p.value for p in PieceType} <= {
        p["type"] for events, _a in every for f in _fragments(events) for p in f.get("parts") or []
    }


VENDORED = (ROOT / "engine/ingest/atif/fragment.py", ROOT / "engine/shared/transcript_render.py")

#: FRAGMENT_VERSION -> the sha256 of each vendored file that makes its fragments.
VENDORED_PINS: dict[int, dict[str, str]] = {
    1: {
        "engine/ingest/atif/fragment.py": (
            "83cbfbe7341db227592c5a82c37e8a47201890def95e1f852b0478534fff89cb"
        ),
        "engine/shared/transcript_render.py": (
            "c0bb4168f7aa86f89a9c6360d1f120b588458d94ad5e0b84cd9c077942008c9e"
        ),
    },
    # Version 2 (pasted images). The same file still builds version 1, byte
    # for byte: test_version_1_fragments_are_what_they_were.
    2: {
        "engine/ingest/atif/fragment.py": (
            "b0d5ac850d7c84ff1028d42ee95a04041c80ce1f73b6ea580ec8b00a2516a809"
        ),
        "engine/shared/transcript_render.py": (
            "c0bb4168f7aa86f89a9c6360d1f120b588458d94ad5e0b84cd9c077942008c9e"
        ),
    },
}


def test_the_vendored_files_are_pinned_for_their_fragment_version() -> None:
    """These two files are vendored into the research-os tap
    (agent/src/probe/tap_core/): the tap runs them to build the fragments it
    uploads (protocol 3), so changing either one changes what clients send.
    Either bump FRAGMENT_VERSION (and teach fold the new version), or
    consciously update the pin here AND re-vendor the files in research-os.
    A wording change in the renderer moves every Line a client sends."""
    assert FRAGMENT_VERSION in VENDORED_PINS, (
        f"FRAGMENT_VERSION {FRAGMENT_VERSION} has no pin: add one, and teach fold the version"
    )
    actual = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in VENDORED
    }
    assert actual == VENDORED_PINS[FRAGMENT_VERSION], (
        "a vendored file changed: bump FRAGMENT_VERSION, or update this pin AND re-vendor "
        "into research-os agent/src/probe/tap_core/"
    )


#: Standard library names newer than Python 3.10, the tap's floor, by module.
_NEWER_THAN_3_10 = {
    "enum": {
        "StrEnum",
        "ReprEnum",
        "EnumCheck",
        "FlagBoundary",
        "verify",
        "member",
        "nonmember",
        "global_enum",
        "show_flag_values",
        "EnumType",
        "property",
    },
    "typing": {
        "Self",
        "LiteralString",
        "Never",
        "assert_never",
        "assert_type",
        "reveal_type",
        "Required",
        "NotRequired",
        "TypeVarTuple",
        "Unpack",
        "dataclass_transform",
        "override",
        "TypeAliasType",
        "get_overloads",
        "clear_overloads",
        "ReadOnly",
        "TypeIs",
        "NoDefault",
        "get_protocol_members",
        "is_protocol",
    },
    "datetime": {"UTC"},
    "itertools": {"batched"},
    "contextlib": {"chdir"},
    "operator": {"call"},
    "hashlib": {"file_digest"},
    "asyncio": {"TaskGroup", "timeout", "timeout_at", "Timeout", "Runner", "Barrier"},
    "warnings": {"deprecated"},
    "copy": {"replace"},
    "re": {"NOFLAG", "PatternError"},
    "math": {"cbrt", "exp2", "sumprod", "fma"},
}
_MODULES_NEWER_THAN_3_10 = {"tomllib", "wsgiref.types"}
_BUILTINS_NEWER_THAN_3_10 = {"ExceptionGroup", "BaseExceptionGroup", "PythonFinalizationError"}
#: Calls whose result changed after 3.10: a fragment built on the tap would differ
#: from one built in the engine. `fromisoformat` accepts far more from 3.11 on.
_VERSION_DEPENDENT_CALLS = {"fromisoformat"}


def _not_3_10(source: str) -> list[str]:
    """What in `source` a Python 3.10 tap cannot run, or would run differently."""
    try:
        tree = ast.parse(source, feature_version=(3, 10))
    except SyntaxError as exc:
        return [f"syntax: {exc.msg} (line {exc.lineno})"]
    found: list[str] = []
    modules: dict[str, str] = {}  # local name -> stdlib module
    for node in ast.walk(tree):
        if isinstance(node, ast.TryStar):  # belt and braces: feature_version is best-effort
            found.append("except*")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _MODULES_NEWER_THAN_3_10:
                    found.append(f"import {alias.name}")
                modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in _MODULES_NEWER_THAN_3_10:
                found.append(f"from {node.module}")
            for alias in node.names:
                if alias.name in _NEWER_THAN_3_10.get(node.module, ()):
                    found.append(f"{node.module}.{alias.name}")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.attr in _NEWER_THAN_3_10.get(modules.get(node.value.id, ""), ())
        ):
            found.append(f"{modules[node.value.id]}.{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in _VERSION_DEPENDENT_CALLS:
            found.append(f".{node.attr} (version-dependent)")
        elif isinstance(node, ast.Name) and node.id in _BUILTINS_NEWER_THAN_3_10:
            found.append(node.id)
    return found


@pytest.mark.parametrize("path", VENDORED, ids=lambda p: p.name)
def test_what_the_tap_vendors_runs_the_same_on_python_3_10(path: Path) -> None:
    """fragment.py goes into the tap (Python >= 3.10) with transcript_render.py:
    standard library and each other only, nothing newer than 3.10, and nothing
    whose result moved between versions."""
    source = path.read_text()
    assert _not_3_10(source) == []
    foreign = set()
    for node in ast.walk(ast.parse(source)):
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
            if isinstance(node, ast.ImportFrom) and node.module
            else []
        )
        foreign |= {
            n
            for n in names
            if n != "engine.shared.transcript_render"
            and n.split(".")[0] not in sys.stdlib_module_names
        }
    assert not foreign, f"{path.name} imports {sorted(foreign)}"


@pytest.mark.parametrize(
    "snippet",
    [
        "from enum import StrEnum",
        "import enum\nclass K(enum.StrEnum): pass",
        "from typing import Self",
        "import typing as t\nx: t.Self",
        "import tomllib",
        "try:\n    pass\nexcept* ValueError:\n    pass",
        "def f[T](x: T) -> T: return x",
        "from datetime import UTC",
        "import datetime\ndatetime.datetime.fromisoformat('2026-10-01')",
        "raise ExceptionGroup('x', [ValueError()])",
    ],
)
def test_the_portability_check_bites(snippet: str) -> None:
    assert _not_3_10(snippet)


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


def test_fold_publishes_the_fragment_versions_it_reads() -> None:
    """The door intersects its own setting with this."""
    assert frozenset({1, 2}) == frozenset(FRAGMENT_VERSIONS) == fold_mod.SUPPORTED_FRAGMENT_VERSIONS
    assert FRAGMENT_VERSION == max(FRAGMENT_VERSIONS) == 2


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


def test_a_repeated_ordinal_is_unreadable_and_the_first_keeps_the_slot() -> None:
    events = [_ev(_user("a"), 0), _ev(_assistant([{"type": "text", "text": "b"}]), 1)]
    fragments = _fragments([events[0], events[1], events[1], events[0]])
    built = _fold(fragments)
    assert [n for n, _flags in built.trajectory["extra"]["probe"]["lines"]] == [0, 1, None, None]
    assert built.unparsed_events == [(1, None, Unreadable.INVALID), (0, None, Unreadable.INVALID)]
    assert [line.text for line in lines_from_trajectory(built.trajectory)] == [
        "USER: a",
        "ASSISTANT: b",
        "",
        "",
    ]
    # Null ordinals (events stored without one) are not ordinals: never repeats.
    unnumbered = _fragments([_ev(_user("x"), None), _ev(_user("y"), None)])
    assert _fold(unnumbered).unparsed == 0


@pytest.mark.parametrize("line_no", [True, False, -1, 2**63, 1.0, "3"])
def test_an_ordinal_that_is_not_a_non_negative_int_is_unreadable(line_no: Any) -> None:
    frag = fragment(_ev(_user("x"), 0))
    frag["line"]["line_no"] = line_no
    built = _fold([frag])
    assert built.unparsed_events == [(None, None, Unreadable.INVALID)]
    assert built.trajectory["extra"]["probe"]["lines"] == [[None, 0]]


def test_token_counts_are_capped_so_the_document_serialises() -> None:
    events = HUGE_USAGE
    frags = _fragments(events)
    assert frags[0]["usage"]["output_tokens"] == 2**63 - 1
    built = _fold(frags)  # orjson.dumps inside
    first, second = built.trajectory["steps"]
    assert first["metrics"]["prompt_tokens"] == 2**63 - 1
    assert second["metrics"]["completion_tokens"] == 2**63 - 1
    totals = built.trajectory["final_metrics"]
    assert totals["total_prompt_tokens"] == totals["total_completion_tokens"] == 2**63 - 1


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ({"stop_reason": "end_turn", "content": [{"type": "text", "text": "x"}]}, "end_turn"),
        (
            {"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "Bash"}]},
            "tool_use",
        ),
        ({"stop_reason": "max_tokens", "content": [{"type": "text", "text": "x"}]}, "max_tokens"),
        ({"stop_reason": "end_turn", "content": "a string reply"}, "end_turn"),
        ({"stop_reason": "stop"}, "stop"),
        ({"stop_reason": "x" * 1000, "content": []}, "x" * 256),
        ({"stop_reason": "", "content": []}, None),
        ({"stop_reason": None, "content": []}, None),
        ({"stop_reason": {"why": "x"}, "content": []}, None),
        ({"stop_reason": 5, "content": []}, None),
    ],
)
def test_every_string_stop_reason_is_carried_capped_and_never_rendered(
    msg: dict[str, Any], expected: str | None
) -> None:
    """`stop_reason` is what ended the reply, on any content: a reader (the
    engine's live trajectory) needs Claude Code's `end_turn`, which no Line
    prints. `stop` stays the printed one, the only one fold renders."""
    ev = _ev(_assistant(None, msg=msg), 0)
    frag = fragment(ev)
    assert frag.get("stop_reason") == expected
    assert fragment_line(frag) == line_from_event(ev), "the Line does not move"
    reference = build_reference.build_trajectory([ev], session_id="s", agent_name="a")
    assert _data(fold([frag], session_id="s", agent_name="a")) == _data(reference)


def test_fold_never_reads_stop_reason() -> None:
    """A `stop_reason` the index would print still renders only through `stop`."""
    frag = fragment(
        _ev(_assistant([{"type": "text", "text": "x"}], msg={"stop_reason": "end_turn"}), 0)
    )
    frag["stop_reason"] = "max_tokens"
    [step] = _fold([frag]).trajectory["steps"]
    assert "stop_reason" not in step["extra"]
    assert lines_from_trajectory(_fold([frag]).trajectory)[0].text == "ASSISTANT: x"


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


# -- version 2: pasted images ---------------------------------------------------------

#: sha256 of every version-1 fragment of every case (in NAMES order, each case's
#: name then its fragments, compact JSON, key order kept), taken from main
#: (edd6819) before version 2 existed. Version 1 is what every tap released so
#: far builds: the same file must keep building it byte for byte.
V1_FRAGMENTS_SHA256 = "0d952a9883c318a47f3b2cbe5c0ef7a4b1ead2d666f1efac6ff51522c82c8d23"


def test_version_1_fragments_are_what_they_were() -> None:
    digest = hashlib.sha256()
    count = 0
    for name in NAMES:
        events, _agent = ALL_CASES[name]
        frags = [fragment(e, 1) for e in events if isinstance(e, dict)]
        count += len(frags)
        digest.update(
            json.dumps([name, frags], separators=(",", ":"), ensure_ascii=False).encode()
        )
    assert (len(NAMES), count) == (344, 4813)
    assert digest.hexdigest() == V1_FRAGMENTS_SHA256


def test_version_2_is_version_1_plus_image_parts() -> None:
    """Without its image parts (and `v`), a version-2 fragment IS the version-1
    fragment, key order included: what the frozen builder is compared with."""
    every = [*(events for events, _a in ALL_CASES.values()), *IMAGE_CASES.values()]
    for events in every:
        for ev in events:
            if not isinstance(ev, dict):
                continue
            v1, v2 = fragment(ev, 1), fragment(ev, 2)
            assert json.dumps(as_version_1(v2)) == json.dumps(v1)
            assert fragment_line(v1) == fragment_line(v2), "an image moves no Line"
    for events, _a in ALL_CASES.values():
        for ev in events:
            if isinstance(ev, dict):
                assert json.dumps(fragment(ev, 2)) == json.dumps({**fragment(ev, 1), "v": 2})


@pytest.mark.parametrize("version", [0, 3, "2", True, None])
def test_only_a_version_this_module_builds_is_built(version: Any) -> None:
    with pytest.raises(ValueError):
        fragment(_ev(_user("x"), 0), version)


def test_a_pasted_image_with_its_digest_is_an_image_part() -> None:
    [first, _second] = IMAGE_CASES["text then two images then text"]
    v2 = fragment(first)
    assert v2["v"] == FRAGMENT_VERSION == 2
    assert v2["parts"] == [
        {"type": "text", "seq": 0, "text": "what is wrong here?"},
        {"type": "image", "seq": 1, "media_type": "image/png", "sha256": _sha("a")},
        {"type": "image", "seq": 2, "media_type": "image/jpeg", "sha256": _sha("b")},
        {"type": "text", "seq": 3, "text": "and here"},
    ]
    assert v2["dropped_blocks"] == ["image", "image"], "still named, as version 1 names them"
    v1 = fragment(first, 1)
    assert [p["type"] for p in v1["parts"]] == ["text", "text"]
    assert "1234" not in json.dumps(v2), "never the bytes, nor their size"


def test_an_image_without_a_usable_digest_or_type_stays_a_dropped_block() -> None:
    [only] = IMAGE_CASES["only some images carry a usable digest"]
    frag = fragment(only)
    assert [(p["type"], p.get("sha256")) for p in frag["parts"]] == [
        ("text", None),
        ("image", _sha("e")),
    ]
    assert frag["dropped_blocks"] == ["image"] * 5


def _image_steps(built: Any) -> list[dict[str, Any]]:
    return [
        s for s in built.trajectory["steps"]
        if isinstance(s["message"], list) and any(p["type"] == "image" for p in s["message"])
    ]


def test_fold_carries_an_uploaded_image_as_an_atif_image_part() -> None:
    events = IMAGE_CASES["text then two images then text"]
    built = _fold(_fragments(events))
    [step] = _image_steps(built)
    assert step["source"] == "user" and step["timestamp"] == "2026-10-12T00:00:00Z"
    assert step["message"] == [
        {"type": "text", "text": "what is wrong here?"},
        {"type": "image", "source": {"media_type": "image/png", "path": f"media/{_sha('a')}.png"}},
        {"type": "image", "source": {"media_type": "image/jpeg", "path": f"media/{_sha('b')}.jpg"}},
        {"type": "text", "text": "and here"},
    ]
    assert "dropped_blocks" not in step["extra"], "a carried image is not dropped"
    # The index is untouched: the trajectory renders the events' own Lines.
    assert lines_from_trajectory(built.trajectory) == lines_from_events(events)
    assert fragment_lines(_fragments(events)).lines == lines_from_events(events)
    # What readers get keeps the image (provenance is all that goes).
    stored = strip_provenance(built.trajectory)
    Trajectory.model_validate(stored)
    assert stored["steps"][0]["message"][1]["source"]["path"] == f"media/{_sha('a')}.png"


def test_an_image_alone_is_a_user_step() -> None:
    events = IMAGE_CASES["an image alone is a prompt"]
    built = _fold(_fragments(events))
    first = built.trajectory["steps"][0]
    assert first["source"] == "user"
    assert first["message"] == [
        {"type": "image", "source": {"media_type": "image/webp", "path": f"media/{_sha('c')}.webp"}}
    ]
    # Version 1 makes no step of it, exactly as the frozen builder.
    v1 = _fold([fragment(e, 1) for e in events])
    assert v1.trajectory["steps"][0]["source"] == "agent"


def test_only_a_carried_image_leaves_dropped_blocks() -> None:
    built = _fold(_fragments(IMAGE_CASES["an image beside a result and other dropped blocks"]))
    [step] = _image_steps(built)
    # One image carried, one without a digest still dropped, the document too.
    assert step["extra"]["dropped_blocks"] == ["document", "image"]
    assert [p["type"] for p in step["message"]] == ["text", "image"]
    assert built.trajectory["steps"][0]["observation"]["results"][0]["extra"]["is_error"] is True


@pytest.mark.parametrize("name", sorted(IMAGE_CASES))
def test_fold_of_version_1_of_image_sessions_is_the_reference(name: str) -> None:
    events = IMAGE_CASES[name]
    reference = build_reference.build_trajectory(events, session_id="s", agent_name="claude_code")
    folded = fold(
        [as_version_1(f) for f in _fragments(events)], session_id="s", agent_name="claude_code"
    )
    assert _data(folded) == _data(reference)
    assert _data(fold([fragment(e, 1) for e in events], session_id="s", agent_name="claude_code")) == (
        _data(reference)
    )


def test_the_fold_builder_shows_images_and_the_reference_does_not() -> None:
    events = IMAGE_CASES["text then two images then text"]
    folded = build_trajectory(events, session_id="s", agent_name="claude_code", builder="fold")
    reference = build_trajectory(events, session_id="s", agent_name="claude_code")
    assert len(_image_steps(folded)) == 1 and not _image_steps(reference)


@pytest.mark.parametrize(
    ("change", "readable"),
    [
        ({}, True),
        ({"sha256": "A" * 64}, False),
        ({"sha256": "a" * 63}, False),
        ({"sha256": "../" + "a" * 61}, False),
        ({"sha256": None}, False),
        ({"media_type": "image/svg+xml"}, False),
        ({"media_type": "text/html"}, False),
        ({"media_type": 5}, False),
        ({"v": 1}, False),
    ],
    ids=str,
)
def test_an_image_part_is_read_as_untrusted(change: dict[str, Any], readable: bool) -> None:
    """Its digest and type become a path a reader fetches: only a type ATIF
    names and 64 lowercase hex (nothing that leaves `media/`) are readable,
    and only in version 2."""
    [frag] = _fragments(IMAGE_CASES["an image alone is a prompt"][:1])
    target = copy.deepcopy(frag)
    if "v" in change:
        target["v"] = change["v"]
    else:
        target["parts"][0].update(change)
    built = _fold([target])
    assert (built.unparsed == 0) is readable
    assert bool(_image_steps(built)) is readable


def test_the_paths_fold_writes_are_the_keys_the_engine_stores() -> None:
    from engine.shared.session_suppression import is_own_folder_key, session_folder
    from kb.session_media import media_key

    folder = session_folder("claude_code", "cust", "sid")
    for media_type, ext in IMAGE_MEDIA_TYPES.items():
        key = media_key("claude_code", "cust", "sid", "a" * 64, media_type)
        assert key == f"{folder}{fold_mod.media_path('a' * 64, media_type)}"
        assert key.endswith(f"/media/{'a' * 64}.{ext}")
        assert is_own_folder_key(key, folder), "a session deletion removes it"
    assert not is_own_folder_key(f"{folder}media/{'a' * 64}.svg", folder)
    assert not is_own_folder_key(f"{folder}media/x/{'a' * 64}.png", folder)


def test_server_fragments_are_built_in_the_clients_version() -> None:
    """The shadow and the scripts compare a client's fragments with this
    engine's: a version-1 client (every tap so far) must not read as
    different because this engine's newest version is 2."""
    events = [*IMAGE_CASES["text then two images then text"], *ALL_CASES["ingest:EVENTS"][0]]
    events = [{**e, "line_no": i} for i, e in enumerate(events)]
    for version in FRAGMENT_VERSIONS:
        client = [fragment(e, version) for e in events]
        assert fragments_like(events, client) == client
    mixed = [fragment(e, 1 + i % 2) for i, e in enumerate(events)]
    assert fragments_like(events, mixed) == mixed
    assert image_parts([fragment(e, 2) for e in events]) == 2
    assert image_parts([fragment(e, 1) for e in events]) == 0


@pytest.mark.parametrize("builder", ["fold", "reference"])
def test_the_shadow_compares_image_sessions_with_the_frozen_builder(builder: str) -> None:
    from engine.ingest.atif.compare import shadow

    events = [
        {**e, "line_no": i}
        for i, e in enumerate(
            [*IMAGE_CASES["text then two images then text"], *IMAGE_CASES["an image alone is a prompt"]]
        )
    ]
    # Protocol 2: the pass's own build (fold of version 2) carries the images.
    built = build_trajectory(events, session_id="s", agent_name="claude_code", builder=builder)
    record = shadow(events, None, built, builder, "s", "claude_code")
    assert record["same_trajectory"] is True, record["trajectory_diff"]
    # Protocol 3, a version-2 client: same fragments, and the same trajectory
    # once the images (which only version 2 has) are set aside.
    client = [fragment(e, 2) for e in events]
    own = fold(client, session_id="s", agent_name="claude_code")
    record = shadow(events, client, own, builder, "s", "claude_code")
    assert (record["same_fragments"], record["same_trajectory"]) == (True, True), record
    assert record["images"] == 3
    # A version-1 client of the same session.
    v1 = [fragment(e, 1) for e in events]
    record = shadow(events, v1, fold(v1, session_id="s", agent_name="claude_code"), builder,
                    "s", "claude_code")
    assert (record["same_fragments"], record["same_trajectory"], record["images"]) == (
        True, True, 0
    )
    # A client whose image part differs from what its events say still shows.
    bad = copy.deepcopy(client)
    bad[0]["parts"][1]["sha256"] = _sha("not-it")
    record = shadow(events, bad, None, builder, "s", "claude_code")
    assert record["same_fragments"] is False and record["fragments_differing"] == 1
    assert record["fragments_diff"]["path"] == "parts[1].sha256"

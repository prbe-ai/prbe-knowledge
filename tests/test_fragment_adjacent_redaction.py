"""A credential split across consecutive ATIF fragments (session upload protocol 3).

The vendored scrubber (`_credential_secrets`, generated from research-os
`agent/src/probe/tap_core/secrets.py`) joins consecutive fragments' prose --
`message`, text and thinking `parts` -- the way it joins adjacent events, and
clears the same halves from each fragment's `line.text` by position. Events
read the same prose as the fragments built from them, so a canary batch
carrying both never keeps halves in one that the other redacted. The fragments
here come from the real `fragment()`, so these also hold the scrubber's reading
of a Line (each prose piece once, in order, behind its label) to what
fragment.py and transcript_render build.
"""

import hashlib
import itertools
import json

import pytest

from engine.ingest._credential_secrets import redact_event
from engine.ingest.atif.fragment import fragment

GITHUB = "ghp_" + hashlib.sha256(b"synthetic-fragment-regression").hexdigest()[:36]
INGEST = "ros_ing_" + hashlib.sha256(b"synthetic-fragment-ingest").hexdigest()[:32]
STAMP = "2026-10-06T12:00:00.000Z"


def event(n, raw):
    return {"line_no": n, "raw": raw}


def user(content):
    return {"type": "user", "timestamp": STAMP, "message": {"role": "user", "content": content}}


def assistant(*blocks):
    return {
        "type": "assistant",
        "timestamp": STAMP,
        "message": {"role": "assistant", "content": list(blocks)},
    }


def system(content):
    return {"type": "system", "subtype": "informational", "content": content, "timestamp": STAMP}


def text(value):
    return {"type": "text", "text": value}


def thinking(value):
    return {"type": "thinking", "thinking": value}


def texts_of(fragments):
    out = []
    for piece in fragments:
        out.append(piece["line"]["text"])
        if "message" in piece:
            out.append(piece["message"])
        out.extend(part["text"] for part in piece.get("parts", []) if "text" in part)
    return out


def scrubbed(raws, halves):
    clean, fired = redact_event(
        {"fragments": [fragment(event(n, raw)) for n, raw in enumerate(raws)]}
    )
    assert fired
    for value in texts_of(clean["fragments"]):
        assert all(half not in value for half in halves), value
    assert redact_event(clean) == (clean, [])
    return clean["fragments"]


@pytest.mark.parametrize("key", [GITHUB, INGEST], ids=["github", "ingest"])
def test_split_across_consecutive_user_messages(key):
    first, second = scrubbed(
        [user("here it is " + key[:20]), user(key[20:] + " thanks")], [key[:20], key[20:]]
    )
    assert first["line"]["text"].startswith("USER: here it is <redacted:")
    assert second["message"].endswith("> thanks")


def test_split_across_user_assistant_and_system_lines():
    lines = [
        piece["line"]["text"]
        for piece in scrubbed(
            [user("push with " + GITHUB[:10]), assistant(text(GITHUB[10:25])), system(GITHUB[25:])],
            [GITHUB[:10], GITHUB[10:25], GITHUB[25:]],
        )
    ]
    assert lines == [
        "USER: push with <redacted:github-token>",
        "ASSISTANT: <redacted:github-token>",
        "SYSTEM (informational): <redacted:github-token>",
    ]


def test_a_one_character_piece_never_rewrites_the_label():
    key = "ghp_" + "aB3" * 11 + "aBA"
    _, second = scrubbed([assistant(text("key " + key[:-1])), assistant(text("A"))], [key[:-1]])
    assert second["line"]["text"] == "ASSISTANT: <redacted:github-token>"


def test_a_piece_repeated_inside_a_later_piece_leaves_no_half_of_either():
    other = "ghp_" + hashlib.sha256(b"second synthetic credential").hexdigest()[:36]
    tail = GITHUB[-3:]
    _, middle, _ = scrubbed(
        [
            assistant(text("k " + GITHUB[:-3])),
            assistant(text(tail), text(" see " + tail + " then " + other[:20])),
            assistant(text(other[20:] + " end")),
        ],
        [GITHUB[:-3], other[:20], other[20:]],
    )
    assert middle["line"]["text"] == (
        "ASSISTANT: <redacted:github-token>\nASSISTANT:  see "
        + tail
        + " then <redacted:github-token>"
    )


def test_a_split_inside_one_fragment_and_around_a_tool_line():
    head = GITHUB[:20]
    (only,) = scrubbed(
        [assistant(text("use " + head), text(GITHUB[20:] + " now"))], [head, GITHUB[20:]]
    )
    assert (
        only["line"]["text"]
        == "ASSISTANT: use <redacted:github-token>\nASSISTANT: <redacted:github-token> now"
    )
    # A tool line naming the same text is never taken for the prose segment.
    tool = {"type": "tool_use", "id": "toolu_01", "name": head}
    first, _ = scrubbed([assistant(tool, text(head)), user(GITHUB[20:])], [GITHUB[20:]])
    assert first["line"]["text"] == "TOOL_USE: " + head + "\nASSISTANT: <redacted:github-token>"


@pytest.mark.parametrize("field", ["id", "name", "summary"])
def test_tool_call_metadata_never_joins_the_next_fragment(field):
    scrubbed([assistant(text(GITHUB[:20])), user(GITHUB[20:] + " ok")], [GITHUB[:20], GITHUB[20:]])
    block = {"type": "tool_use", "id": "toolu_01", "name": "Bash", "summary": "ls"} | {
        field: GITHUB[:20]
    }
    payload = {
        "fragments": [
            fragment(event(0, assistant(block))),
            fragment(event(1, user(GITHUB[20:] + " ok"))),
        ]
    }
    assert redact_event(payload) == (payload, [])


# Each harness's sanitizer output (research-os tap_core, 2026-10-06), by slot:
# Claude Code prompts are strings; Codex, pi and Kimi send blocks; Codex and
# Kimi write a reasoning or a streamed part as its own event.
HARNESSES = {
    "claude_code": {
        "user": lambda t: user(t),
        "text": lambda t: assistant(text(t)),
        "think": lambda t: assistant(thinking(t)),
        "system": system,
    },
    "codex": {
        "user": lambda t: user([text(t)]),
        "text": lambda t: assistant(text(t)),
        "think": lambda t: assistant(thinking(t)),
    },
    "pi": {
        "user": lambda t: user([text(t)]),
        "text": lambda t: assistant(text(t)),
        "think": lambda t: assistant(thinking(t)),
    },
    "kimi": {
        "user": lambda t: user([text(t)]),
        "text": lambda t: assistant(text(t)) | {"inference_id": "t1:1"},
        "think": lambda t: assistant(thinking(t)) | {"inference_id": "t1:1"},
    },
}
SHAPES = [
    ("user", "text"),
    ("text", "user"),
    ("text", "text"),
    ("think", "think"),
    ("user", "user"),
    ("text", "think"),
    ("user", "text", "text"),
    ("think", "text", "user"),
    ("user", "system"),
    ("system", "text"),
    ("text", "system", "user"),
]


def splits(key, slots):
    for cuts in itertools.combinations((5, 12, 20, 28, len(key) - 5), slots - 1):
        bounds = (0, *cuts, len(key))
        yield [key[a:b] for a, b in itertools.pairwise(bounds)]


def event_prose(events):
    out = []
    for item in events:
        content = item["raw"].get("message", {}).get("content", item["raw"].get("content"))
        if isinstance(content, str):
            out.append(content)
        for block in content if isinstance(content, list) else ():
            out.extend(block[k] for k in ("text", "thinking") if isinstance(block.get(k), str))
    return out


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_events_redact_every_split_their_fragments_redact(harness):
    slots, cases = HARNESSES[harness], 0
    for shape, key in itertools.product(
        [s for s in SHAPES if set(s) <= set(HARNESSES[harness])], (GITHUB, INGEST)
    ):
        for chunks in splits(key, len(shape)):
            values = list(chunks)
            values[0], values[-1] = "use " + values[0], values[-1] + " now"
            events = [
                event(n, slots[slot](v))
                for n, (slot, v) in enumerate(zip(shape, values, strict=True))
            ]
            # One scrub of a canary batch, as the tap's journal runs it.
            clean, _ = redact_event({"events": events, "fragments": [fragment(e) for e in events]})
            assert not [c for c in chunks if any(c in t for t in texts_of(clean["fragments"]))], (
                shape,
                chunks,
            )
            assert not [c for c in chunks if any(c in t for t in event_prose(clean["events"]))], (
                shape,
                chunks,
            )
            assert redact_event(clean)[0] == clean
            cases += 1
    assert cases == (140 if harness == "claude_code" else 100)


def test_fragment_and_event_read_the_same_text():
    # The positional edit relies on it: each prose piece is rendered verbatim,
    # once, in order, behind a label.
    raw = assistant(
        thinking("plan: a\nb"),
        text("x"),
        {"type": "tool_use", "id": "t", "name": "Bash"},
        text("y"),
    )
    built = fragment(event(0, raw))
    assert (
        built["line"]["text"]
        == "ASSISTANT (thinking): plan: a\nb\nASSISTANT: x\nTOOL_USE: Bash\nASSISTANT: y"
    )
    assert json.dumps([p.get("text") for p in built["parts"]]) == json.dumps(
        ["plan: a\nb", "x", None, "y"]
    )

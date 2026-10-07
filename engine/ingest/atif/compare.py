"""Where two ATIF builds, or two fragment lists, first differ -- without content.

Used by the ingest pass's fragment shadow (`shadow`, below; setting
SESSION_FRAGMENT_SHADOW) and by `scripts/atif_sessions.py replay`. Both report
to places that must never carry transcript text (worker logs, job output), so a
difference is a PATH of schema keys and indexes plus what differs: never a
value, and never a key outside the document's own vocabulary (a harness
extra's name, anything a client chose prints as `*`).

THE SHADOW (contract §4, plan R3/R4). Run on a session's completing pass, off
the event loop, it says whether the fragment path would have built what the
frozen builder (build_reference.py) builds:

  protocol 2   fold(map(fragment, events)) against build_reference(events).
               One of the two is the pass's own build (SESSION_ATIF_BUILDER
               names which); only the other is computed.
  protocol 3   when the batches carried `events` (the canary, R5): the CLIENT's
  + events     fragments against this engine's fragment(events), ordinal by
               ordinal, and fold(client fragments) -- the pass's own build --
               against build_reference(events).

The client's fragments are what the pass serves either way: they are the
client's word, and the events are canary evidence only. A difference is what
the flip waits on (R3), never a reason to serve something else.
"""

from __future__ import annotations

from typing import Any

from engine.ingest.atif import build_reference
from engine.ingest.atif.build import Builder, builder_for
from engine.ingest.atif.fold import BuildResult, fold
from engine.ingest.atif.fragment import fragment
from engine.ingest.atif.uploaded import fragment_ordinal

#: Root `extra` keys of a stored trajectory.json that stamp the WRITE rather
#: than come out of the build (scripts/atif_sessions.py compares without them).
WRITE_STAMPS = ("session_ended", "validation_error")

#: Keys a difference path may name: the document's own vocabulary (ATIF, the
#: render provenance, BuildResult, the fragment schema in fragment.py). Any
#: other key -- a harness extra's name, anything a client chose -- prints as
#: `*`, so a report carries ids, counts and schema names, never transcript
#: content.
PATH_KEYS = frozenset(
    {
        # BuildResult
        "trajectory",
        "unparsed",
        "unparsed_events",
        # ATIF and its render provenance
        "schema_version",
        "session_id",
        "agent",
        "name",
        "version",
        "model_name",
        "steps",
        "step_id",
        "source",
        "message",
        "type",
        "text",
        "timestamp",
        "reasoning_content",
        "tool_calls",
        "tool_call_id",
        "function_name",
        "arguments",
        "observation",
        "results",
        "source_call_id",
        "metrics",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cache_creation_input_tokens",
        "final_metrics",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_steps",
        "extra",
        "probe",
        "format",
        "lines",
        "probe_parts",
        "subtype",
        "event_type",
        "attachment_type",
        "inference_id",
        "continues_inference",
        "origin",
        "stop_reason",
        "dropped_blocks",
        "summary",
        "stats",
        "added_lines",
        "removed_lines",
        "replace_all",
        "is_error",
        "tool_use_id",
        "result_bytes",
        "codex_extras",
        "pi_extras",
        "kimi_extras",
        # the fragment (fragment.py, FRAGMENT_VERSION 1)
        "v",
        "line",
        "line_no",
        "user_turn",
        "compact_boundary",
        "compact_summary",
        "line_error",
        "kind",
        "error",
        "extras",
        "lineage",
        "uuid",
        "parentUuid",
        "logicalParentUuid",
        "compaction",
        "parts",
        "seq",
        "call_id",
        "id_text",
        "model",
        "usage",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "stop",
        "reason",
        "id",
        "agent_version",
        *WRITE_STAMPS,
    }
)


def first_difference(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    """Where two JSON values first differ: a path of schema keys and indexes
    and what differs (`type`, `missing_left` / `missing_right`, `length` with
    both lengths, `value`). Never a value."""
    here = path or "$"
    if type(left) is not type(right):
        return {
            "path": here,
            "kind": "type",
            "left": type(left).__name__,
            "right": type(right).__name__,
        }
    if isinstance(left, dict):
        for key in left:
            name = key if key in PATH_KEYS else "*"
            sub = f"{path}.{name}" if path else name
            if key not in right:
                return {"path": sub, "kind": "missing_right"}
            if (diff := first_difference(left[key], right[key], sub)) is not None:
                return diff
        for key in right:
            if key not in left:
                name = key if key in PATH_KEYS else "*"
                return {"path": f"{path}.{name}" if path else name, "kind": "missing_left"}
        return None
    if isinstance(left, list):
        for i, (a, b) in enumerate(zip(left, right, strict=False)):
            if (diff := first_difference(a, b, f"{path}[{i}]")) is not None:
                return diff
        if len(left) != len(right):
            return {"path": here, "kind": "length", "left": len(left), "right": len(right)}
        return None
    return None if left == right else {"path": here, "kind": "value"}


def result_data(built: BuildResult) -> dict[str, Any]:
    """A build as compared: the document, provenance included, and what it
    could not map."""
    return {
        "trajectory": built.trajectory,
        "unparsed": built.unparsed,
        "unparsed_events": [list(e) for e in built.unparsed_events],
    }


def builds_difference(left: BuildResult, right: BuildResult) -> dict[str, Any] | None:
    """`first_difference` of two builds; the whole-value check first, in C."""
    a, b = result_data(left), result_data(right)
    return None if a == b else first_difference(a, b)


def fragments_difference(client: list[Any], server: list[Any]) -> tuple[dict[str, Any] | None, int]:
    """(the first ordinal whose fragments differ and where, how many differ).

    Paired by ordinal, not by position: one missing fragment must not make
    every later one look different. The path is inside the fragment
    (`line.text`, `parts[0].text`), `kind` as in `first_difference`, or
    `missing_client` / `missing_server` for an ordinal only one side has.
    """

    def by_ordinal(items: list[Any]) -> dict[int | None, Any]:
        out: dict[int | None, Any] = {}
        for item in items:
            out.setdefault(fragment_ordinal(item), item)
        return out

    mine, theirs = by_ordinal(client), by_ordinal(server)
    ordinals = sorted(set(mine) | set(theirs), key=lambda n: (n is None, n or 0))
    first: dict[str, Any] | None = None
    differing = 0
    for ordinal in ordinals:
        if ordinal not in theirs or ordinal not in mine:
            diff: dict[str, Any] | None = {
                "path": "$",
                "kind": "missing_server" if ordinal not in theirs else "missing_client",
            }
        elif mine[ordinal] == theirs[ordinal]:
            continue
        else:
            diff = first_difference(mine[ordinal], theirs[ordinal]) or {
                "path": "$",
                "kind": "value",
            }
        differing += 1
        if first is None:
            first = {"ordinal": ordinal, **diff}
    return first, differing


def shadow(
    events: list[Any],
    client_fragments: list[Any] | None,
    built: BuildResult | None,
    builder: str,
    session_id: str,
    agent_name: str,
) -> dict[str, Any]:
    """The comparison above for one session, as the fields of one log line:
    counts, booleans and content-free paths. Positional, plain data: the
    ingest pass runs it in `cpu_pool`.

    `client_fragments` None: a protocol-2 session (the server makes the
    fragments from `events`). `built`: the pass's own build when it has one --
    of `builder` for protocol 2, of the client's fragments for protocol 3 --
    so only the other side is computed here.
    """
    events = [e for e in events if isinstance(e, dict)]
    server = [fragment(e) for e in events]
    record: dict[str, Any] = {"events": len(events)}
    if client_fragments is None:
        configured = builder_for(builder)
        if built is not None and configured is Builder.FOLD:
            folded = built
        else:
            folded = fold(server, session_id=session_id, agent_name=agent_name)
        if built is not None and configured is Builder.REFERENCE:
            reference = built
        else:
            reference = _reference(events, session_id, agent_name)
        record.update(same_fragments=None, fragments_diff=None, fragments_differing=None)
    else:
        diff, differing = fragments_difference(client_fragments, server)
        if built is not None:
            folded = built
        else:
            folded = fold(client_fragments, session_id=session_id, agent_name=agent_name)
        reference = _reference(events, session_id, agent_name)
        record.update(
            fragments=len(client_fragments),
            same_fragments=diff is None,
            fragments_diff=diff,
            fragments_differing=differing,
        )
    trajectory_diff = builds_difference(folded, reference)
    record.update(
        same_trajectory=trajectory_diff is None,
        trajectory_diff=trajectory_diff,
        steps=len(folded.trajectory.get("steps") or []),
        reference_steps=len(reference.trajectory.get("steps") or []),
        unparsed=folded.unparsed,
        reference_unparsed=reference.unparsed,
    )
    return record


def _reference(events: list[dict[str, Any]], session_id: str, agent_name: str) -> BuildResult:
    built = build_reference.build_trajectory(events, session_id=session_id, agent_name=agent_name)
    return BuildResult(built.trajectory, built.unparsed, list(built.unparsed_events))

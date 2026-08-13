"""Unit tests for WikiAgentRuntime tool handlers + state safety.

Tests the dispatch path with stubbed persistence helpers so we exercise
state transitions / snapshot rollback / tool result shapes without a
live DB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from engine.shared.exceptions import ToolValidationError
from kb.synthesis import wiki_agent as wa_module
from kb.synthesis.wiki_agent import WikiAgentRuntime


@pytest.fixture
def runtime(monkeypatch) -> WikiAgentRuntime:
    """Build a runtime with stubbed persistence helpers.

    Avoids any DB / Embedder / R2 contact; the runtime operates purely
    on in-memory state for unit testing.
    """
    # Patch fetch_existing_page to return None by default (page not
    # found). Per-test overrides as needed via the stub below.
    page_db: dict[tuple[str, str], dict[str, Any]] = {}
    event_bodies: dict[int, tuple[str, dict[str, Any]]] = {}
    manifest_state: dict[str, Any] = {"events": [], "remaining": 0}

    async def stub_fetch_existing_page(customer_id, wiki_type, slug):
        return page_db.get((wiki_type, slug))

    async def stub_get_event_body(customer_id, queue_id):
        return event_bodies.get(queue_id)

    async def stub_fetch_triaged_manifest(customer_id, *, excluded_queue_ids, count):
        events = [
            ev
            for ev in manifest_state["events"]
            if ev["queue_id"] not in (excluded_queue_ids or [])
        ][:count]
        return events, manifest_state.get("remaining", 0)

    async def stub_fetch_wiki_index(customer_id):
        return list(page_db.values())  # close-enough stub

    monkeypatch.setattr(
        "kb.synthesis.persistence.fetch_existing_page",
        stub_fetch_existing_page,
    )
    monkeypatch.setattr(
        "kb.synthesis.persistence.get_event_body_for_agent",
        stub_get_event_body,
    )
    monkeypatch.setattr(
        "kb.synthesis.persistence.fetch_triaged_manifest",
        stub_fetch_triaged_manifest,
    )
    monkeypatch.setattr(
        "kb.synthesis.persistence.fetch_wiki_index",
        stub_fetch_wiki_index,
    )

    # Build the runtime without importing the Normalizer / store / etc.
    rt = WikiAgentRuntime.__new__(WikiAgentRuntime)
    rt.customer_id = "cust"
    rt.agent_run_id = "run-1"
    rt._run_id = 42
    rt._run_kind = "wake"
    rt._normalizer = None  # tests don't call commit
    rt._store = None
    rt._ctx = None
    rt._pending_updates = {}
    rt._pending_creates = {}
    rt._applied_queue_ids = set()
    rt._skipped_queue_ids = set()
    rt.is_done = False
    rt._wiki_index_cache = None

    # Expose stub stores so tests can poke them.
    rt._test_page_db = page_db  # type: ignore[attr-defined]
    rt._test_event_bodies = event_bodies  # type: ignore[attr-defined]
    rt._test_manifest_state = manifest_state  # type: ignore[attr-defined]
    return rt


# ---------------------------------------------------------------------------
# read_page
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_read_page_not_found_returns_typed_error(
    runtime: WikiAgentRuntime,
) -> None:
    out = await runtime.dispatch_tool("read_page", {"wiki_type": "decision", "slug": "x"})
    assert out["error"] == "page_not_found"


# ---------------------------------------------------------------------------
# get_event_body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_get_event_body_page_2_returns_next_chunk(
    runtime: WikiAgentRuntime,
) -> None:
    big_body = "a" * 6000 + "b" * 6000
    runtime._test_event_bodies[42] = (
        big_body,
        {  # type: ignore[attr-defined]
            "doc_id": "d:1",
            "version": 1,
            "title": "t",
            "source_system": "github",
            "source_ts": datetime(2026, 5, 4, tzinfo=UTC),
        },
    )
    out = await runtime.dispatch_tool("get_event_body", {"queue_id": 42, "page": 2})
    assert out["page"] == 2
    assert out["total_pages"] == 2
    assert out["body"].startswith("b")


@pytest.mark.asyncio
async def test_tool_get_event_body_out_of_range_page_errors(
    runtime: WikiAgentRuntime,
) -> None:
    runtime._test_event_bodies[5] = (
        "hello",
        {  # type: ignore[attr-defined]
            "doc_id": "d:1",
            "version": 1,
            "title": "t",
            "source_system": "github",
            "source_ts": datetime(2026, 5, 4, tzinfo=UTC),
        },
    )
    out = await runtime.dispatch_tool("get_event_body", {"queue_id": 5, "page": 9})
    assert out["error"] == "page_out_of_range"


def _seed(runtime: WikiAgentRuntime, wiki_type: str, slug: str, body: str = "SEED") -> None:
    """Put a page on disk for the agent to edit.

    `update_page` sends anchored EDITS now, so it needs something to anchor on,
    and it refuses a slug that does not exist rather than staging an update the
    commit would silently drop.
    """
    runtime._test_page_db[(wiki_type, slug)] = {  # type: ignore[attr-defined]
        "doc_id": f"wiki:{wiki_type}:{slug}",
        "body": body,
        "title": slug,
        "version": 1,
        "doc_class": "compiled_wiki",
    }


def _edit(text: str, find: str = "SEED") -> list[dict[str, str]]:
    """One replace edit, the shape almost every test needs."""
    return [{"op": "replace", "find": find, "text": text}]


# ---------------------------------------------------------------------------
# update_page (staging + last-write-wins + applied_queue_ids merge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_update_page_staged(runtime: WikiAgentRuntime) -> None:
    _seed(runtime, "decision", "x")
    out = await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            "edits": _edit("body v1"),
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [1, 2],
        },
    )
    assert out["status"] == "staged"
    assert out["pages_pending"] == 1
    assert ("decision", "x") in runtime._pending_updates


@pytest.mark.asyncio
async def test_tool_update_page_re_staged_edits_on_top_of_the_staged_body(
    runtime: WikiAgentRuntime,
) -> None:
    """A second update edits what the FIRST one staged, not the live row.

    This is the property that stopped a re-stage discarding earlier work. With
    a whole-body tool the second call simply overwrote the first, so the model
    could lose its own edit and not know; anchored on the staged text, the two
    compose.
    """
    _seed(runtime, "decision", "x")
    await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            "edits": _edit("body v1"),
            "summary": "s1",
            "commit_message": "m1",
            "applied_queue_ids": [1],
        },
    )
    await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            # Anchored on the text the first edit wrote -- proof the base is
            # the staged body and not the row on disk, which still says SEED.
            "edits": _edit("body v2", find="body v1"),
            "summary": "s2",
            "commit_message": "m2",
            "applied_queue_ids": [2],
        },
    )
    staged = runtime._pending_updates[("decision", "x")]
    assert staged.body_markdown == "body v2"
    assert staged.summary == "s2"
    # applied_queue_ids unioned (1 + 2)


@pytest.mark.asyncio
async def test_tool_update_page_applied_queue_ids_accumulate_across_stages(
    runtime: WikiAgentRuntime,
) -> None:
    _seed(runtime, "runbook", "f")
    for ids in [[1], [2, 3], [4]]:
        await runtime.dispatch_tool(
            "update_page",
            {
                "wiki_type": "runbook",
                "slug": "f",
                # `append_after` on an anchor that survives its own edit, so
                # all three stages apply -- a `replace` would consume the
                # anchor and the second call would have nothing to match.
                "edits": [{"op": "append_after", "find": "SEED", "text": " b"}],
                "summary": "s",
                "commit_message": "m",
                "applied_queue_ids": ids,
            },
        )
    staged = runtime._pending_updates[("runbook", "f")]
    assert staged.applied_queue_ids == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# create_page
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_create_page_staged(runtime: WikiAgentRuntime) -> None:
    out = await runtime.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "new",
            "title": "T",
            "body_markdown": "b",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [9],
        },
    )
    assert out["status"] == "staged"
    assert ("decision", "new") in runtime._pending_creates


@pytest.mark.asyncio
async def test_tool_create_page_slug_exists_on_disk_errors(
    runtime: WikiAgentRuntime,
) -> None:
    runtime._test_page_db[("decision", "exists")] = {  # type: ignore[attr-defined]
        "title": "T",
        "doc_class": "compiled_wiki",
        "body": "b",
        "frontmatter": {},
        "summary": "s",
    }
    out = await runtime.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "exists",
            "title": "X",
            "body_markdown": "b",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [],
        },
    )
    assert out["error"] == "slug_exists"
    assert ("decision", "exists") not in runtime._pending_creates


# ---------------------------------------------------------------------------
# skip_events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_skip_events_marks_correctly(
    runtime: WikiAgentRuntime,
) -> None:
    out = await runtime.dispatch_tool("skip_events", {"queue_ids": [10, 11, 12], "reason": "noise"})
    assert out["status"] == "marked"
    assert out["skipped_count"] == 3
    assert runtime._skipped_queue_ids == {10, 11, 12}


@pytest.mark.asyncio
async def test_tool_skip_events_skip_wins_over_apply(
    runtime: WikiAgentRuntime,
) -> None:
    # First apply event 7 to a staged page.
    _seed(runtime, "decision", "x")
    await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            "edits": _edit("b"),
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [7, 8],
        },
    )
    # Then skip 7 + 9. Skip wins; 7 should leave the staged update's
    # applied_queue_ids and join skipped_queue_ids.
    await runtime.dispatch_tool("skip_events", {"queue_ids": [7, 9], "reason": "second thought"})
    staged = runtime._pending_updates[("decision", "x")]
    assert 7 not in staged.applied_queue_ids
    assert 8 in staged.applied_queue_ids
    assert {7, 9}.issubset(runtime._skipped_queue_ids)


# ---------------------------------------------------------------------------
# done()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_done_invokes_commit(
    runtime: WikiAgentRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch commit() to a stub; assert done() calls it + flips is_done."""
    called = {"n": 0}

    async def fake_commit(self) -> None:
        called["n"] += 1

    monkeypatch.setattr(WikiAgentRuntime, "commit", fake_commit)
    out = await runtime.dispatch_tool("done", {})
    assert called["n"] == 1
    assert runtime.is_done is True
    assert out["committed"] is True


# ---------------------------------------------------------------------------
# discard()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_drops_pending_updates(
    runtime: WikiAgentRuntime,
) -> None:
    _seed(runtime, "decision", "x")
    await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            "edits": _edit("b"),
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [1],
        },
    )
    assert runtime.pending_update_count == 1
    await runtime.discard()
    assert runtime.pending_update_count == 0


# ---------------------------------------------------------------------------
# Snapshot-rollback on tool exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_then_mutate_rolls_back_on_tool_error(
    runtime: WikiAgentRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force _tool_update_page to raise mid-mutation; assert state is
    restored to pre-call snapshot."""
    # First successful update so we have something to snapshot.
    _seed(runtime, "decision", "ok")
    await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "ok",
            "edits": _edit("b"),
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [1],
        },
    )
    pre = runtime.state_snapshot_for_summary()

    # Now patch one of the staging methods to raise mid-call.
    async def bad_update(self, args):
        # Mutate first, then raise — mimics a partial mutation.
        self._pending_updates[("decision", "broken")] = "PARTIAL"  # type: ignore[assignment]
        raise RuntimeError("boom")

    monkeypatch.setattr(WikiAgentRuntime, "_tool_update_page", bad_update)
    with pytest.raises(RuntimeError):
        _seed(runtime, "decision", "broken")
        await runtime.dispatch_tool(
            "update_page",
            {
                "wiki_type": "decision",
                "slug": "broken",
                "edits": _edit("b"),
                "summary": "s",
                "commit_message": "m",
                "applied_queue_ids": [],
            },
        )
    post = runtime.state_snapshot_for_summary()
    # Mutation rolled back: pre and post identical.
    assert pre == post
    assert ("decision", "broken") not in runtime._pending_updates


# ---------------------------------------------------------------------------
# Token estimate (consistent with Gemini count expectations)
# ---------------------------------------------------------------------------


def test_estimate_tokens_consistent_with_gemini_count() -> None:
    """The harness's _estimate_tokens scales with ~1/3.5 chars per token.

    Pure unit assertion against the harness, not the runtime — verify a
    known string maps to the expected ballpark.
    """
    from kb.synthesis.agent_harness import AgentLoop

    # Skip dataclass init: build the loop manually so we don't need a
    # full LLM stub here.
    loop = AgentLoop.__new__(AgentLoop)
    loop._conversation = [{"role": "user", "parts": [{"text": "x" * 350}]}]
    est = loop._estimate_tokens()
    # 350 chars at 1/3.5 = 100 tokens.
    assert 90 <= est <= 110


# ---------------------------------------------------------------------------
# Validator: unknown tool name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises_tool_validation_error(
    runtime: WikiAgentRuntime,
) -> None:
    with pytest.raises(ToolValidationError):
        await runtime.dispatch_tool("not_a_tool", {})


# ---------------------------------------------------------------------------
# Sanity: helper module export + the batch size constant
# ---------------------------------------------------------------------------


def test_default_batch_size_export() -> None:
    """Public re-export so tests / tools can read the same constant."""
    assert wa_module.default_batch_size() > 0


# ---------------------------------------------------------------------------
# create-then-edit within one drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_page_can_edit_a_page_created_in_the_same_drain(
    runtime: WikiAgentRuntime,
) -> None:
    """`update_page` consulted `_pending_updates`, then fell through to the DB.
    It never looked at `_pending_creates`.

    So an agent that created a page and then edited it in the same drain was
    told "no wiki page decision/fresh; use create_page for a new one" -- about a
    page it had just created. Its only ways out were to re-create the page,
    losing the edit, or to give up on it. The staged set has to be one view.
    """
    await runtime.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "fresh",
            "title": "T",
            "body_markdown": "alpha beta",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [1],
        },
    )

    out = await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "fresh",
            "edits": [{"op": "replace", "find": "beta", "text": "gamma"}],
            "summary": "s2",
            "commit_message": "m2",
            "applied_queue_ids": [2],
        },
    )

    assert out["status"] == "staged"
    staged = runtime._pending_creates[("decision", "fresh")]
    assert staged.body_markdown == "alpha gamma"
    # STILL a create. It has never been published, so publishing it as an update
    # would target a row that does not exist -- and staging it in both maps is
    # what the preflight refuses.
    assert ("decision", "fresh") not in runtime._pending_updates
    assert staged.applied_queue_ids == [1, 2]


@pytest.mark.asyncio
async def test_a_page_created_then_edited_stays_a_single_staged_page(
    runtime: WikiAgentRuntime,
) -> None:
    """The preflight's own invariant, checked end to end: the fold-back above
    must leave exactly one staged entry for the page, or the batch it produces
    would be refused as a duplicate."""
    from kb.synthesis.staged_graph import validate_batch

    await runtime.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "fresh",
            "title": "T",
            "body_markdown": "alpha",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [],
        },
    )
    await runtime.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "fresh",
            "edits": [{"op": "replace", "find": "alpha", "text": "omega"}],
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [],
        },
    )

    staged = runtime._staged_pages()
    assert len(staged) == 1
    assert validate_batch(staged) == []

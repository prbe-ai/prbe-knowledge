"""WikiAgentRuntime — the in-process state + tool dispatcher for the agent loop.

The harness (`agent_harness.AgentLoop`) drives the turn-by-turn LLM
interaction; this module owns:

  - mutable agent state (pending_updates, pending_creates, applied_queue_ids,
    skipped_queue_ids)
  - 8 tool handlers (next_events, list_wiki_pages, read_page,
    get_event_body, update_page, create_page, skip_events, done)
  - snapshot-then-mutate inside dispatch_tool: a tool exception rolls
    back any in-flight state mutations
  - commit() — one atomic txn that calls Normalizer._persist for each
    staged update + create, marks queue rows done / synthesis_skipped,
    regenerates the wiki index
  - discard() — drop pending_updates/creates, mark all 'synthesizing'
    rows DLQ on agent halt

Tool result shapes match the spec under "Tool palette" — every tool
returns a dict the harness wraps as a function_response part.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import asyncpg

from engine.ingest.handlers.base import ConnectorContext, make_default_context
from engine.ingest.normalizer import Normalizer, fetch_live_body_from_chunks
from engine.shared.constants import (
    WIKI_AGENT_BATCH_SIZE,
    WIKI_AGENT_MODEL,
    WIKI_DOC_TYPE_PREFIX,
    WIKI_INDEX_DOC_TYPE,
    CompileTrigger,
    DocClass,
    SourceSystem,
)
from engine.shared.db import with_tenant
from engine.shared.embeddings import GeminiEmbedder, get_embedder_v2
from engine.shared.exceptions import AgentHaltError, ToolValidationError
from engine.shared.locks import advisory_lock_key
from engine.shared.logging import get_logger
from engine.shared.models import NormalizationResult, WebhookEvent
from engine.shared.storage import ObjectStore, get_store
from kb.handlers.wiki import (
    INDEX_SLUG,
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from kb.synthesis import index_renderer, persistence, staged_graph
from kb.synthesis.agent_tools import (
    TOOL_VALIDATORS,
    CreatePageArgs,
    DoneArgs,
    GetEventBodyArgs,
    ListWikiPagesArgs,
    NextEventsArgs,
    ReadPageArgs,
    SkipEventsArgs,
    UpdatePageArgs,
)
from kb.synthesis.directed_phrases import persist_directed_vectors
from kb.synthesis.index_renderer import _INDEX_SYSTEM_PROMPT
from kb.synthesis.page_edits import EditError, PageEdit, apply_edits
from kb.synthesis.wiki_links import extract_links, persist_links_for_page

log = get_logger(__name__)


# 6KB pages for get_event_body. Per the plan: pages are 6000 chars.
_EVENT_BODY_PAGE_SIZE = 6000


@dataclass(slots=True)
class _StagedUpdate:
    wiki_type: str
    slug: str
    body_markdown: str
    summary: str
    commit_message: str
    applied_queue_ids: list[int] = field(default_factory=list)
    #: UTF-8 size of the body this edit replaced. Threaded to the preflight so
    #: it grandfathers a shrinking edit on the same rule the tool used -- a tool
    #: that accepts what preflight refuses halts the drain after the
    #: conversation has ended, where nothing can act on it.
    prev_size_bytes: int | None = None


@dataclass(slots=True)
class _StagedCreate:
    wiki_type: str
    slug: str
    title: str
    body_markdown: str
    summary: str
    frontmatter: dict[str, Any]
    commit_message: str
    applied_queue_ids: list[int] = field(default_factory=list)
    #: True once `update_page` has folded edits into `body_markdown`. A second
    #: `create_page` for the same slug would overwrite the body wholesale while
    #: still unioning the queue ids, so the folded events would be marked done
    #: with their content never landing. Reachable after a compaction, where
    #: `state_snapshot_for_summary` shows the model only the slug and the ids --
    #: not that a body it no longer has in context was edited.
    has_folded_edits: bool = False


class WikiAgentRuntime:
    """Per-drain runtime: agent state + tool dispatch + commit.

    One instance per customer per drain. The harness owns the agent
    loop; this owns the world the agent acts on.
    """

    def __init__(
        self,
        customer_id: str,
        *,
        agent_run_id: str,
        run_id: int,
        run_kind: str,
        ctx: ConnectorContext | None = None,
        store: ObjectStore | None = None,
        embedder: GeminiEmbedder | None = None,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.customer_id = customer_id
        self.agent_run_id = agent_run_id
        self._run_id = run_id
        self._run_kind = run_kind
        self._ctx = ctx or make_default_context()
        self._store = store or get_store()
        self._normalizer = normalizer or Normalizer(self._ctx, store=self._store, embedder=embedder)

        # Mutable state (snapshot/restore in dispatch_tool).
        self._pending_updates: dict[tuple[str, str], _StagedUpdate] = {}
        self._pending_creates: dict[tuple[str, str], _StagedCreate] = {}
        self._applied_queue_ids: set[int] = set()
        self._skipped_queue_ids: set[int] = set()
        self.is_done: bool = False

        # Cached wiki index (built once at drain start; refreshed on
        # list_wiki_pages call so the agent's local view matches DB).
        self._wiki_index_cache: list[dict[str, Any]] | None = None

    # -----------------------------------------------------------------------
    # Properties used by the harness
    # -----------------------------------------------------------------------

    @property
    def pending_update_count(self) -> int:
        return len(self._pending_updates) + len(self._pending_creates)

    def state_snapshot_for_summary(self) -> dict[str, Any]:
        """The shape the compactor reads for verbatim preservation."""
        return {
            "pending_updates": [
                {
                    "wiki_type": u.wiki_type,
                    "slug": u.slug,
                    "applied_queue_ids": list(u.applied_queue_ids),
                }
                for u in self._pending_updates.values()
            ],
            "pending_creates": [
                {
                    "wiki_type": c.wiki_type,
                    "slug": c.slug,
                    "applied_queue_ids": list(c.applied_queue_ids),
                }
                for c in self._pending_creates.values()
            ],
            "applied_queue_ids": sorted(self._applied_queue_ids),
            "skipped_queue_ids": sorted(self._skipped_queue_ids),
        }

    async def initial_manifest(self, count: int) -> dict[str, Any]:
        return await self._next_events(count)

    async def wiki_index(self) -> list[dict[str, Any]]:
        if self._wiki_index_cache is None:
            self._wiki_index_cache = await persistence.fetch_wiki_index(self.customer_id)
        return list(self._wiki_index_cache)

    # -----------------------------------------------------------------------
    # Dispatch — snapshot-then-mutate
    # -----------------------------------------------------------------------

    async def dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        validator = TOOL_VALIDATORS.get(name)
        if validator is None:
            raise ToolValidationError(f"unknown tool: {name}")
        try:
            validated = validator.model_validate(args)
        except Exception as exc:
            raise ToolValidationError(f"invalid args for {name}: {exc}") from exc

        snapshot = self._snapshot()
        try:
            return await self._dispatch_validated(name, validated)
        except Exception:
            self._restore(snapshot)
            raise

    async def _dispatch_validated(self, name: str, validated: Any) -> dict[str, Any]:
        if name == "next_events":
            return await self._tool_next_events(validated)
        if name == "list_wiki_pages":
            return await self._tool_list_wiki_pages(validated)
        if name == "read_page":
            return await self._tool_read_page(validated)
        if name == "get_event_body":
            return await self._tool_get_event_body(validated)
        if name == "update_page":
            return await self._tool_update_page(validated)
        if name == "create_page":
            return await self._tool_create_page(validated)
        if name == "skip_events":
            return await self._tool_skip_events(validated)
        if name == "done":
            return await self._tool_done(validated)
        raise ToolValidationError(f"unknown tool: {name}")

    # -----------------------------------------------------------------------
    # Tool handlers
    # -----------------------------------------------------------------------

    async def _tool_next_events(self, args: NextEventsArgs) -> dict[str, Any]:
        return await self._next_events(args.count)

    async def _next_events(self, count: int) -> dict[str, Any]:
        excluded = sorted(self._applied_queue_ids | self._skipped_queue_ids)
        events, remaining = await persistence.fetch_triaged_manifest(
            self.customer_id,
            excluded_queue_ids=excluded,
            count=count,
        )
        return {
            "events": events,
            "remaining": remaining,
            "drain_complete": remaining == 0,
        }

    async def _tool_list_wiki_pages(self, args: ListWikiPagesArgs) -> dict[str, Any]:
        # Always re-fetch on explicit call; the agent might be looking
        # for an entry that was added between the cache build and now.
        self._wiki_index_cache = await persistence.fetch_wiki_index(self.customer_id)
        return {"entries": list(self._wiki_index_cache)}

    async def _tool_read_page(self, args: ReadPageArgs) -> dict[str, Any]:
        key = (args.wiki_type, args.slug)
        # If we have a staged version, return that (the agent's view of
        # the page is what _it_ has decided so far).
        if key in self._pending_updates:
            staged = self._pending_updates[key]
            return {
                "title": None,
                "body_markdown": staged.body_markdown,
                "summary": staged.summary,
                "frontmatter": {},
                "last_updated": None,
                "version": None,
                "is_staged": True,
                "stage_kind": "update",
            }
        if key in self._pending_creates:
            staged_c = self._pending_creates[key]
            return {
                "title": staged_c.title,
                "body_markdown": staged_c.body_markdown,
                "summary": staged_c.summary,
                "frontmatter": dict(staged_c.frontmatter),
                "last_updated": None,
                "version": None,
                "is_staged": True,
                "stage_kind": "create",
            }
        existing = await persistence.fetch_existing_page(
            self.customer_id, args.wiki_type, args.slug
        )
        if existing is None:
            return {
                "error": "page_not_found",
                "wiki_type": args.wiki_type,
                "slug": args.slug,
            }
        return {
            "title": existing.get("title"),
            "body_markdown": existing.get("body") or "",
            "summary": existing.get("summary"),
            "frontmatter": existing.get("frontmatter") or {},
            "last_updated": None,
            "version": None,
            "is_staged": False,
            "stage_kind": None,
            # False means a person has frozen this page: read it for context,
            # but an update_page against it will be dropped at persist time.
            "pipeline_updates": existing.get("pipeline_updates", True),
        }

    async def _tool_get_event_body(self, args: GetEventBodyArgs) -> dict[str, Any]:
        loaded = await persistence.get_event_body_for_agent(self.customer_id, args.queue_id)
        if loaded is None:
            return {"error": "event_not_found", "queue_id": args.queue_id}
        body, meta = loaded
        total_pages = max(1, (len(body) + _EVENT_BODY_PAGE_SIZE - 1) // _EVENT_BODY_PAGE_SIZE)
        if args.page > total_pages:
            return {
                "error": "page_out_of_range",
                "queue_id": args.queue_id,
                "page": args.page,
                "total_pages": total_pages,
            }
        start = (args.page - 1) * _EVENT_BODY_PAGE_SIZE
        chunk = body[start : start + _EVENT_BODY_PAGE_SIZE]
        return {
            "queue_id": args.queue_id,
            "body": chunk,
            "page": args.page,
            "total_pages": total_pages,
            "truncated": total_pages > 1,
            "meta": {
                "doc_id": meta["doc_id"],
                "version": meta["version"],
                "title": meta.get("title"),
                "source_system": meta.get("source_system"),
                "source_ts": meta["source_ts"].isoformat()
                if isinstance(meta.get("source_ts"), datetime)
                else None,
            },
        }

    async def _tool_update_page(self, args: UpdatePageArgs) -> dict[str, Any]:
        """Apply anchored edits to a page. The agent never sends a whole body.

        THE BASE IS WHAT THE AGENT LAST SAW. A page already staged this drain
        edits on top of the staged text, not the live row -- otherwise a second
        edit to the same page would silently discard the first. An unstaged page
        edits the live body.

        Applying HERE rather than at persist time is what lets a failed anchor
        reach the model: `apply_edits` raises, the harness turns that into a
        tool_validation_error result, and the model gets to re-anchor on the
        next turn. Deferring it to the commit would surface the same failure
        after the conversation had ended, where nothing can act on it.
        """
        key = (args.wiki_type, args.slug)
        if key in self._pending_updates:
            existing = self._pending_updates[key]
            base = existing.body_markdown
            merged_qids = sorted(set(existing.applied_queue_ids) | set(args.applied_queue_ids))
        elif key in self._pending_creates:
            # A page CREATED earlier in this same drain. Without this branch the
            # lookup fell through to the database, found nothing (the create has
            # not been published yet), and told the agent "no such page; use
            # create_page" -- about a page it had just created. The agent's only
            # ways out were to re-create it, losing the edit, or to give up.
            #
            # Edits land on the staged create's body and stay in
            # `_pending_creates`: the page is still new, so it must publish as a
            # create. Promoting it to `_pending_updates` here would stage the
            # same page in both maps, which the preflight then refuses.
            staged_create = self._pending_creates[key]
            base = staged_create.body_markdown
            merged_qids = sorted(set(staged_create.applied_queue_ids) | set(args.applied_queue_ids))
        else:
            page = await persistence.fetch_existing_page(
                self.customer_id, args.wiki_type, args.slug
            )
            if page is None:
                raise ToolValidationError(
                    f"no wiki page {args.wiki_type}/{args.slug}; use create_page for a new one"
                )
            base = page.get("body") or ""
            merged_qids = sorted(set(args.applied_queue_ids))

        try:
            body = apply_edits(
                base,
                [PageEdit(op=e.op, find=e.find, text=e.text) for e in args.edits],
            )
        except EditError as exc:
            # The model's mistake, and it can fix it: say which edit and why.
            raise ToolValidationError(str(exc)) from exc

        over = staged_graph.page_over_cap(
            staged_graph.StagedPage(
                wiki_type=args.wiki_type,
                slug=args.slug,
                body=body,
                is_new=False,
                # What the edit is replacing, so `page_over_cap` can grandfather
                # a shrinking edit to an already-over-cap page. Passing it here
                # rather than deciding here keeps the tool and the preflight on
                # one rule.
                prev_size_bytes=len(base.encode("utf-8")),
            )
        )
        if over is not None:
            # HERE, not only at preflight, for the reason this method already
            # gives about anchors: a refusal at commit time reaches the model
            # after the conversation has ended, where nothing can act on it.
            # Told now, it can split the page and carry on in the same drain.
            return {
                "error": "page_over_cap",
                "wiki_type": args.wiki_type,
                "slug": args.slug,
                "hint": over.detail,
            }

        if key in self._pending_creates:
            # Edits to a page created THIS drain fold back into the create. It
            # has never been published, so it is still a create; staging it in
            # `_pending_updates` as well would put the same page in both maps
            # and the preflight would (correctly) refuse the batch.
            staged_create = self._pending_creates[key]
            self._pending_creates[key] = _StagedCreate(
                wiki_type=staged_create.wiki_type,
                slug=staged_create.slug,
                title=staged_create.title,
                body_markdown=body,
                summary=args.summary,
                frontmatter=dict(staged_create.frontmatter),
                commit_message=args.commit_message,
                applied_queue_ids=merged_qids,
                has_folded_edits=True,
            )
        else:
            self._pending_updates[key] = _StagedUpdate(
                wiki_type=args.wiki_type,
                slug=args.slug,
                body_markdown=body,
                summary=args.summary,
                commit_message=args.commit_message,
                applied_queue_ids=merged_qids,
                prev_size_bytes=len(base.encode("utf-8")),
            )
        # Track the union for excluded_queue_ids on the next next_events.
        # Skip wins over apply per spec, so don't add ids that are
        # already in skipped.
        for qid in args.applied_queue_ids:
            if qid not in self._skipped_queue_ids:
                self._applied_queue_ids.add(qid)
        return {
            "status": "staged",
            "slug": args.slug,
            "edits_applied": len(args.edits),
            # The model asked for a change, not a rewrite. Reporting the size
            # delta lets it notice an anchor that matched more than it meant.
            "chars_before": len(base),
            "chars_after": len(body),
            "pages_pending": self.pending_update_count,
            "events_applied_total": len(self._applied_queue_ids),
        }

    async def _tool_create_page(self, args: CreatePageArgs) -> dict[str, Any]:
        key = (args.wiki_type, args.slug)
        # If the slug already exists on disk, the agent must call
        # update_page instead. Detect by re-checking persistence.
        if key not in self._pending_creates:
            existing = await persistence.fetch_existing_page(
                self.customer_id, args.wiki_type, args.slug
            )
            if existing is not None:
                return {
                    "error": "slug_exists",
                    "wiki_type": args.wiki_type,
                    "slug": args.slug,
                    "hint": "call update_page to modify; create_page rejects existing slugs",
                }

        over = staged_graph.page_over_cap(
            staged_graph.StagedPage(
                wiki_type=args.wiki_type, slug=args.slug, body=args.body_markdown, is_new=True
            )
        )
        if over is not None:
            # The create path never touches `apply_edits`, so without this the
            # cap would only be enforced at preflight -- which is to say, on a
            # brand-new page, too late for the model to do anything about.
            return {
                "error": "page_over_cap",
                "wiki_type": args.wiki_type,
                "slug": args.slug,
                "hint": over.detail,
            }

        if key in self._pending_creates:
            existing_c = self._pending_creates[key]
            if existing_c.has_folded_edits:
                # Re-creating over edits would drop them silently AND union the
                # queue ids, so the folded events commit as done with their
                # content nowhere. Refuse and say what to do instead: the page
                # is already staged, so the model wants update_page.
                return {
                    "error": "staged_create_has_edits",
                    "wiki_type": args.wiki_type,
                    "slug": args.slug,
                    "hint": (
                        f"{args.wiki_type}/{args.slug} is already staged this drain and has "
                        "edits applied to it. create_page would replace the body and lose "
                        "them. Use update_page, or read_page first if you need the current "
                        "staged text."
                    ),
                }
            merged_qids = sorted(set(existing_c.applied_queue_ids) | set(args.applied_queue_ids))
        else:
            merged_qids = sorted(set(args.applied_queue_ids))
        self._pending_creates[key] = _StagedCreate(
            wiki_type=args.wiki_type,
            slug=args.slug,
            title=args.title,
            body_markdown=args.body_markdown,
            summary=args.summary,
            frontmatter=dict(args.frontmatter),
            commit_message=args.commit_message,
            applied_queue_ids=merged_qids,
        )
        for qid in args.applied_queue_ids:
            if qid not in self._skipped_queue_ids:
                self._applied_queue_ids.add(qid)
        return {
            "status": "staged",
            "slug": args.slug,
            "pages_pending": self.pending_update_count,
            "events_applied_total": len(self._applied_queue_ids),
        }

    async def _tool_skip_events(self, args: SkipEventsArgs) -> dict[str, Any]:
        # Skip wins over apply: any qid the agent skips is removed from
        # the applied set, so a later re-stage of update_page can't
        # rescue it.
        added = 0
        for qid in args.queue_ids:
            self._skipped_queue_ids.add(qid)
            if qid in self._applied_queue_ids:
                self._applied_queue_ids.discard(qid)
            added += 1
        # Walk staged updates / creates and remove any qid the agent
        # has now skipped from their applied_queue_ids list. This is
        # the conservative path: skip wins.
        for staged in self._pending_updates.values():
            staged.applied_queue_ids = [
                q for q in staged.applied_queue_ids if q not in self._skipped_queue_ids
            ]
        for staged_c in self._pending_creates.values():
            staged_c.applied_queue_ids = [
                q for q in staged_c.applied_queue_ids if q not in self._skipped_queue_ids
            ]
        log.info(
            "agent.skip_events",
            customer=self.customer_id,
            agent_run_id=self.agent_run_id,
            count=added,
            reason=args.reason,
        )
        return {
            "status": "marked",
            "skipped_count": added,
            "total_skipped": len(self._skipped_queue_ids),
        }

    async def _tool_done(self, args: DoneArgs) -> dict[str, Any]:
        await self.commit()
        self.is_done = True
        return {
            "committed": True,
            "pages_updated": len(self._pending_updates),
            "pages_created": len(self._pending_creates),
            "events_applied": len(self._applied_queue_ids),
            "events_skipped": len(self._skipped_queue_ids),
        }

    # -----------------------------------------------------------------------
    # Commit / discard
    # -----------------------------------------------------------------------

    def _staged_pages(self) -> list[staged_graph.StagedPage]:
        """The whole staged set, creates and updates together, in one view.

        The thing that did not exist before: every rule that spans pages needed
        this and there was nowhere to get it, so the rules went into modules
        that could only see one page (or could only run after the write).
        """
        return [
            staged_graph.StagedPage(
                wiki_type=c.wiki_type, slug=c.slug, body=c.body_markdown, is_new=True
            )
            for c in self._pending_creates.values()
        ] + [
            staged_graph.StagedPage(
                wiki_type=u.wiki_type,
                slug=u.slug,
                body=u.body_markdown,
                is_new=False,
                prev_size_bytes=u.prev_size_bytes,
            )
            for u in self._pending_updates.values()
        ]

    async def _persist_staged_batch(self) -> None:
        """Preflight the whole staged set, then publish it in order.

        Shared with `BackfillWikiRuntime`, which overrides `commit()` to skip
        the queue and index steps. Before this existed the subclass carried its
        own copy of the persist loop, so it validated nothing and published in
        the old order -- on the path that creates the MOST new pages. Any rule
        added to `validate_batch` would have been silently unenforced there.

        A refusal RAISES. It must not return normally: `_tool_done` sets
        `is_done` and reports `committed: True` off the back of this call, the
        harness reads `is_done` as a clean finish, and the worker then closes
        the run `complete` with `error=None`. The queue rows would be DLQ'd
        underneath all of that. `synthesis_worker` says exactly why that is not
        allowed, a few lines below where it catches this: "a green signal over
        work that did not happen". AgentHaltError is the channel that already
        routes to DLQ-with-reason and a failed run status, so the refusal uses
        it rather than inventing a quieter one.
        """
        staged = self._staged_pages()
        # Read the live subpage edges so depth and cycles are checked against
        # the tree that will exist rather than the slice of it in this batch.
        # One query per drain, at the point where a wrong answer is expensive.
        # Live pages as well as live edges. Existence has to be answered by the
        # PAGE set: `live_parents` only holds pages that have a parent, so using
        # it as the existence check falsely orphaned every published top-level
        # page -- including the ordinary case of adopting one as a subpage.
        index = await self.wiki_index()
        violations = staged_graph.validate_batch(
            staged,
            live_parents=await persistence.fetch_subpage_parents(self.customer_id),
            live_pages=[
                (row["wiki_type"], row["slug"])
                for row in index
                if row.get("wiki_type") and row.get("slug")
            ],
        )
        if violations:
            log.warning(
                "agent.preflight_refused",
                customer=self.customer_id,
                agent_run_id=self.agent_run_id,
                violations=[{"rule": v.rule, "ref": v.ref, "detail": v.detail} for v in violations],
            )
            # The rules go in the reason so the DLQ row says WHY, not just
            # "preflight". The worker does the DLQ and the discard; doing
            # either here as well would double-count the rows.
            raise AgentHaltError(
                f"agent.preflight_refused: {', '.join(sorted({v.rule for v in violations}))}"
            )

        for page in staged_graph.publish_order(staged):
            if page.is_new:
                await self._persist_create(self._pending_creates[page.key])
            else:
                await self._persist_update(self._pending_updates[page.key])

    async def commit(self) -> None:
        """Atomic commit of all staged updates + creates.

        For each staged page, build a synthetic WebhookEvent and call
        Normalizer._persist (same path the manual upload route uses).
        Mark queue rows 'done' (applied) or 'synthesis_skipped' (skipped).
        Regenerate the wiki.index page from the live set.

        PREFLIGHT RUNS BEFORE THE FIRST WRITE. Rules that span more than one
        page have nowhere else to live: `apply_edits` sees a single body and the
        create path bypasses it entirely, while link persistence happens after
        the page is already published. See `staged_graph`. A batch that violates
        one is refused whole -- nothing is written, the events are DLQ'd with
        the rule that refused them, and the agent sees why on its next drain.

        PUBLISH ORDER IS CREATES FIRST. A page created and linked from a page
        updated in the same batch has to exist by the time the linking page
        persists its links. The previous order (updates, then creates) had this
        backwards.

        This isn't a true single-DB-transaction (Normalizer does its own
        Phase A/B split for embedding cost), but the queue mark-done
        runs after every page persist succeeds, so a partial failure
        in any page rolls back the whole drain (the tool_exception
        surfaces back to the agent, which can decide to skip the
        offending events and retry).

        The honest limit: preflight makes VALIDATION all-or-nothing, not
        persistence. A batch that passes and then fails on its third page leaves
        the first two published, because each document persists in its own
        transaction and there is no batch transaction to enroll them in.
        """
        await self._persist_staged_batch()

        applied_qids = sorted(self._applied_queue_ids - self._skipped_queue_ids)
        skipped_qids = sorted(self._skipped_queue_ids)
        if applied_qids:
            await persistence.mark_synthesis_done(self.customer_id, applied_qids, self._run_id)
        if skipped_qids:
            await persistence.mark_synthesis_skipped(
                self.customer_id,
                skipped_qids,
                self._run_id,
                reason="agent skipped",
            )
        # Mark any rows that the agent neither applied nor explicitly
        # skipped — they're still 'synthesizing'. Treat them as
        # implicit skips; the agent decided not to use them.
        await self._mark_residual_synthesizing_as_skipped()
        # Regenerate the wiki index after the customer drain. Same
        # convention as v3's synthesis_worker — this is best-effort;
        # the read endpoint has a fallback when the index doesn't
        # exist yet.
        try:
            await self._regenerate_index()
        except Exception as exc:
            log.warning(
                "agent.index_regen_failed",
                customer=self.customer_id,
                agent_run_id=self.agent_run_id,
                error=str(exc),
            )

    async def discard(self) -> None:
        """Drop staged updates / creates, DLQ the in-flight 'synthesizing'
        rows. Called by the worker after AgentLoop raises AgentHaltError.
        """
        log.info(
            "agent.discard",
            customer=self.customer_id,
            agent_run_id=self.agent_run_id,
            staged_updates=len(self._pending_updates),
            staged_creates=len(self._pending_creates),
        )
        self._pending_updates.clear()
        self._pending_creates.clear()
        # The worker calls dlq_agent_synthesizing_rows separately so
        # the dlq_reason carries the categorized halt reason; this
        # method is purely the in-memory cleanup half.

    # -----------------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------------

    async def _persist_update(self, update: _StagedUpdate) -> None:
        # Per-page advisory lock holds for the entire read-then-write so
        # two cross-machine writers can't race on the same (customer,
        # wiki_type, slug). Blocks (not try_) — the second writer waits
        # for the first's commit, then sees the latest content. Note:
        # Normalizer._persist opens its own with_tenant txn, so this
        # lock-holder conn is a separate session whose only job is
        # serialization across the cluster. The advisory lock is global,
        # so it works regardless.
        page_slug = f"{update.wiki_type}:{update.slug}"
        lock_key = advisory_lock_key("page", self.customer_id, page_slug)
        async with with_tenant(self.customer_id) as lock_conn:
            await lock_conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)
            # Re-fetch the existing page AFTER the lock so we see the latest
            # committed state, and so the pipeline_updates check below cannot
            # race a toggle that landed while this run was thinking.
            existing = await persistence.fetch_existing_page(
                self.customer_id, update.wiki_type, update.slug
            )
            if existing is None:
                log.warning(
                    "agent.update_target_missing",
                    customer=self.customer_id,
                    wiki_type=update.wiki_type,
                    slug=update.slug,
                )
                return
            # THE ONLY THING THAT STOPS A REWRITE IS THE EXPLICIT SETTING.
            #
            # This used to read `doc_class == MANUAL_ENTRY`, which meant a
            # person fixing a typo froze the page forever with no way back
            # short of SQL. A hand edit is evidence about what the page should
            # say, not an instruction to stop maintaining it -- the agent has
            # already read that text through `read_page` and the prompt binds
            # it to treat it as authoritative. So editing no longer freezes;
            # only asking to freeze freezes.
            if existing.get("pipeline_updates") is False:
                log.info(
                    "agent.skipped_pipeline_updates_off",
                    customer=self.customer_id,
                    wiki_type=update.wiki_type,
                    slug=update.slug,
                )
                return
            # Reuse the existing page's frontmatter for BOTH the page write
            # and the link-graph extraction. _StagedUpdate has no frontmatter
            # of its own; the prior page's frontmatter is what stays on disk
            # and what the link writer must mirror, otherwise frontmatter-
            # derived rows in wiki_links get wiped on every body-only update.
            existing_frontmatter: dict[str, Any] = existing.get("frontmatter") or {}
            event = self._build_wiki_event(
                wiki_type=update.wiki_type,
                slug=update.slug,
                title=existing.get("title") or "",
                body=update.body_markdown,
                frontmatter=existing_frontmatter,
                summary=update.summary,
                commit_message=update.commit_message,
                compiled_from_doc_ids=[],
                doc_class=DocClass.COMPILED_WIKI,
            )
            norm: NormalizationResult = build_normalization_result(event)
            await self._normalizer._persist(self.customer_id, SourceSystem.WIKI, norm)
            # Lane B: extract typed links from the body + the (preserved)
            # frontmatter and replace this page's markdown+frontmatter rows
            # in wiki_links. Best-effort — page persist already committed.
            await self._persist_links_safely(
                wiki_type=update.wiki_type,
                slug=update.slug,
                body_markdown=update.body_markdown,
                frontmatter=existing_frontmatter,
            )
        # lock auto-releases on with_tenant's txn commit at scope exit.

        # Directed-vector trigger phrases run OUTSIDE the page-write lock.
        # The persist call hits the LLM (Anthropic round-trip + retries)
        # and we don't want concurrent agents on the same page slug
        # serialized for that multi-second window — the page already
        # committed, the lock has done its job. The directed reconcile
        # uses idempotent ON CONFLICT semantics on (customer, doc, hash),
        # so any racing run lands cleanly.
        await self._persist_directed_safely(
            wiki_type=update.wiki_type,
            slug=update.slug,
            title=existing.get("title") or "",
            body_markdown=update.body_markdown,
            frontmatter=existing_frontmatter,
        )

    async def _persist_create(self, create: _StagedCreate) -> None:
        # Same per-page lock as _persist_update. If a concurrent writer
        # already created this slug (UNIQUE collision on the documents
        # row), Normalizer._persist's INSERT-then-UPDATE shape handles
        # the race; the lock just makes that path rare enough that
        # logs stay quiet.
        page_slug = f"{create.wiki_type}:{create.slug}"
        lock_key = advisory_lock_key("page", self.customer_id, page_slug)
        async with with_tenant(self.customer_id) as lock_conn:
            await lock_conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)
            event = self._build_wiki_event(
                wiki_type=create.wiki_type,
                slug=create.slug,
                title=create.title,
                body=create.body_markdown,
                frontmatter=create.frontmatter,
                summary=create.summary,
                commit_message=create.commit_message,
                compiled_from_doc_ids=[],
                doc_class=DocClass.COMPILED_WIKI,
            )
            norm: NormalizationResult = build_normalization_result(event)
            await self._normalizer._persist(self.customer_id, SourceSystem.WIKI, norm)
            # Lane B: extract typed links from body + frontmatter. Best-effort.
            await self._persist_links_safely(
                wiki_type=create.wiki_type,
                slug=create.slug,
                body_markdown=create.body_markdown,
                frontmatter=create.frontmatter,
            )

        # Directed-vector trigger phrases run OUTSIDE the page-write lock
        # so the multi-second LLM call doesn't serialize concurrent
        # agents on the same slug. Reconcile is idempotent on
        # (customer, doc, hash); a racing run lands cleanly.
        await self._persist_directed_safely(
            wiki_type=create.wiki_type,
            slug=create.slug,
            title=create.title,
            body_markdown=create.body_markdown,
            frontmatter=create.frontmatter,
        )

    async def _persist_links_safely(
        self,
        *,
        wiki_type: str,
        slug: str,
        body_markdown: str,
        frontmatter: dict[str, Any],
    ) -> None:
        """Extract + persist typed links for a freshly-written wiki page.

        Two-transaction design: Normalizer._persist opens (and closes) its
        own `with_tenant` connection internally, so the page write and
        the link write cannot share a transaction. Link persistence runs
        here as a second tx immediately after the page commits. The page
        is the source of truth; if the link write fails transiently, the
        link graph goes stale but the page is intact.

        Best-effort semantics: only transient / IO errors (asyncpg
        errors, OSError, TimeoutError) are swallowed-with-warning.
        Programmer errors (TypeError, AttributeError, KeyError, ...) from
        a parser bug propagate, so tests catch them rather than silently
        skipping a link write. See the wiki-backfill-plan TODO entry
        ("Link-graph reconciliation cron") for the planned mitigation of
        the staleness window.
        """
        try:
            extracted = extract_links(body_markdown, frontmatter)
            async with with_tenant(self.customer_id) as conn:
                await persist_links_for_page(
                    conn,
                    customer_id=self.customer_id,
                    src_wiki_type=wiki_type,
                    src_slug=slug,
                    extracted=extracted,
                )
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            log.warning(
                "agent.link_persist_failed",
                customer=self.customer_id,
                wiki_type=wiki_type,
                slug=slug,
                error=str(exc),
                error_class=type(exc).__name__,
            )

    async def _persist_directed_safely(
        self,
        *,
        wiki_type: str,
        slug: str,
        title: str,
        body_markdown: str,
        frontmatter: dict[str, Any],
    ) -> None:
        """Reconcile directed_vectors rows for a freshly-written wiki page.

        Calls kb.synthesis.directed_phrases.persist_directed_vectors
        with the page's frontmatter pins + an LLM-generated phrase set.
        Best-effort: any failure logs and is swallowed so the page write
        path stays bulletproof. The directed retriever silently treats a
        missing-rows page as "no booster signal" — same outcome as a
        page with no phrases at all.
        """
        doc_id = f"wiki:{wiki_type}:{slug}"
        try:
            res = await persist_directed_vectors(
                customer_id=self.customer_id,
                doc_id=doc_id,
                page_title=title,
                page_body=body_markdown,
                frontmatter=frontmatter,
                synthesis_run_id=self._run_id,
            )
            log.info(
                "agent.directed_persisted",
                customer=self.customer_id,
                wiki_type=wiki_type,
                slug=slug,
                human_added=res.human_added,
                human_removed=res.human_removed,
                llm_added=res.llm_added,
                llm_removed=res.llm_removed,
                llm_failed=res.llm_failed,
                # Threshold tuning signal: high drop rates (especially
                # llm_dropped_internal) suggest DIRECTED_DEDUPE_COSINE_THRESHOLD
                # is over-pruning legitimate distinct phrasings.
                llm_dropped_vs_human=res.llm_dropped_vs_human,
                llm_dropped_internal=res.llm_dropped_internal,
            )
        except Exception as exc:
            log.warning(
                "agent.directed_persist_failed",
                customer=self.customer_id,
                wiki_type=wiki_type,
                slug=slug,
                error=str(exc),
                error_class=type(exc).__name__,
            )

    def _build_wiki_event(
        self,
        *,
        wiki_type: str,
        slug: str,
        title: str,
        body: str,
        frontmatter: dict[str, Any],
        summary: str,
        commit_message: str,
        compiled_from_doc_ids: list[str],
        doc_class: DocClass,
    ) -> WebhookEvent:
        received_at = datetime.now(UTC)
        compile_trigger = (
            CompileTrigger.SOURCE_UPDATE if self._run_kind == "wake" else CompileTrigger.SCHEDULED
        )
        raw_payload: dict[str, Any] = {
            WIKI_PAYLOAD_KEY: {
                "wiki_type": wiki_type,
                "slug": slug,
                "title": title,
                "body": body,
                "frontmatter": dict(frontmatter),
                "doc_class": doc_class.value,
                "compiled_from_doc_ids": list(compiled_from_doc_ids),
                "compile_trigger": compile_trigger.value,
                "is_delete": False,
                "updated_at": received_at.isoformat(),
                "summary": summary,
                "commit_message": commit_message,
                "commit_author": "agent:wiki-synthesis-cron",
                "commit_run_id": self._run_id,
                "author_id": "agent:wiki-synthesis-cron",
            }
        }
        return WebhookEvent(
            customer_id=self.customer_id,
            source_system=SourceSystem.WIKI,
            source_event_id=f"{wiki_type}:{slug}:edit:{received_at.isoformat()}",
            received_at=received_at,
            payload_s3_key="",
            payload_s3_keys=[],
            raw_payload=raw_payload,
            headers={},
        )

    async def _mark_residual_synthesizing_as_skipped(self) -> None:
        """Any 'synthesizing' rows the agent didn't touch -> skipped."""
        async with with_tenant(self.customer_id) as conn:
            row_qids = await conn.fetch(
                """
                UPDATE wiki_synthesis_queue
                SET status = 'synthesis_skipped',
                    synthesis_run_id = $2,
                    synthesis_completed_at = NOW(),
                    synthesis_error = 'agent did not apply or skip explicitly'
                WHERE customer_id = $1 AND status = 'synthesizing'
                RETURNING queue_id
                """,
                self.customer_id,
                self._run_id,
            )
        if row_qids:
            log.info(
                "agent.residual_marked_skipped",
                customer=self.customer_id,
                agent_run_id=self.agent_run_id,
                count=len(row_qids),
            )

    async def _regenerate_index(self) -> None:
        """Delegate to the standalone regenerator (see `regenerate_wiki_index`).

        Kept as an instance method so existing call sites
        (commit() + crawlers/github.py) compile unchanged. The
        standalone version is what gets called from non-runtime
        paths: the periodic cross-repo refresh and any future admin
        endpoint that wants a fresh index without rerunning the
        whole agent loop.
        """
        await regenerate_wiki_index(
            customer_id=self.customer_id,
            run_id=self._run_id,
            commit_author="agent:wiki-synthesis-cron",
            normalizer=self._normalizer,
        )

    # -----------------------------------------------------------------------
    # Snapshot / restore
    # -----------------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        return {
            "pending_updates": copy.deepcopy(self._pending_updates),
            "pending_creates": copy.deepcopy(self._pending_creates),
            "applied_queue_ids": set(self._applied_queue_ids),
            "skipped_queue_ids": set(self._skipped_queue_ids),
            "is_done": self.is_done,
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self._pending_updates = snapshot["pending_updates"]
        self._pending_creates = snapshot["pending_creates"]
        self._applied_queue_ids = snapshot["applied_queue_ids"]
        self._skipped_queue_ids = snapshot["skipped_queue_ids"]
        self.is_done = snapshot["is_done"]


# ---------------------------------------------------------------------------
# Convenience for tests
# ---------------------------------------------------------------------------


def default_batch_size() -> int:
    return WIKI_AGENT_BATCH_SIZE


def _index_signature(rows: list[dict[str, Any]]) -> str:
    """A stable fingerprint of everything that decides what the index says.

    Sorted so dict/row ordering cannot change it, and covering title and
    summary alongside the body hash because the prompt carries all three -- a
    renamed page must re-render even though its body never moved.

    THE PROMPT AND THE MODEL ARE PART OF THE INPUT. The first version of this
    hashed the pages alone, which quietly broke the gate's own escape hatch: a
    prompt change moved no page, so the signature did not move, so the gate
    skipped the render forever and the change never reached the front page.
    That shipped -- the fix asking the model for resolvable `[[type:slug]]`
    links could not take effect, and the only symptom was a front page that
    stayed subtly wrong with every run reporting a clean skip.

    Hashing the prompt TEXT rather than a hand-maintained version constant is
    deliberate: a constant is a second thing to remember, and the failure mode
    of forgetting it is exactly the silent one above. Edit the prompt and the
    next render happens, with no ceremony.
    """
    import hashlib

    parts = [
        f"__prompt__|{hashlib.sha256(_INDEX_SYSTEM_PROMPT.encode('utf-8')).hexdigest()}",
        f"__model__|{WIKI_AGENT_MODEL}",
    ]
    for row in sorted(rows, key=lambda r: str(r.get("doc_id"))):
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        digest = (
            meta.get("body_sha256")
            or hashlib.sha256((row.get("body") or "").encode("utf-8")).hexdigest()
        )
        parts.append(
            f"{row.get('doc_id')}|{digest}|{row.get('title') or ''}|{meta.get('summary') or ''}"
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


async def regenerate_wiki_index(
    *,
    customer_id: str,
    run_id: int | None = None,
    commit_author: str = "system:wiki-index-regen",
    normalizer: Normalizer | None = None,
    force: bool = False,
) -> None:
    """Regenerate the wiki index page for a customer.

    Reusable entry point — callable from anywhere (the wiki agent
    runtime, the periodic cross-repo refresh, an admin endpoint, etc.)
    Reads live wiki pages, asks the LLM to produce a markdown body
    using verified cross-repo edges as the architecture-diagram
    source-of-truth, and persists the result through the standard
    Normalizer pipeline.

    `run_id` is optional and only used as audit metadata
    (`commit_run_id`). When omitted the commit message reads as a
    standalone refresh rather than a tail of an agent run.

    `normalizer` allows callers to supply a pre-built instance (the
    agent runtime does this so the embedder + R2 client are reused
    across an entire drain). Standalone callers can omit it and the
    function constructs a fresh default.
    """
    if normalizer is None:
        ctx = make_default_context()
        normalizer = Normalizer(ctx, store=get_store(), embedder=get_embedder_v2())

    async with with_tenant(customer_id) as conn:
        # Plan A Component 6: the wiki agent's drain-time index regen
        # only sees approved artifacts. Drafts are reviewer-pending
        # writebacks (Component 5) and must not appear in the auto-index.
        rows = await conn.fetch(
            """
            SELECT doc_id, title, body_preview, source_id, version, updated_at,
                   metadata
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND doc_type LIKE $3
              AND doc_type <> $4
              AND valid_to IS NULL
              AND deleted_at IS NULL
              AND visibility = 'approved'
            ORDER BY updated_at DESC
            """,
            customer_id,
            SourceSystem.WIKI.value,
            f"{WIKI_DOC_TYPE_PREFIX}%",
            WIKI_INDEX_DOC_TYPE,
        )
        # THE BODIES, not just the previews. The front page is a synthesis of
        # what these pages SAY, and the renderer used to get titles and
        # one-line summaries only -- so it could re-list the corpus and
        # nothing more. Read on the same connection the rows came from, so
        # RLS on `chunks` sees the tenant GUC.
        rows = [
            {
                **dict(row),
                # NOT sliced here. This used to cut every body to
                # PER_PAGE_BODY_CHARS + 1 at fetch, to avoid materialising the
                # whole wiki in memory -- a sound instinct aimed at the wrong
                # number. The cut discarded 46% of this team's corpus while the
                # corpus was 35% of the renderer's actual budget, and because it
                # happened HERE the renderer could not see it, report it, or
                # weigh it against the budget it does own.
                #
                # Sizing: the whole wiki is ~141k chars against a 250k ceiling.
                # Holding it briefly is a fraction of what the prompt built from
                # it already costs. If that stops being true, the fix is a
                # streaming assembly in the renderer, not a blind prefix here.
                "body": (await fetch_live_body_from_chunks(conn, customer_id, row["doc_id"]) or ""),
            }
            for row in rows
        ]

    # THE GATE, and it lives HERE rather than at the call sites because there
    # are three of them (commit(), the GitHub crawler, the cross-repo refresh)
    # and gating at each would fix two and leave the third -- the crawler being
    # the easiest to forget.
    #
    # The index is a function of the pages it reads. If none of their content
    # moved since the last render, re-rendering spends a Gemini call to produce
    # a near-identical page and bumps the version chain for nothing; that chain
    # is where someone looks to answer "when did the overview actually change",
    # and it is already at v115.
    #
    # Keyed on `body_sha256` -- the content hash, NOT `content_hash`, which
    # mixes in `received_at` and therefore moves on every write whether or not
    # the text did. Pages with no digest yet (written before that field
    # shipped) hash their body here: the renderer already holds every body in
    # full, so it costs a sha256 over a string already in memory.
    #
    # The page SET is part of the signature, not just the hashes: a page added
    # or deleted changes the index without changing any surviving page's
    # content, and a hash-only gate would skip exactly those.
    # SINGLE-FLIGHT per customer. Two regenerators running at once both read
    # the same signature, both decide to render, and both spend a Gemini call
    # to write the same page -- and the second write bumps the version chain
    # again for content the first already produced.
    #
    # `pg_try_advisory_xact_lock`, not the blocking form: a regen that is
    # already in flight will produce the page this caller wanted, so waiting
    # for it only to render again is worse than standing down. The lock is held
    # for the transaction, so it releases even if this task dies.
    #
    # Reachable today: the nightly trigger, the GitHub crawler and a drain's
    # commit() can all reach `regenerate_wiki_index` for one customer, and the
    # crawler runs alongside the daily drain.
    lock_key = advisory_lock_key("wiki-index-regen", customer_id)
    async with with_tenant(customer_id) as lock_conn:
        got_lock = await lock_conn.fetchval("SELECT pg_try_advisory_xact_lock($1)", lock_key)
        if not got_lock:
            log.info(
                "agent.index_regen_skipped_locked",
                customer=customer_id,
                agent_run_id=run_id,
            )
            return

        signature = _index_signature(rows)
        previous = await persistence.fetch_index_signature(customer_id)
        if not force and previous is not None and previous == signature:
            log.info(
                "agent.index_regen_skipped_unchanged",
                customer=customer_id,
                agent_run_id=run_id,
                page_count=len(rows),
            )
            return

        body = await index_renderer.render_index_via_llm(rows, customer_id=customer_id)
        if body is None:
            # The renderer could not write an overview this run. Leave the
            # published one alone: it is stale by one drain, which is strictly
            # better than replacing a real overview with a placeholder, and far
            # better than the page list the old fallback substituted.
            log.warning(
                "agent.index_regen_skipped_no_body",
                customer=customer_id,
                agent_run_id=run_id,
                page_count=len(rows),
            )
            return
        received_at = datetime.now(UTC)
        run_id_suffix = f" #{run_id}" if run_id is not None else ""
        raw_payload: dict[str, Any] = {
            WIKI_PAYLOAD_KEY: {
                "wiki_type": "index",
                "slug": INDEX_SLUG,
                "title": "Wiki",
                "body": body,
                "frontmatter": {"page_count": len(rows), "index_signature": signature},
                "doc_class": DocClass.AGENT_ARTIFACT.value,
                "is_delete": False,
                "updated_at": received_at.isoformat(),
                "summary": f"Wiki overview ({len(rows)} pages).",
                "commit_message": (f"Regenerate index ({len(rows)} pages){run_id_suffix}"),
                "commit_author": commit_author,
                "commit_run_id": run_id,
                "author_id": commit_author,
            }
        }
        event = WebhookEvent(
            customer_id=customer_id,
            source_system=SourceSystem.WIKI,
            source_event_id=f"index:{INDEX_SLUG}:edit:{received_at.isoformat()}",
            received_at=received_at,
            payload_s3_key="",
            payload_s3_keys=[],
            raw_payload=raw_payload,
            headers={},
        )
        norm: NormalizationResult = build_normalization_result(event)
        await normalizer._persist(customer_id, SourceSystem.WIKI, norm)

        # Persisted INSIDE the lock. Releasing after the render would let a
        # second regenerator read the pre-write signature, conclude nothing
        # changed was false, and render the same page again.


__all__ = [
    "WikiAgentRuntime",
    "default_batch_size",
    "regenerate_wiki_index",
]

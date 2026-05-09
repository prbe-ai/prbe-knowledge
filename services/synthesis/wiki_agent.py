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

from services.ingestion.handlers.base import ConnectorContext, make_default_context
from services.ingestion.handlers.wiki import (
    INDEX_SLUG,
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from services.ingestion.normalizer import Normalizer
from services.synthesis import index_renderer, persistence
from services.synthesis.agent_tools import (
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
from services.synthesis.directed_phrases import persist_directed_vectors
from services.synthesis.wiki_links import extract_links, persist_links_for_page
from shared.constants import (
    WIKI_AGENT_BATCH_SIZE,
    WIKI_DOC_TYPE_PREFIX,
    WIKI_INDEX_DOC_TYPE,
    CompileTrigger,
    DocClass,
    SourceSystem,
)
from shared.db import with_tenant
from shared.embeddings import Embedder
from shared.exceptions import ToolValidationError
from shared.locks import advisory_lock_key
from shared.logging import get_logger
from shared.models import NormalizationResult, WebhookEvent
from shared.storage import ObjectStore, get_store

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
        embedder: Embedder | None = None,
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
        key = (args.wiki_type, args.slug)
        # Last-write-wins on body / summary / commit_message; union-merge
        # applied_queue_ids so a re-stage doesn't drop earlier events.
        if key in self._pending_updates:
            existing = self._pending_updates[key]
            merged_qids = sorted(set(existing.applied_queue_ids) | set(args.applied_queue_ids))
        else:
            merged_qids = sorted(set(args.applied_queue_ids))
        self._pending_updates[key] = _StagedUpdate(
            wiki_type=args.wiki_type,
            slug=args.slug,
            body_markdown=args.body_markdown,
            summary=args.summary,
            commit_message=args.commit_message,
            applied_queue_ids=merged_qids,
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

        if key in self._pending_creates:
            existing_c = self._pending_creates[key]
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

    async def commit(self) -> None:
        """Atomic commit of all staged updates + creates.

        For each staged page, build a synthetic WebhookEvent and call
        Normalizer._persist (same path the manual upload route uses).
        Mark queue rows 'done' (applied) or 'synthesis_skipped' (skipped).
        Regenerate the wiki.index page from the live set.

        This isn't a true single-DB-transaction (Normalizer does its own
        Phase A/B split for embedding cost), but the queue mark-done
        runs after every page persist succeeds, so a partial failure
        in any page rolls back the whole drain (the tool_exception
        surfaces back to the agent, which can decide to skip the
        offending events and retry).
        """
        for update in self._pending_updates.values():
            await self._persist_update(update)
        for create in self._pending_creates.values():
            await self._persist_create(create)

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
            # Re-fetch the existing page so MANUAL_ENTRY pages are skipped
            # (we don't want the agent to clobber a human-authored page).
            # The fetch happens AFTER the lock so it sees the latest
            # committed state.
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
            if existing.get("doc_class") == DocClass.MANUAL_ENTRY.value:
                log.info(
                    "agent.skipped_manual_entry",
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

        Calls services.synthesis.directed_phrases.persist_directed_vectors
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


async def regenerate_wiki_index(
    *,
    customer_id: str,
    run_id: int | None = None,
    commit_author: str = "system:wiki-index-regen",
    normalizer: Normalizer | None = None,
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
        normalizer = Normalizer(ctx, store=get_store(), embedder=Embedder())

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT title, body_preview, source_id, version, updated_at,
                   metadata
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND doc_type LIKE $3
              AND doc_type <> $4
              AND valid_to IS NULL
              AND deleted_at IS NULL
            ORDER BY updated_at DESC
            """,
            customer_id,
            SourceSystem.WIKI.value,
            f"{WIKI_DOC_TYPE_PREFIX}%",
            WIKI_INDEX_DOC_TYPE,
        )
    body = await index_renderer.render_index_via_llm(
        rows, customer_id=customer_id
    )
    received_at = datetime.now(UTC)
    run_id_suffix = f" #{run_id}" if run_id is not None else ""
    raw_payload: dict[str, Any] = {
        WIKI_PAYLOAD_KEY: {
            "wiki_type": "index",
            "slug": INDEX_SLUG,
            "title": "Wiki",
            "body": body,
            "frontmatter": {"page_count": len(rows)},
            "doc_class": DocClass.AGENT_ARTIFACT.value,
            "is_delete": False,
            "updated_at": received_at.isoformat(),
            "summary": f"Wiki overview ({len(rows)} pages).",
            "commit_message": (
                f"Regenerate index ({len(rows)} pages){run_id_suffix}"
            ),
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


__all__ = [
    "WikiAgentRuntime",
    "default_batch_size",
    "regenerate_wiki_index",
]

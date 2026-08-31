"""FastMCP server exposing prbe-knowledge retrieval as MCP tools.

Tools read customer_id from the per-request ContextVar set by the
McpAuthMiddleware (see app/dependencies/auth_context.py). They never
receive customer_id as a tool argument — auth context isn't user input.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent

from engine.mcp.clients._responses import DETAIL_EVIDENCE, VALID_DETAILS, detail_error
from engine.mcp.clients.knowledge import KnowledgeError, get_client
from engine.mcp.consts import (
    ALLOWED_HOSTS,
    ALLOWED_ORIGINS,
    MCP_INSTRUCTIONS,
    MCP_SERVER_NAME,
    PROBE_PLAN_PROMPT_TEMPLATE,
    PROBE_PROMPT_TEMPLATE,
)
from engine.mcp.dependencies.auth_context import get_current_customer
from engine.mcp.services.response_budget import (
    fit_response_to_budget,
    serialize_tool_response,
)

#: Typed so FastMCP advertises the vocabulary in the tool schema and rejects
#: unknown values pre-handler. Must mirror VALID_DETAILS — the compaction
#: test pins the two together (test_detail_literal_matches_the_vocabulary),
#: so a fourth profile that touches one and not the other fails loudly.
DetailMode = Literal["ids", "evidence", "full"]

mcp = FastMCP(
    MCP_SERVER_NAME,
    instructions=MCP_INSTRUCTIONS,
    # Stateless mode: every request is self-contained, no per-worker
    # session affinity. Required because Fly runs uvicorn --workers 2;
    # with the default stateful mode, a session created on worker A
    # 404s when the next request lands on worker B. Our tools
    # (search_knowledge, query_knowledge, get_source) hold no
    # per-session state, so stateless is correct.
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=ALLOWED_HOSTS,
        allowed_origins=ALLOWED_ORIGINS,
    ),
)

# FastMCP 1.27 serializes structured dict returns twice: once as JSON text
# in ``content`` and again in ``structuredContent``. These evidence-heavy
# tools intentionally advertise unstructured output and return pre-serialized
# compact JSON so clients receive one copy whose wire size exactly matches the
# response budget.


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    """Emit exactly one compact JSON payload with explicit MCP error state."""
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=serialize_tool_response(payload),
            )
        ],
        structuredContent=None,
        isError=is_error,
    )


def _error_response(exc: KnowledgeError) -> CallToolResult:
    payload: dict[str, Any] = {"error": str(exc), "status": exc.status}
    if exc.trace_id:
        payload["trace_id"] = exc.trace_id
    return _tool_result(payload, is_error=True)


def _budgeted_response(response: dict[str, Any]) -> CallToolResult:
    fitted = fit_response_to_budget(response)
    is_error = fitted.get("error_code") == "response_too_large"
    return _tool_result(fitted, is_error=is_error)


@mcp.tool(structured_output=False)
async def search_knowledge(
    query: str,
    top_k: int = 5,
    source: str | None = None,
    strict_entity_filtering: bool = False,
    top_k_related: int = 10,
    discovery: bool = False,
    verbose: bool = False,
    detail: DetailMode = DETAIL_EVIDENCE,
) -> CallToolResult:
    """Search team operational history when a concrete history question could
    change the answer or approach.

    Searches the user's team operational memory — Slack threads, GitHub PRs,
    Linear tickets, Notion docs, Sentry incidents. Use it for prior rationale,
    incidents, ownership, constraints, or similar/parallel work—not repo facts,
    routine implementation/review, status, or shipping. Requests, plans, phase
    changes, compaction, and elapsed time are not triggers. Reuse relevant
    results for the same decision.

    Pass a bag of entities/keywords as the query — ticket IDs, repo or
    service names, file/symbol names, error strings, feature flags. NOT a
    question or sentence; prose dilutes BM25, vector, and entity extraction.

      Good: "PRB-17 Linear enrichment per-source toggle workspace_prefs JSONB"
      Bad:  "Why is PRB-17 still considered broken?"

    Surface useful findings. Use `get_source`, retry, or follow related entities
    only when needed to resolve the decision.

    NOT source-code search. For code, read the repo directly.

    Response shape: `results[]`, each Document result carrying its identity
    (`doc_id`, `source_system`, `source_url`, `title`, `score`,
    `chunk_count`) and a nested `chunks[]` array of the matching spans
    within that document. Each chunk carries `content` and, when the
    gatherer wrote one, `why_relevant` — absent keys throughout this
    response mean "nothing here", never "unknown". (One scoped exception:
    a CHUNK's `matched_via` is omitted when identical to its document's,
    so read the document's provenance as covering its chunks.) The
    knowledge-graph evidence trails (`graph_evidence` entries per chunk)
    ride ONLY on `verbose=True`: the top-level `confidence_breakdown`
    already says whether graph evidence exists and at what confidence, so
    re-call with `verbose=True` only when you need the actual edge trails.
    Audit metadata (`author_id`, `created_at`,
    `updated_at`) and full provenance ride only on detail="full" — see
    `detail` below; for time-ordering questions ("which came first?"), ask
    for detail="full" or read timestamps off `get_source`. Top-level
    `confidence_breakdown` is an aggregate count of evidence confidences
    (`EXTRACTED` / `INFERRED` / `AMBIGUOUS`) across all returned chunks —
    a low EXTRACTED ratio means most matches are inferred and you should
    treat the result set as weaker.

    The response also includes `related_entities` — non-Document graph
    nodes attached to the returned docs, each just its identity
    (`canonical_id`, `label`, `display_name`): a crawl-candidate handle to
    drop into a follow-up search. Follow one only when the adjacent context
    is relevant to the current decision. Set `top_k_related=0` to skip this
    enrichment for token-sensitive flows. `related_entities=null` with
    `related_entities_error` present means the walk failed — documents are
    still trustworthy (the error field appears only when it fired).

    CHECK `degraded` FIRST. Top-level boolean. `true` means the search
    agent could not complete normally and this response is the raw
    pre-fan-out pool WITHOUT the agent's curation — it is not the answer a
    healthy run would have given, even though it looks like one and
    arrived as a normal 200.

    Do NOT report "the team has no context on X" off a degraded response.
    You cannot distinguish that from the agent never having run.

    `degraded_reason` says what to do next, and the right action differs:
      * `provider_error_prefanout_fallback`, `loop_timeout` — transient.
        Re-run ONCE. If it degrades again, report the degradation rather
        than retrying further; repeated retries during a provider outage
        add load to the thing that is already failing.
      * `context_overflow`, `tool_budget_exceeded`, `output_truncated` —
        deterministic. The identical request will fail identically, so
        re-running is pure waste. NARROW instead: lower `top_k`, tighten
        the keyword bag. (`output_truncated` is deterministic because the
        gatherer runs at temperature=0 with a query-derived seed, so the
        same input reproduces the same over-long emit.)
      * `channel_degraded` — one of the four retrieval channels failed and
        contributed nothing, so the answer came from a strictly smaller
        candidate pool. The results you got are REAL, just incomplete.
        Transient (usually a database statement timeout). Re-run ONCE. If it
        repeats, say the search was partial rather than reporting the gap as
        an absence — this is the one reason where "I found nothing about X"
        is most likely to be wrong.
      * `no_llm_configured` — deployment-level. Do not retry. Surface it.
      * `schema_violation`, `passthrough_harness_fallback` — the agent
        returned something unusable. Re-run ONCE, then treat as transient
        if it repeats.
      * anything else — treat as transient, re-run once, do not loop.

    `false` on healthy runs and on honestly-empty results.

    `gatherer_notes` — self-reported metadata from the search agent
    (gatherer). When present, surface `gatherer_notes.confidence`
    ("high" / "medium" / "low") to decide how much to trust the result
    set. `high` means the gatherer's turn-1 fan-out clearly answered
    the query and you surfaced strong matches. `medium` means
    exploration helped but not all leads resolved. `low` means
    turn-1 came back thin and exploration didn't surface anchors —
    consider rephrasing the query or raising `top_k`. The list
    `gatherer_notes.dropped` enumerates candidates the agent saw but
    chose not to surface, with a one-line reason — useful when a result
    you expected to be there isn't. Absent on non-gatherer paths
    (legacy router responses pre-cutover).

    The response is byte-budgeted (~20KB target, 24KB hard) so it
    never trips the MCP harness disk-spill fallback. If the underlying
    retrieval would have returned more, you'll see `truncated: true`
    plus counts for dropped chunks, whole results, and related entities.
    Those were removed from the lowest-ranked tail first. If you need
    them anyway, lower `top_k` for a tighter focused query, or call
    `get_source` on specific `doc_id`s. `cursor` is reserved for future
    stateful continuation; today it's always null.

    At detail="full", each document carries `author_id` — the raw author
    identifier from the source system (GitHub login, commit-author email,
    Slack user id, Linear user id); the default detail="evidence" omits it.
    It is NOT canonicalized: the same person can
    appear under multiple values across sources, and even within one
    source (e.g. a GitHub commit may surface as `mahit` when the email
    resolved to a login or as `mahit@example.com` when it didn't). Use
    it as a strong-but-fuzzy signal, not an identity primary key.
    `null` when the source had no author. For commits, additional
    co-authors from `Co-authored-by:` trailers are returned via
    `get_source` under `metadata.co_authors`.

    Args:
        query: Bag of entities/keywords (ticket IDs, repo/service names,
            file or symbol names, error strings, feature flags). NOT a
            question or sentence — prose dilutes BM25 and the vector.
        top_k: How many documents to return. Default 5, max 50. Each
            document may contain multiple matching chunks, so the total
            chunk count is typically higher than `top_k`. This is your
            recall dial. Raise it only when missing expected context would
            block or materially change the current decision.
        source: Optional filter — "slack", "github", "linear", "notion",
            "sentry". Omit to search across all connected sources.
        strict_entity_filtering: Default False — broad recall, pure
            vector + BM25 + graph fusion, accepts some noise. Turn ON
            (True) when your query names a specific entity (project,
            person, ticket ID, repo, channel) and you're getting hits
            that look semantically similar but aren't actually about
            that entity — e.g. "what's going on with klavis" matching
            generic Slack greetings on conversational shape. With it
            on, results that don't textually contain the router-
            extracted entity's canonical_id or display_name get
            dropped. Don't turn it on for vague/exploratory queries:
            if the entity extractor misfires or the canonical form
            isn't in the docs, you'll zero out the result set.
        top_k_related: How many `related_entities` to return as crawl
            candidates. Default 10, max 20. Set to 0 to skip the graph
            walk entirely for token-sensitive flows. Returned entries
            are non-Document graph nodes attached to the result-set
            docs, ranked by IDF-adjusted `score` so generic high-
            degree entities (e.g. busy channels, prolific people) are
            demoted in favor of specific ones. Follow one with another
            search only when its context is relevant to the decision.
        discovery: Default False (focus mode). Set True for **discovery
            mode**: the graph channel gets a wider retrieval budget, so
            more of its surprise-ranked tail reaches the answer. Graph
            hits are ordered by a per-edge surprise score that rewards
            cross-source and cross-community edges and demotes edges
            between two hub nodes, so widening the budget surfaces
            entity-anchored canonical docs (the actual PR, ticket,
            design rationale, runbook) rather than the hub-to-hub
            index docs that connect to everything. Vector and BM25
            are unaffected.

            Default to True for most queries against this corpus.
            Empirically (post-anti-bonus, 6 paired acme
            queries): 5/6 cases see canonical PRs/commits/Notion docs
            move into top-3 that focus mode buried at rank 6+ behind
            transcripts or hub anchors. The 1 neutral case was already
            surfacing the right canonical doc at top-1, so discovery
            had nothing to fix.

            Use it when:
              - You want the canonical answer (PR, commit, design
                doc, ticket, runbook) and the corpus has recent
                claude_code/codex/Slack transcripts that semantically
                match the query — discovery cuts through that noise.
              - The query is conceptual ("how should we approach X",
                "what's blocking Y", "design rationale for Z") and
                you want the entity-anchored discussion above
                ambient chatter.
              - You ran focus mode and got transcript-shaped results
                where you expected PR/commit/Notion-shaped ones.
              - You want adjacent context: design rationale for a
                code change, Slack thread about a ticket, sibling
                tickets sharing a service.

            Skip it when:
              - The query is already returning the canonical answer
                at top-1 in focus mode and you're token-sensitive.
              - You explicitly want recent activity / transcripts as
                primary results, not the canonical artifact behind
                them.

            How it can fail: if the router extracts only entity types
            with no graph nodes at ingest (`feature`, `decision`),
            the graph contributes nothing and discovery returns
            identical results to focus mode — no harm, no gain.
            Rephrase with concrete entity terms (PR#, repo, service,
            person, ticket) to give the graph something to anchor on.
            Cheap to flip and retry.
        verbose: Default False — strips diagnostic fields agents
            don't reason over (timing, ranks, per-retriever score
            breakdown). The opaque `trace_id` stays for log correlation.
            Top-line `score`,
            `total_candidates`, `extracted_entities`, and
            `applied_temporal` stay so the caller can tell when to
            raise top_k or when the router misinterpreted the query.
            Set True for the full upstream payload when debugging;
            verbose=True outranks `detail`.
        detail: Your altitude dial — how much of each result to return.
            The ENVELOPE (`degraded`, `truncated`, `confidence_breakdown`,
            `total_candidates`, ...) is identical at every detail; only the
            rows inside `results` change, so a partial or degraded answer can
            never masquerade as a complete one by being asked for leaner.
              "evidence" (default): doc identity (`doc_id`, `title`,
                `source_url`, `score`) + full chunk content. Drops per-doc
                audit metadata (`created_at`/`updated_at`/`author_id`) and
                provenance boilerplate; provenance that arrived over a
                knowledge-graph edge (`edge_type`/`why`) is always kept.
                Everything dropped is one detail="full" call away, and
                `doc_id` + `get_source` reach the entire document.
              "ids": triage — doc identity and scores, no chunk content.
                For "did the lab touch X at all?" sweeps before spending
                tokens reading evidence.
              "full": every field the compact response carries, including
                audit metadata and complete provenance. The debugging and
                audit shape.
            Leaner details also survive the response byte budget better: the
            budget trims tail chunks and then whole documents to fit, so at
            "evidence" more of your actual hits make it under the cap.
    """
    if detail not in VALID_DETAILS:
        return _tool_result(
            {"error": detail_error(detail), "status": 422},
            is_error=True,
        )
    customer_id = get_current_customer()
    client = get_client()
    sources = [source] if source else None
    try:
        response = await client.retrieve(
            query=query,
            customer_id=customer_id,
            top_k=min(top_k, 50),
            sources=sources,
            entity_must_match=strict_entity_filtering,
            top_k_related=min(max(top_k_related, 0), 20),
            discovery=discovery,
            verbose=verbose,
            detail=detail,
        )
    except KnowledgeError as exc:
        return _error_response(exc)
    return _budgeted_response(response)


@mcp.tool(structured_output=False)
async def query_knowledge(
    question: str,
    top_k: int = 5,
    strict_entity_filtering: bool = False,
    discovery: bool = False,
    top_k_related: int = 0,
    verbose: bool = False,
) -> CallToolResult:
    """Use when the user asks a direct question and wants a synthesized
    answer with citations — not evidence for you to reason over.

    Runs the same retrieval as `search_knowledge` then asks an LLM to
    synthesize a concise answer with inline citations. Don't pre-summarize
    on top of it; surface the answer (and let the user click through the
    citations).

    For agent reasoning about team history, prefer `search_knowledge`; it
    exposes the evidence without an LLM in the middle.

    Response shape: `answer` (string), `citations` (list referencing the
    underlying documents), `insufficient_context` (bool — true when the
    LLM couldn't find enough grounded evidence and refused to guess),
    `model` (which LLM produced the answer), and the full retrieval
    payload as doc-grouped `results[]` with nested `chunks[]`, each chunk
    carrying `content`, plus `why_relevant` (the gatherer's per-chunk
    rationale) and chunk-level `matched_via` when populated and distinct
    from the document's — absent keys mean "nothing here", never
    "unknown". Graph-evidence trails are not included (the synthesis
    already consumed them; `confidence_breakdown` summarizes them). These
    rows match `search_knowledge` at detail="full" (this tool has no
    detail parameter and keeps the audit metadata the search default
    omits). When `top_k_related >= 1`, also carries
    top-level `related_entities` + `query_root_doc_id` + `gatherer_notes`.
    When `insufficient_context=true`, surface that refusal to the user
    instead of paraphrasing it.

    CHECK `degraded` FIRST — same top-level flag and same semantics as
    `search_knowledge` (see that tool's docstring for the per-reason
    action table). It matters MORE here: this tool runs an LLM over the
    evidence, so a degraded, uncurated pool comes back as fluent, confident
    prose with citations. The synthesis hides the degradation that raw
    results would have made obvious.

    The two flags interact, and the combination is a trap. When
    `degraded=true` AND `insufficient_context=true`, do NOT surface the
    refusal as "the team has no context on this" — a degraded run means the
    agent never curated, so a refusal is indistinguishable from it never
    having looked. Act on `degraded_reason` first; only treat
    `insufficient_context` as a real answer when `degraded` is false.

    Args:
        question: Natural-language question, ideally how the user phrased it.
        top_k: How many documents to feed the LLM. Default 5, max 50.
            Each document may carry multiple chunks, so the LLM sees
            more spans than `top_k`. Raise it (e.g. 15 or 50) when the
            synthesized answer is missing context you'd expect to be
            there — more documents = more recall for the LLM to draw
            from, at the cost of a longer prompt.
        strict_entity_filtering: Default False — broad recall, lets the
            LLM synthesize over pure vector + BM25 + graph fusion. Turn
            ON (True) when the question names a specific entity
            (project, person, ticket ID, repo, channel) and the
            synthesized answer is drifting onto chunks that look
            semantically similar but aren't actually about that
            entity. With it on, results that don't textually contain
            the router-extracted entity's canonical_id or display_name
            get dropped before synthesis. Don't turn it on for
            vague/exploratory questions: if the entity extractor
            misfires or the canonical form isn't in the docs, the
            answer comes back empty.
        discovery: Default False (focus mode). Set True to widen the
            graph channel's retrieval budget so more of its
            surprise-ranked tail reaches the LLM — surfacing
            entity-anchored canonical docs (the actual PR, commit,
            design doc, ticket, runbook) rather than the hub-to-hub
            index docs that connect to everything. Vector and BM25
            budgets are unchanged. Default to True
            for most questions against this corpus. Use for conceptual
            questions ("how should we approach X", "what's blocking
            Y") and any case where focus mode is returning transcript-
            or hub-shaped evidence when you expected the canonical
            artifact. Skip when the canonical answer is already at
            top-1 in focus mode and you're token-sensitive. Same
            toggle and caveats as `search_knowledge`'s discovery flag.
        top_k_related: Default 0 — synthesis itself doesn't read
            related_entities, so the BFS walk that populates them is
            skipped by default to save one DB round-trip per /query.
            Set >= 1 to populate `related_entities` (graph nodes
            attached to result docs, useful for the agent to crawl
            laterally) alongside the synthesized answer. Max 50.
        verbose: Default False — strips diagnostic fields (timing and
            applied filters) from the response. The opaque `trace_id`
            stays for log correlation. Set True only when debugging.
    """
    customer_id = get_current_customer()
    client = get_client()
    try:
        response = await client.query(
            question=question,
            customer_id=customer_id,
            top_k=min(top_k, 50),
            entity_must_match=strict_entity_filtering,
            discovery=discovery,
            top_k_related=min(top_k_related, 50),
            verbose=verbose,
        )
    except KnowledgeError as exc:
        return _error_response(exc)
    # Same byte cap as search_knowledge: query_knowledge returns the
    # same `results[]` evidence shape (alongside `answer` + `citations`).
    # The synthesized `answer` field is bounded by the LLM; the heavy
    # part is the documents.
    return _budgeted_response(response)


SourceViewMode = Literal[
    "preview", "search", "grep", "range", "chunk", "tail", "full"
]


@mcp.tool(structured_output=False)
async def get_source(
    doc_id: str,
    mode: SourceViewMode = "preview",
    query: str | None = None,
    pattern: str | None = None,
    start_line: int | None = None,
    limit_lines: int = 80,
    chunk_index: int | None = None,
    context_lines: int = 3,
    max_matches: int = 20,
    cursor: str | None = None,
    verbose: bool = False,
) -> CallToolResult:
    """Use AFTER `search_knowledge` when a returned chunk looks relevant and
    you need broader context from the same source.

    Defaults to a bounded preview. Use modes to drill down safely:
    - preview: first lines of the source (default)
    - search: chunked in-source search; requires `query`
    - grep: literal case-insensitive line search; requires `pattern`
    - range: read up to `limit_lines` from `start_line` or `cursor`
    - chunk: read one ingested chunk; requires `chunk_index`
    - tail: last lines of the source
    - full: whole document only when it fits the MCP response budget.
      Oversized documents return a 413 with guidance to use a bounded mode.
      Use only when you genuinely need broad context or the user asks. A
      preview + targeted `search`/`grep` is usually enough.

    The `doc_id` is the value `search_knowledge` returned in
    a Document entry under `results[].doc_id` — typically a string like
    "linear:org-acme:issue:uuid-9821" or
    "slack:T_ACME:C_GENERAL:1714000000.123".

    Includes `author_id` at the top level (the same raw form
    `search_knowledge` returns at detail="full"). The response includes navigation
    metadata such as `sections`, `line_start`, `line_end`,
    `total_lines`, `next_cursor`, `truncated`, `chunk_count`, and
    `body_size_bytes`.

    Args:
        doc_id: Identifier from a `search_knowledge` document.
        mode: Source-reading mode. Default `preview`.
        query: Required for `mode="search"`.
        pattern: Required for `mode="grep"`; literal, not regex.
        start_line: 1-based line start for `mode="range"`.
        limit_lines: Max lines per returned section. Server max is 100.
            Ignored in `mode="full"`.
        chunk_index: 0-based chunk index for `mode="chunk"`.
        context_lines: Lines around grep matches. Server max is 20.
        max_matches: Max search/grep sections. Server max is 50.
        cursor: Continuation cursor returned by a prior bounded view.
        verbose: Default False — strips source-system internals if present.
            It does not bypass server safety limits.
    """
    customer_id = get_current_customer()
    client = get_client()
    try:
        response = await client.get_source(
            doc_id=doc_id,
            customer_id=customer_id,
            mode=mode,
            query=query,
            pattern=pattern,
            start_line=start_line,
            limit_lines=limit_lines,
            chunk_index=chunk_index,
            context_lines=context_lines,
            max_matches=max_matches,
            cursor=cursor,
            verbose=verbose,
        )
        return _budgeted_response(response)
    except KnowledgeError as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Prompts — surface as slash commands in MCP clients.
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="probe",
    description=(
        "Search team operational memory for context relevant to your current "
        "task before continuing. Use before architectural decisions, "
        "debugging unfamiliar code, or any 'how do we / why did we' question."
    ),
)
def probe(task: str = "") -> str:
    """Slash command that nudges the agent to search before proceeding.

    Args:
        task: Optional 1-line summary of what the user is working on. If
            omitted, the agent is instructed to summarize its own current
            task and search for that.
    """
    return PROBE_PROMPT_TEMPLATE.format(
        task_block=(
            f"Specifically, search for: {task.strip()}\n\n"
            if task and task.strip()
            else (
                "First summarize my current task in one line, then search for that.\n\n"
            )
        )
    )


@mcp.prompt(
    name="probe-plan",
    description=(
        "Search team operational memory before presenting an implementation, "
        "architecture, or refactor plan, then include Probe context in the plan."
    ),
)
def probe_plan(task: str = "") -> str:
    """Slash command for plan-mode context injection without the watcher.

    Args:
        task: Optional 1-line summary of the plan being formed. If omitted, the
            agent is instructed to summarize its current planned work.
    """
    return PROBE_PLAN_PROMPT_TEMPLATE.format(
        task_block=(
            f"Specifically, search for plan context about: {task.strip()}\n\n"
            if task and task.strip()
            else (
                "First summarize the plan you are about to present in one line, "
                "then search for that.\n\n"
            )
        )
    )

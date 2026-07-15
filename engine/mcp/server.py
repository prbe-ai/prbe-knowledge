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

from services.mcp.clients.knowledge import KnowledgeError, get_client
from services.mcp.consts import (
    ALLOWED_HOSTS,
    ALLOWED_ORIGINS,
    MCP_INSTRUCTIONS,
    MCP_SERVER_NAME,
    PROBE_PLAN_PROMPT_TEMPLATE,
    PROBE_PROMPT_TEMPLATE,
)
from services.mcp.dependencies.auth_context import get_current_customer
from services.mcp.services.response_budget import (
    fit_response_to_budget,
    serialize_tool_response,
)

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
) -> CallToolResult:
    """Call BEFORE making design decisions, debugging unfamiliar systems,
    producing an implementation/architecture/refactor plan, or answering
    "how do we / why did we / what about" questions.

    Searches the user's team operational memory — Slack threads, GitHub PRs,
    Linear tickets, Notion docs, Sentry incidents. The team has probably
    discussed your current task before; this surfaces that history as
    full-fidelity evidence you can quote, cite, or reason over directly.

    Pass a bag of entities/keywords as the query — ticket IDs, repo or
    service names, file/symbol names, error strings, feature flags. NOT a
    question or sentence; prose dilutes BM25, vector, and entity extraction.

      Good: "PRB-17 Linear enrichment per-source toggle workspace_prefs JSONB"
      Bad:  "Why is PRB-17 still considered broken?"

    Surface what you find to the user before proceeding.

    NOT source-code search. For code, read the repo directly.

    Response shape: `results[]`, each Document result carrying doc-level metadata
    (`doc_id`, `source_system`, `source_url`, `title`, `author_id`,
    `created_at`, `updated_at`, `score`, `chunk_count`) and a nested
    `chunks[]` array of the matching spans within that document. Each
    chunk carries its own `score`, `content`, and `graph_evidence`
    (a list of `{edge_type, confidence, via_entity, reason}` entries —
    the trail of knowledge-graph edges that connected the chunk to your
    query; empty list when the chunk matched on text alone). Top-level
    `confidence_breakdown` is an aggregate count of evidence confidences
    (`EXTRACTED` / `INFERRED` / `AMBIGUOUS`) across all returned chunks —
    a low EXTRACTED ratio means most matches are inferred and you should
    treat the result set as weaker.

    The response also includes `related_entities` — non-Document graph
    nodes attached to the returned docs. When you want to crawl laterally
    (BFS the knowledge graph), pick one with the highest `score`
    (IDF-adjusted, demotes generic high-degree entities) and drop its
    `canonical_id` into the next `search_knowledge` query bag. Set
    `top_k_related=0` to skip this enrichment for token-sensitive flows.
    `related_entities=null` with `related_entities_error` set means the
    walk failed — documents are still trustworthy.

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

    Each document carries `author_id` — the raw author identifier from
    the source system (GitHub login, commit-author email, Slack user
    id, Linear user id). It is NOT canonicalized: the same person can
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
            recall dial — if the result you're looking for isn't in the
            top 5, raise it (e.g. 15 or 50) and search again before
            concluding the team hasn't discussed something.
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
            demoted in favor of specific ones. Pick the highest-`score`
            one and feed its `canonical_id` into the next call's
            `query` to BFS the knowledge graph.
        discovery: Default False (focus mode — vector + BM25 only).
            Set True for **discovery mode**: graph hits' contribution
            to fusion is amplified by their surprise score (capped 2x),
            and edges between two hub nodes are demoted via a log-decay
            anti-bonus. The combined effect: entity-anchored canonical
            docs (the actual PR, ticket, design rationale, runbook)
            rise above two common kinds of noise — recent transcripts
            that semantically mention the topic, and wiki/index summary
            docs that connect to everything.

            Default to True for most queries against this corpus.
            Empirically (post-anti-bonus, 6 paired acme
            queries): 5/6 cases see canonical PRs/commits/Notion docs
            move into top-3 that focus mode buried at rank 6+ behind
            transcripts or wiki anchors. The 1 neutral case was already
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
            Set True for the full upstream payload when debugging.
    """
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

    For your own reasoning or before-you-design context-gathering, prefer
    `search_knowledge` — it gives you the same evidence without an LLM
    in the middle.

    Response shape: `answer` (string), `citations` (list referencing the
    underlying documents), `insufficient_context` (bool — true when the
    LLM couldn't find enough grounded evidence and refused to guess),
    `model` (which LLM produced the answer), and the full retrieval
    payload as `search_knowledge` (doc-grouped `results[]` with nested
    `chunks[]`, each chunk carrying `score`, `content`, `graph_evidence`,
    `why_relevant` (the gatherer's per-chunk rationale), and
    chunk-level `matched_via`). When `top_k_related >= 1`, also carries
    top-level `related_entities` + `query_root_doc_id` + `gatherer_notes`.
    When `insufficient_context=true`, surface that refusal to the user
    instead of paraphrasing it.

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
        discovery: Default False (focus mode). Set True to amplify
            graph hits' contribution to fusion (capped 2x via surprise
            score, with hub-to-hub edges demoted via log-decay) so the
            LLM sees entity-anchored canonical docs (the actual PR,
            commit, design doc, ticket, runbook) instead of recent
            transcripts that semantically mention the topic or wiki
            summary docs that connect to everything. Default to True
            for most questions against this corpus. Use for conceptual
            questions ("how should we approach X", "what's blocking
            Y") and any case where focus mode is returning transcript-
            or wiki-shaped evidence when you expected the canonical
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

    Includes `author_id` at the top level (same raw form as on
    `search_knowledge` Document results). The response includes navigation
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

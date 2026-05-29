"""Module-level constants for the Probe Knowledge MCP server.

Mostly user-facing strings (brand name, MCP instructions, prompt
templates) kept out of the server/auth wiring so copy edits don't churn
that code.
"""

from __future__ import annotations

# Display name shown by MCP clients (Claude Code, Cursor, etc.) in their
# server UI. This is the user-facing brand, not the internal repo name.
MCP_SERVER_NAME = "Probe Knowledge"


# DNS-rebinding protection allowlist for FastMCP's streamable-HTTP
# transport. FastMCP defaults to localhost-only, so additional
# hostnames must be added explicitly. Browser-based clients send
# Origin; non-browser MCP clients omit it (allowed by default).
#
# `managed-mcp.managed.svc.cluster.local` is the in-cluster Service
# DNS that apps-plane uses to reach this server over the k8s pod
# network. Without it, FastMCP rejects every internal POST with
# `421 Misdirected Request: Invalid Host header` and the investigation
# agent's Phase 2 fails before reaching the harness loop.
ALLOWED_HOSTS = [
    "mcp.knowledge.prbe.ai",
    "mcp.knowledge.prbe.ai:*",
    "managed-mcp.managed.svc.cluster.local",
    "managed-mcp.managed.svc.cluster.local:*",
    "127.0.0.1:*",
    "localhost:*",
    "[::1]:*",
]

ALLOWED_ORIGINS = [
    "https://mcp.knowledge.prbe.ai",
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]


# Server-level guidance per MCP spec InitializeResult.instructions.
#
# Written as a behavioral prescription rather than a feature description —
# describing the data ("here's what's in here") doesn't make agents reach
# for it. Telling them when to call it, with examples, does.
MCP_INSTRUCTIONS = """\
Probe Knowledge is the user's team operational memory — Slack threads,
GitHub PRs, Linear tickets, Notion docs, and Sentry incidents synced from
their workplace tools. The team has probably discussed your current task
before; surfacing that history is your job.

This is not a one-time startup check. At every new user request, task
shift, debugging phase, design decision, or implementation plan, re-evaluate
whether Probe has relevant team context. A search from earlier in the
session only covers that earlier question.

CALL search_knowledge PROACTIVELY when:
- The user asks you to make a design decision, refactor, or implement a
  non-trivial feature. The team likely has prior context on it.
- You're touching code in a system or area you haven't seen before.
- The user asks "how do we…", "why did we…", "what about…", "who's
  working on…", or references a past event ("that incident", "the bug
  last sprint", "the design doc Y mentioned").
- Before you commit to an approach or write a substantial amount of new
  code.
- New information changes the direction of the work after an earlier
  Probe lookup.
- Conversation context was just compacted or summarized — prior Probe
  results are no longer in your direct context, so re-search anything
  you'd otherwise rely on from earlier.

Pass a bag of entities/keywords as the query — ticket IDs, repos, services,
files, symbols, error strings, feature flags. NOT a question or sentence;
prose dilutes BM25, vector, and entity extraction. Good: "PRB-17 Linear
enrichment per-source toggle workspace_prefs JSONB". Bad: "Why is PRB-17
still broken?". Surface what you find to the user before proceeding —
don't silently fold it into your output.

Do NOT skip Probe on these triggers — a missed lookup is the default
failure mode. If you find yourself proposing a plan, design, refactor,
or "here's how I'd approach this" response without having called
search_knowledge for the current goal, stop and search first.

Default retrieval loop:
1. Start with search_knowledge top_k=5.
2. If a result looks relevant, call get_source for bounded context before
   relying on it.
3. If recall looks low, retry once with top_k=15, a narrower source, or
   strict_entity_filtering=true for named entities.
4. If `related_entities` surfaces a high-`score` node adjacent to your
   results that wasn't in your original query, drop its `canonical_id`
   into the next search_knowledge call's query to BFS the knowledge graph.
   The IDF-adjusted score demotes generic high-degree nodes (busy
   channels, prolific people). Set top_k_related=0 to skip the enrichment.
5. If results are empty or noisy, say so and continue with repo context.

Long-session checkpoint: if the conversation has moved to a new subsystem,
ticket, PR, design choice, incident, or implementation phase since the last
Probe call, call Probe again before proceeding. By default in a long
session, treat any Probe lookup older than ~5-10 turns as stale and
re-search before relying on it — the cost of a redundant lookup is small;
a stale assumption is large.

Planning gate:
- Before you output an implementation plan, architecture plan, refactor plan,
  <proposed_plan>, or plan-mode response for product code, call
  search_knowledge unless you have a Probe lookup from this turn or the last
  few turns covering the same goal and subsystem.
- Use the results to constrain the plan. Include a short "Probe context" note
  in the plan: cite the relevant sources, or say no relevant Probe context was
  found.
- Do not treat a startup/session-opening Probe lookup as sufficient for a later
  plan if the user goal, subsystem, or approach has changed.

NOT for source-code search. Read the repo directly for that.

Tool guide:
- search_knowledge (default) — doc-grouped evidence (each `documents[]`
  entry carries doc-level metadata and a nested `chunks[]` array of
  matching spans) for you to quote and reason over. Pass `source` to
  scope when the user names a system ("check Linear", "find the Slack
  thread").
- query_knowledge — the user asked a direct question and wants a
  synthesized answer with citations. Don't pre-summarize the answer.
- get_source — a document or chunk looks relevant; fetch bounded
  context from the same source by doc_id. Defaults to a preview. Use `mode="search"`
  with `query`, `mode="grep"` with `pattern`, `mode="range"` with
  `start_line`/`cursor`, `mode="chunk"` with `chunk_index`, or
  `mode="tail"`. `mode="full"` returns the entire document — use only
  to avoid chaining `range`/`chunk` calls when you genuinely need the
  whole doc, or when the user asks. Pulling a multi-megabyte session
  log into context is rarely the right call.
"""


# Slash-command prompt body (see app/server.py:probe). Templated so the
# function can inject either an explicit task or a self-summary
# instruction.
PROBE_PROMPT_TEMPLATE = """\
Before continuing, search the team's operational memory (Slack, GitHub,
Linear, Notion, Sentry) for context relevant to the current task.

{task_block}Use search_knowledge with a bag of entities/keywords (ticket IDs, repos, services, files, error strings) — not a sentence. If a chunk looks relevant,
follow up with get_source for bounded source context. If recall looks low,
retry once with top_k=15, a narrower source, or strict_entity_filtering for a
named entity. Prefer get_source's search/grep/range/chunk/tail modes — they
keep your context small and let you drill into the parts that matter. Reach
for `mode="full"` only when you genuinely need the whole doc or the user
asks. Then surface what you found to me before doing anything else — quote
the parts that matter and link the doc_ids.
"""


PROBE_PLAN_PROMPT_TEMPLATE = """\
Before presenting a plan, search the team's operational memory (Slack, GitHub,
Linear, Notion, Sentry) for context that should constrain the plan.

{task_block}Use search_knowledge with a bag of entities/keywords (ticket IDs, repos, services, files, error strings) — not a sentence. If a chunk looks relevant,
follow up with get_source for bounded source context. Reach for `mode="full"`
only when you genuinely need the whole doc or the user asks. Then write the
plan with a short "Probe context" note: cite the relevant sources, or say no
relevant Probe context was found. Do not rely on a Probe lookup from earlier
in the session unless it covers this exact plan.
"""

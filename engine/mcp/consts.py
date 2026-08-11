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
Probe searches team operational history in Slack, GitHub, Linear, Notion, and
Sentry. Use search_knowledge only when a concrete history question could change
the answer or approach: prior rationale, incidents, ownership, constraints, or
similar/parallel work.

Do not call Probe for repo facts, routine implementation/review, status, or
shipping. A new request, plan, phase change, compaction, or elapsed time is not a
trigger. Reuse a relevant lookup for the same decision.

Query with an entity/keyword bag and top_k=5. Use get_source, retry, or follow
related_entities only when needed to resolve the decision. Surface useful
findings, then continue from repo evidence. Use query_knowledge only for a direct
question needing a synthesized, cited answer. Probe is not source-code search.

If Probe informs a plan, cite the useful sources; otherwise omit a Probe note.
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

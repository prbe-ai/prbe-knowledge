# Wiki bootstrap: per-source crawler agents for initial generation

> **Status:** plan draft. Pre-eng-review.
> **Scope:** initial wiki creation only. Realtime updates stay on the v4
> event-replay loop (untouched).

## Context

v4 (shipped tonight) generates the wiki by **replaying queued events** through
a single Pro agent loop. That works for daily incremental updates — the agent
sees yesterday's events, distills, updates pages.

But it has a coverage problem at customer-onboarding time:

- A new customer's wiki starts empty.
- The queue only contains events ingested AFTER they connected sources.
- Historical knowledge sitting in their codebase READMEs, three-year-old
  Slack threads, Linear backlog, Notion runbooks, Granola transcripts
  is never seen by the agent — those events were never enqueued.
- Even a customer who's been ingesting for months has gaps: events that
  triage rejected (score < 7.0) hold real knowledge in aggregate but
  never reach the agent.

This plan addresses ONLY the initial-creation gap. The v4 daily loop is
untouched.

## What "initial creation" means

Three trigger points run the bootstrap:

```
1. Customer wiki created from scratch:
     dashboard "Rebuild wiki" button (admin) -> bootstrap orchestrator
2. New integration connected:
     OAuth callback for source X completes -> single-source crawler
     for X kicks off; other sources unaffected
3. Programmatic re-bootstrap:
     scripts/wiki_bootstrap.py <customer> [--source X] for ops use
```

After bootstrap completes, v4 daily replay continues as today.

## v3 architecture comparison

```
TODAY (v4 daily replay)               BOOTSTRAP (this plan)
────────────────────────              ────────────────────────────────
1 Pro agent per drain.                N agents in parallel, one per source.
Sees only queued events               Each agent fetches everything API-
   for past day.                       accessible from its source.
Reads day in time order.              Each agent decides what's wiki-
                                       worthy as it ingests.
```

## Per-source agent design

Each crawler is its own `AgentLoop` (same harness as v4) with a
source-specialized system prompt + tool palette + model.

```python
class SlackCrawlerAgent(BootstrapAgent):
    source = "slack"
    model = WIKI_BOOTSTRAP_MODEL_SLACK   # Gemini 3.x Pro
    system_prompt = ...  # "you are reading every Slack channel for {cust}"
    tools = [
        # Source API tools (paginating wrappers)
        list_channels(),
        list_messages(channel_id, since, until, cursor),
        get_thread(thread_ts),
        # Shared wiki-write tools (same impl as v4 agent)
        list_wiki_pages(),
        read_page(wiki_type, slug),
        update_page(...),
        create_page(...),
        skip_events(...),     # records "I read these but no page change"
        done(),
    ]
```

Sources covered in v1:
- `slack`     — messages + threads, all channels
- `github`    — repos, PRs, issues, commits, code reviews (NOT file contents — that's `codebase`)
- `linear`    — projects, tickets, comments
- `notion`    — pages, databases, doc trees
- `granola`   — meeting transcripts
- `claude_code` — sessions / memories
- `codebase`  — repo file contents (READMEs, ARCHITECTURE.md, language-specific docs, top-level files)

Each agent ends up calling **the same `update_page` / `create_page` runtime**
that v4 already uses. Wiki output schema unchanged; only the input pipeline
per source is new.

## Coordination between crawlers

Two crawlers may both want to write `runbook/auth-deploy`. **Locked: parallel
from day 1 with optimistic concurrency.**

Each agent's `update_page` call carries the doc version it read. On version
mismatch the writer returns `STALE_VERSION` and the agent re-reads + retries
(merges its delta into the new body). `documents.version` already exists on
the schema, so no migration. Conflicts are rare in practice (different
crawlers tend to write different page types) and the retry cost is bounded
by the agent's per-call budget.

Order of operations is *not* coordinated — every crawler starts at once, hits
its source API in parallel, writes to the wiki as it goes. One slow crawler
no longer blocks fast ones; one rate-limited crawler no longer blocks the
rest. Sequential ordering was rejected in eng review (would push pebble
bootstrap from ~30 min to ~3 hours wall-clock).

## Recency-first ingestion ordering

Default: each crawler walks **newest -> oldest**. Rationale:

- Current state of the team is the most valuable wiki signal. A 2026-Q2
  decision matters more than a 2024-Q3 one for "what does X currently
  mean."
- If bootstrap is interrupted (halt, machine crash, rate-limit
  exhaustion), recency-first means the most useful knowledge already
  landed. Older content fills in *why* — nice to have, not load-bearing.
- The agent's auto-compact strategy works better when current data is
  read first: newer events tend to reference older ones, so when the
  agent later reads older events it has a frame for relevance and can
  decide quickly whether to fold in or skip.

Pagination per source flips to descending order:

| Source | Recency-first pagination |
|---|---|
| Slack | `latest=now` cursor walking backward via `oldest` ts |
| GitHub PRs/issues | `sort=updated&direction=desc` |
| GitHub commits | `since=null, until=now` walking backward |
| Linear | `orderBy: { updatedAt: Desc }` |
| Notion | `sort: { property: last_edited_time, direction: descending }` |
| Granola | `started_at desc` |
| Claude Code | `last_used desc` (per session) |
| Codebase | HEAD only — recency notion doesn't apply to file contents in v1 |

The agent's system prompt reinforces this:

> "You are reading {source} for {customer}, newest first. Build the
> wiki from current-state knowledge first; only walk further back when
> earlier signals are still adding new pages or shifting existing
> ones. If 50 consecutive items don't change the wiki, call done() —
> the rest is unlikely to."

That last sentence is a soft termination heuristic: stop crawling when
incremental signal dries up.

## Auto-compact for "technically infinite" ingestion

v4 already compacts at 60% context. For a bootstrap crawler that wants to
ingest unbounded data, two distinct things compact:

1. **Conversational history** (existing v4 compaction)
2. **Source-data ingestion summary** (NEW)

```
loop:
    api_chunk = fetch_next_page(cursor)
    agent processes chunk, possibly writes pages
    if context_size > 60% of model window:
        compact()      # summarize prior turns AND
                       # generate "ingestion summary": which channels
                       # / repos / pages have I covered so far + key
                       # signals seen
    if cursor exhausted:
        done()
```

The ingestion summary is what makes "infinite ingestion" work in practice.
On next compaction, the agent picks up where it left off without re-reading.
Compaction loses fidelity but never loses coverage.

## GBrain v0.25 schema adoptions

After reading GBrain's actual `src/schema.sql` (17,888 pages in production
on the same primitives), adopting more than just the entity-types idea.

### Schema changes (one alembic migration covers all)

1. **Split page body into `compiled_truth` + `timeline`.**
   `documents.body` for `doc_class IN ('compiled_wiki','agent_artifact')`
   becomes two fields stored in `documents.metadata` JSONB (no schema
   migration on `documents` itself):
   ```
   metadata = {
     "compiled_truth": "...",   # distilled knowledge — what the team currently knows
     "timeline": "..."          # chronological audit — events that contributed
   }
   ```
   Read endpoint surfaces both. Dashboard renders compiled_truth as the
   page body and timeline as a collapsible "audit" section below.

2. **`wiki_timeline_entries` (NEW table).**
   ```sql
   CREATE TABLE wiki_timeline_entries (
     id              BIGSERIAL PRIMARY KEY,
     customer_id     TEXT NOT NULL,
     wiki_type       TEXT NOT NULL,
     slug            TEXT NOT NULL,
     entry_date      DATE NOT NULL,
     source          TEXT NOT NULL,        -- "slack", "github", "linear", etc.
     summary         TEXT NOT NULL,
     detail          TEXT NOT NULL DEFAULT '',
     created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
     CONSTRAINT uq_wiki_timeline_dedup
       UNIQUE (customer_id, wiki_type, slug, entry_date, summary)
   );
   CREATE INDEX ix_wiki_timeline_page ON wiki_timeline_entries (customer_id, wiki_type, slug, entry_date DESC);
   CREATE INDEX ix_wiki_timeline_date ON wiki_timeline_entries (customer_id, entry_date DESC);
   ```
   Crawlers append entries as they ingest. Dedup is automatic via the
   unique constraint — re-bootstrap doesn't double-fill.

3. **`wiki_links` (already locked).** Adopting GBrain's column shape:
   ```sql
   CREATE TABLE wiki_links (
     id                SERIAL PRIMARY KEY,
     customer_id       TEXT NOT NULL,
     src_wiki_type     TEXT NOT NULL,
     src_slug          TEXT NOT NULL,
     dst_wiki_type     TEXT NOT NULL,
     dst_slug          TEXT NOT NULL,
     link_type         TEXT NOT NULL DEFAULT '',     -- "works_at", "decided", ""
     context           TEXT NOT NULL DEFAULT '',     -- ~80 chars surrounding
     link_source       TEXT CHECK (link_source IN ('markdown','frontmatter','manual')),
     created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
     CONSTRAINT uq_wiki_links UNIQUE NULLS NOT DISTINCT
       (customer_id, src_wiki_type, src_slug, dst_wiki_type, dst_slug, link_type, link_source)
   );
   CREATE INDEX ix_wiki_links_from ON wiki_links (customer_id, src_wiki_type, src_slug);
   CREATE INDEX ix_wiki_links_to ON wiki_links (customer_id, dst_wiki_type, dst_slug);
   ```

4. **TSVECTOR search_vector on wiki pages.**
   Add `documents.search_vector TSVECTOR` (only populated for wiki-class
   docs). Trigger maintains it from title + compiled_truth + timeline +
   linked timeline_entries. Hybrid retrieval queries pgvector + tsvector
   together (Postgres FTS rank + cosine similarity, RRF-merged).

5. **`wiki_raw_data` sidecar table.**
   ```sql
   CREATE TABLE wiki_raw_data (
     id            BIGSERIAL PRIMARY KEY,
     customer_id   TEXT NOT NULL,
     wiki_type     TEXT NOT NULL,
     slug          TEXT NOT NULL,
     source        TEXT NOT NULL,           -- "slack" / "github" / "linear" / etc
     source_ref    TEXT NOT NULL,           -- thread_ts, PR id, etc
     data          JSONB NOT NULL,          -- the raw API response
     fetched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
     CONSTRAINT uq_wiki_raw_data UNIQUE (customer_id, wiki_type, slug, source, source_ref)
   );
   ```
   For each (page, source) pairing, store the original API response.
   Crucial for debugging "why does the wiki say X?" — you can trace back
   to the exact Slack thread or GitHub PR.

6. **`content_hash` on wiki pages.**
   Already present on `documents` via metadata is awkward. Add explicit
   `wiki_content_hash` column populated by trigger from
   compiled_truth + timeline. Re-bootstrap skips writes when hash matches.

7. **`page_versions`.** Skipped for v1 — `documents.version` already
   gives us monotonic versioning per slug, and `wiki_synthesis_runs`
   gives audit. Adopt the snapshot table only if rollback UX gets built.

### Zero-LLM-call link extraction (GBrain's approach)

Critical insight: GBrain extracts entity references and builds the link
graph with **zero LLM calls per write**. Pure parser:

```python
# In wiki_agent.tool_update_page() / tool_create_page() (or a writer
# helper that wraps both):

def extract_links(body_markdown: str, frontmatter: dict) -> list[Link]:
    links = []
    # 1. Markdown body: \[\[([^\]|]+)(?:\|([^\]]+))?(?:\|([^\]]+))?\]\]
    #    Match groups: target_ref, optional relation, optional display.
    #    target_ref = "type:slug" e.g. "person:maison" or "decision:auth-rollback"
    for m in MARKDOWN_LINK_RE.finditer(body_markdown):
        target_ref, relation, _display = m.groups()
        wiki_type, slug = target_ref.split(":", 1)
        links.append(Link(
            dst_wiki_type=wiki_type, dst_slug=slug,
            link_type=relation or "",
            link_source="markdown",
            context=body_markdown[max(0, m.start()-40):m.end()+40],
        ))
    # 2. Frontmatter: any value that's a "type:slug" string OR list of
    #    them. Field name becomes link_type. Example:
    #      works_at: company:probe
    #      owns: [service_card:auth, service_card:wiki]
    for field, value in frontmatter.items():
        for slug_ref in _coerce_to_slug_list(value):
            wiki_type, slug = slug_ref.split(":", 1)
            links.append(Link(
                dst_wiki_type=wiki_type, dst_slug=slug,
                link_type=field,
                link_source="frontmatter",
                context="",
            ))
    return links

# Wrap update_page / create_page commit with:
#   1. Persist page (compiled_truth, timeline, frontmatter, ...)
#   2. extracted = extract_links(body, frontmatter)
#   3. DELETE FROM wiki_links WHERE customer_id=... AND src=...
#   4. INSERT extracted links (ON CONFLICT DO NOTHING)
#   5. Update content_hash, trigger search_vector refresh
#
# The atomic boundary is at the page write. Link extraction +
# persistence runs as a separate transaction immediately after the
# page commits, with transient/IO exceptions swallowed (logged as
# warnings). The page is the source of truth; if link writes fail
# transiently, the link graph goes stale but the page is intact.
# Reconciliation: a future cron job (TODO, deferred) can detect
# pages whose body contains [[type:slug]] markers absent from
# wiki_links and re-run extraction.
```

Crawlers don't need to call the link-extraction LLM — they just produce
markdown with `[[type:slug|relation]]` syntax (or YAML frontmatter), and
the writer handles the rest.

## GBrain-inspired upgrades: entity pages + typed links + backlinks

Borrowing from GBrain's "self-wiring knowledge graph" pattern. Today's
v4 produces flat **topical** pages (service_card/decision/feature/runbook).
Bootstrap should produce a richer mix: **entity** pages, **event**
pages, and **typed links** between them.

### Schema extension

`wiki_type` extends from 4 -> 10:

| wiki_type | Example | Source |
|---|---|---|
| service_card | `service_card/auth` | (existing) |
| decision | `decision/vector-store-pgvector` | (existing) |
| feature | `feature/wiki-synthesis` | (existing) |
| runbook | `runbook/auth-deploy` | (existing) |
| person | `person/maison` | NEW: extracted from any source mentioning a person |
| company | `company/probe` | NEW: customer/vendor/competitor |
| vendor | `vendor/anthropic` | NEW: tool/service supplier |
| customer | `customer/willow-voice` | NEW: a customer of the customer |
| project | `project/q3-roadmap` | NEW: cross-cutting initiative |
| event | `event/2026-05-05-1on1` | NEW: meeting / incident / launch |

No DB migration required if we keep `wiki_type TEXT` (already is).
Schema update is just the `Literal[...]` in
`services/synthesis/models.py:WikiType`.

### Typed links

Page bodies use `[[type:slug]]` and `[[type:slug|display]]` syntax.
Optional relationship label: `[[type:slug|verb|display]]`.

Examples in markdown:
```markdown
Maison [[person:maison|works_at]] [[company:probe|Probe]]. She drove
[[decision:auth-rollback-policy|the auth rollback policy]] in
[[event:2026-05-05-1on1|today's 1:1]].
```

At every `update_page` / `create_page` call, parse the body for
`\[\[(\w+):([\w-]+)(?:\|([^\]]+))?\]\]` and persist to the `wiki_links`
table (schema defined above in "GBrain v0.25 schema adoptions" #3).

### Backlinks

Read endpoint extends:
```
GET /knowledge/wiki/pages/{type}/{slug}
  Response: { ...existing fields..., backlinks: [...] }
```

`backlinks` is the set of pages that link TO this one, grouped by
relation. Dashboard renders inline as a sidebar:

```
person/maison
─────────────────────────
[main page body]

Mentioned by (backlinks):
  Decisions Maison drove (works_at):
    - decision/auth-rollback-policy
  Services owned:
    - service_card/auth
  Events:
    - event/2026-05-05-1on1
```

### Crawler prompt deltas

Each source's system prompt explicitly tells the agent:

> "When you read a {source} item, extract: (1) the people involved
> (-> person pages), (2) the companies referenced (-> company pages),
> (3) the meeting/incident/launch context (-> event page), (4) the
> durable decision or runbook step (-> decision/runbook pages).
> Link them together with `[[type:slug|relation]]` syntax. The wiki
> graph is more valuable than any single page."

### Hybrid search

GBrain pairs FTS5 + vector. Postgres has both:
- `pgvector` for semantic — already in our retrieval stack
- `tsvector + GIN` for keyword — likely partial coverage; should
  audit `services/retrieval/retrievers/{bm25,vector,sql}.py` to
  confirm hybrid ranking is engaged on wiki pages.

Out of bootstrap scope; flagged in TODOs.

## Coexistence with v4 daily loop

**Locked: option B — `synthesis_skipped` with reason `bootstrap`.**

Bootstrap, when it finishes reading a Slack thread / GitHub PR / etc.,
marks any matching `wiki_synthesis_queue` rows as `synthesis_skipped`
with `synthesis_error = 'bootstrap_absorbed'`. Daily replay's existing
terminal-status filter then ignores them. No schema migration needed;
audit trail lives in the `synthesis_error` field.

Matching is by `(customer_id, doc_id)` — bootstrap fetches data via
source APIs, but every fetched item still corresponds to a `documents`
row (or will once the regular ingestion pipeline catches up). Bootstrap
records the doc_ids it absorbed; daily-loop reconciliation marks queue
rows accordingly post-hoc.

## Per-source models

User instinct: "model per integration source." Practical interpretation:

- v1: same Gemini 3.1 Pro for all sources, different SYSTEM PROMPTS per
  source (specialized: "you are a Slack analyst" vs "you are a code
  architect"). Same harness, same auto-compact, easy to test.
- v2 (deferred): swap models per source if specific ones hit ceiling.
  Codebase crawler may benefit from a code-specialized model; chat
  crawler may benefit from longer context.

## Cost (acknowledged, ignored per user)

Rough estimates per customer one-time bootstrap:

| Customer size | Slack | GitHub | Linear | Notion | Codebase | Total |
|---|---:|---:|---:|---:|---:|---:|
| Small (mahits-workspace) | $1 | $2 | $0.50 | $0.50 | $1 | ~$5 |
| Medium (probe-founders) | $5 | $10 | $2 | $2 | $5 | ~$25 |
| Large (willow-voice) | $20 | $40 | $10 | $10 | $20 | ~$100 |
| Pebble | $100 | $300 | $50 | $50 | $100 | ~$600 |

One-time cost. Ongoing is the v4 daily loop (~$30/drain pebble).

## Files to add (prbe-knowledge)

| File | Role |
|---|---|
| `services/synthesis/bootstrap_app.py` | Entry point for the new fly app. Listens for bootstrap-trigger events; orchestrates per-source agents. |
| `services/synthesis/bootstrap_orchestrator.py` | Sequential crawler runner. Picks crawler order, opens runs, handles per-source failure isolation. |
| `services/synthesis/crawlers/__init__.py` | Crawler registry. |
| `services/synthesis/crawlers/base.py` | `BootstrapAgent` ABC with shared logic (compaction, halt, ingestion summary). |
| `services/synthesis/crawlers/slack_agent.py` | SlackCrawlerAgent + tools (list_channels, list_messages, get_thread). |
| `services/synthesis/crawlers/github_agent.py` | GitHubCrawlerAgent (PRs, issues, commits, reviews — NOT file contents). |
| `services/synthesis/crawlers/linear_agent.py` | LinearCrawlerAgent (projects, tickets, comments). |
| `services/synthesis/crawlers/notion_agent.py` | NotionCrawlerAgent (pages, databases, trees). |
| `services/synthesis/crawlers/granola_agent.py` | GranolaCrawlerAgent (meeting transcripts). |
| `services/synthesis/crawlers/claude_code_agent.py` | ClaudeCodeCrawlerAgent (sessions, memories). |
| `services/synthesis/crawlers/codebase_agent.py` | CodebaseCrawlerAgent (repo file contents — READMEs, ARCHITECTURE.md, top-level docs). |
| `services/synthesis/api_clients/{slack,github,linear,notion,granola,claude_code}.py` | Paginating async wrappers. Some sources already have ingestion-side clients we can refactor to share. |
| `services/synthesis/bootstrap_persistence.py` | Run-tracking, ingestion summary persistence, bootstrap_absorbed marking. |
| `services/synthesis/prompts.py` | Add 7 source-specific system prompts. |
| `fly.wiki-bootstrap.toml` | New fly app config. 4-8 GB machines (high context, many in-flight API connections). |
| `services/ingestion/wiki_routes.py` | New endpoint: `POST /api/wiki/bootstrap/trigger` (internal-key, per customer + optional source filter). |
| `app/routers/dashboard/knowledge.py` (prbe-backend) | New BFF route: `POST /knowledge/wiki/bootstrap/trigger` (admin gate). |
| `src/lib/api/wiki.ts` (prbe-dashboard) | `triggerWikiBootstrap()` + status surface. |
| `src/app/(dashboard)/(main)/wiki/page.tsx` | "Rebuild wiki from scratch" button (admin). |
| Tests | New module per crawler + integration tests for orchestrator. |

## Test plan (sketch — full diagram in eng review)

Per crawler:
- Unit: paginates correctly, handles rate limits, calls update_page/create_page on right inputs.
- Integration: with mocked source API, agent makes expected wiki calls.
- Eval (LLM): given a hand-curated source corpus, agent produces wiki pages within edit-distance ε of expected.

Orchestrator:
- Sequential crawler order applied.
- Per-source failure isolation (Slack rate-limited -> GitHub continues).
- Re-bootstrap is idempotent (bootstrap_absorbed marker prevents re-reading).

Auto-compact:
- Compaction triggers at 60%.
- Ingestion summary preserves coverage across compactions.
- Final wiki includes pages from data ingested both before and after compaction.

## NOT in scope (this plan)

- Realtime/streaming source updates. v4 daily loop handles new events.
- Cross-tenant analysis. Each customer's bootstrap is fully isolated.
- Schema migration for new statuses. Reusing `synthesis_skipped` per
  Coexistence section above.
- Source-specific model swaps. Same Gemini 3.x Pro for all v1 crawlers.
- Phase 2 parallel crawlers. Sequential first; parallel later if latency
  bites.
- Customer-specific time-horizon UI. Hardcoded per-source defaults
  (e.g., Slack 12mo, Linear all-time, Notion all-time, Codebase head only).
- A "bootstrap progress" UI in real-time. Status endpoint returns
  current state; no streaming progress bar.
- Cost gating / per-customer budget caps. User said ignore cost.

## What already exists / reuse

| Component | Source | Reused as |
|---|---|---|
| AgentLoop harness | services/synthesis/agent_harness.py (v4) | Each crawler instantiates one. Same auto-compact, halt, snapshot-then-mutate dispatch. |
| WikiAgentRuntime tools | services/synthesis/wiki_agent.py (v4) | `update_page`, `create_page`, `read_page`, `list_wiki_pages` — shared verbatim. Crawlers add source-API tools. |
| GeminiAgentClient | services/synthesis/gemini_agent_client.py (v4) | All crawlers use same Gemini wrapper. Per-source model switch is a config knob. |
| Slack ingestion client | services/ingestion/connectors/slack/* | Refactor paginate logic into `services/synthesis/api_clients/slack.py` (read-only); ingestion keeps using its enrichment paths. |
| GitHub client | services/ingestion/connectors/github/* | Same pattern. |
| Linear client | services/ingestion/connectors/linear/* | Same. |
| Notion client | services/ingestion/connectors/notion/* | Same. |
| Granola client | services/ingestion/connectors/granola/* | Same. |
| Claude Code client | services/ingestion/connectors/claude_code/* | Same. |
| Codebase client | NEW (no ingestion-side equivalent) | Build fresh on top of GitHub client (filesystem read of repo contents). |
| Compaction primitive | services/synthesis/agent_harness.py:_maybe_compact | Reused. Adds the "ingestion summary" data alongside conversational compaction. |
| wiki_synthesis_runs table | shared, used by v4 | New `kind = "bootstrap"` value; new `source` column (or use existing `error` field if no migration). |

## Decisions locked (eng review pass 1)

| # | Question | Decision |
|---|---|---|
| 1 | Time horizon per source | Hard time-window per source: Slack 12mo, Linear all-time, Notion all-time, Granola 24mo, GitHub 12mo PRs/issues + all-time commits, Claude Code all-time, Codebase HEAD only. |
| 2 | Sequential vs parallel crawlers | Parallel from day 1, optimistic concurrency on doc version. |
| 3 | Coexistence with daily loop | Reuse `synthesis_skipped` (option B) with `synthesis_error='bootstrap_absorbed'`. |
| 4 | Per-source model swap | Same Gemini 3.x Pro for all crawlers in v1; per-source config knob already in place for v2 swap. |
| 5 | Bootstrap run tracking | Extend `wiki_synthesis_runs.kind` with new value `bootstrap`; add nullable `source` column for per-crawler runs. |
| 6 | Trigger surface | All 3: dashboard "Rebuild wiki" admin button, OAuth-callback per-source hook, `scripts/wiki_bootstrap.py` for ops. |
| 7 | MVP path | Ship GitHub crawler first (richest structured intent + trickiest rate-limit case), then layer in remaining 6. |
| 8 | Codebase scope | Tier 2: docs + dir tree + file headers + public API surface. **TODO:** integrate or replace with codegraph (does the same job at higher fidelity). |
| 9 | Re-bootstrap behavior | Wipe wiki for the customer first, then bootstrap clean. `content_hash` + `wiki_raw_data.uq_wiki_raw_data` give idempotency on re-runs of the same source. |
| 10 | Entity model | Adopt all 6 GBrain entity types (person/company/vendor/customer/project/event) plus typed links + backlinks. |

## TODOs (deferred)

- **codegraph integration.** The `codebase_agent` ships in v1 with Tier 2 file scanning (READMEs, ARCHITECTURE.md, top-level docs, public API headers). codegraph already extracts call-graph + symbol-graph from a repo; either integrate it as the codebase crawler's tool palette, or delete `codebase_agent` and let codegraph emit wiki pages directly. Decision deferred until codegraph's stability + output format stabilizes.
- **Hybrid retrieval audit.** Confirm `services/retrieval/retrievers/{bm25,vector,sql}.py` engages the new `documents.search_vector` for wiki-class docs alongside pgvector. RRF-merge ranking already exists; just needs to include the new vector.
- **Per-source model swap.** v1 uses Gemini 3.x Pro everywhere. If codebase crawler benefits from a code-specialized model (e.g., longer context, code-aware tokenization), introduce per-source `WIKI_BOOTSTRAP_MODEL_<SOURCE>` env vars.
- **Durable subagent state tables** (`subagent_messages`, `subagent_tool_executions` from GBrain). v1 keeps state in-memory inside the agent harness. Adopt durable tables once v1 ships and we've seen the failure modes.
- **`page_versions` snapshot table.** Skipped for v1 — `documents.version` is enough. Add when rollback UX gets built.
- **Stale-bootstrap alert.** A bootstrap job that's been `running` for > 6 hours probably wedged. Add a daily metric query + alert.
- **Per-customer time-horizon UI.** v1 hardcodes the per-source defaults above. Add a customer-settings UI once we see customers actually want this dial.
- **Link-graph reconciliation cron.** Detect pages whose body / frontmatter contains `[[type:slug]]` markers not reflected in `wiki_links` and re-run extraction. Mitigates the staleness window from the two-transaction page-then-links design (Normalizer._persist owns its own connection, so link writes can't share the page's tx; transient link-write failures are logged-and-swallowed today).

## Failure modes

| Mode | Detection | Mitigation |
|---|---|---|
| Source rate-limit exhaustion mid-crawl | 429 from API client | Adaptive token bucket per source @ 70% of published quota; exponential backoff + jitter on 429; agent's `next_page` tool returns `RATE_LIMITED` and the agent calls `done()` with partial coverage rather than retrying forever |
| Two crawlers race-write the same page | Doc version mismatch in `update_page` | Optimistic concurrency: writer returns `STALE_VERSION`, agent re-reads + re-writes its delta |
| Crawler agent halts (turn cap, stall, error) | Agent harness halt detection (existing) | Per-source failure isolation: orchestrator marks that crawler's run `failed` with reason; other crawlers continue |
| Auto-compact loses key context | Compaction count metric in `AgentRunResult` | Ingestion summary preserves coverage even if conversational fidelity drops; agents pick up where they left off via cursor state stored in run row |
| Re-bootstrap doubles up content | Double-write to wiki page | `content_hash` skip-on-no-change + `wiki_raw_data` unique constraint + wipe-first policy |
| Source API returns malformed data | Pydantic validation in api_client wrapper | Agent sees `MALFORMED_RESPONSE`, logs to ingestion summary, advances cursor |
| Bootstrap orchestrator crashes mid-run | `wiki_synthesis_runs.kind='bootstrap'` row stays in `running` state | Reclaim loop (mirror existing pattern) flips runs older than 1 hour `running` -> `failed`; admin can re-trigger |
| Customer disconnects source mid-bootstrap | OAuth token revoked, API returns 401 | Crawler agent halts cleanly with `AUTH_REVOKED`; partial data already committed stays in wiki; re-bootstrap retries from scratch |

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | DONE | Pass 1: 7 architecture findings, all locked. Pass 2 (this commit): 5 consistency issues fixed (sequential/parallel contradiction, dup wiki_links table, stale open-questions section, missing codegraph TODO, missing failure-modes table). |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | n/a (one button) | — |
| Adversarial | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Outside Voice | (codex/claude) | Cross-model challenge | 0 | — | — |

- **UNRESOLVED:** 0
- **VERDICT:** ENG REVIEW DONE — plan is internally consistent and ready to implement. Recommend GitHub-crawler MVP first (per Decision #7) before layering in the other 6 sources.

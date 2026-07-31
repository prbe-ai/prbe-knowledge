# Changelog

All notable changes to the Probe knowledge engine are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `per_source_top_k` no longer loses a whole corpus to the final `LIMIT`. The
  window function gave each `source_system` its own slot budget and the query
  then applied `ORDER BY score DESC LIMIT top_k`, handing every slot back to
  whichever source scores highest in the absolute — undoing the guarantee one
  line after computing it. Measured on the research cluster with
  `per_source_top_k=20, top_k=30`: the response contained
  `{github: 4, claude_code: 20, code_graph: 6}` and **no `custom_ingest` at
  all**, which first appeared at rank 61. Scores are not comparable across
  sources — a terse structured projection never out-scores a chatty transcript
  on a natural-language query — so the final ordering now leads with the
  per-source rank, interleaving sources instead of letting one sweep the limit.
  Callers that do not pass `per_source_top_k` keep a straight score ranking.

### Added

- BM25 matches document TITLES, not only chunk content (migration 0099). The
  title was unreachable by keyword search: `chunks.content_tsv` is a GENERATED
  column, and a generation expression may only reference its own row, so it can
  never reach `documents.title` — the two are in different tables. A file named
  `model.ckpt`, or a PR titled "Fix the retry loop", was findable by keyword
  only if those words also appeared in the body. Titles are now indexed in a
  `documents.title_tsv` generated column, weighted `A` against unweighted
  content so a title hit outranks a body hit under `ts_rank_cd`.

  The title is indexed twice over — verbatim, and with path punctuation
  flattened to spaces. Postgres' `english` parser emits `model.ckpt` as a
  single `file` lexeme while the retriever splits queries on alphanumeric runs,
  so a verbatim-only index would have missed exactly the case this exists for.

  Title-only matches return one representative chunk (`chunk_index = 0`). A
  title belongs to the document rather than any chunk, so matching without that
  restriction would return every chunk of the document at an identical score
  and a long document would bury everything else. Chunks matching on their own
  content are unaffected.

- Per-turn `finish_reason` and `usage.completion_tokens` are recorded on the
  gatherer loop state, logged on `agent.turn_complete`, carried in the trace
  blob (schema v2), and summarised in the trace digest as
  `completion_tokens_max` / `was_truncated`. The gatherer emits each chunk's
  content verbatim, so emit size scales with results and cannot be inferred
  from a synthetic probe — this is the measurement needed before capping
  `max_tokens`. `finish_reason` is also the truncation alarm whose absence
  from the extractor's failure logs once kept a blown token cap invisible.
- `LLM_TPM_BUDGET` / `LLM_TPM_MAX_WAIT_SECONDS`: an optional per-process
  token-rate limiter in `engine/shared/llm.py`, covering every provider call
  (`import litellm` appears in exactly one engine file). Limits tokens rather
  than requests, because the quota is denominated in tokens and this engine's
  requests differ by ~50x. **Disabled by default** (`0`), and it **fails
  open** — if budget is not available within the max wait it proceeds and
  lets the provider decide, because blocking longer than the caller's own
  deadline turns a fast 429 into a slow timeout that returns nothing.
- Internal ingestion stats can now break Claude Code and Codex totals down by
  device, including historical derived documents linked through their parent
  session, so downstream dashboards can show trustworthy per-laptop counts.
- `POST /api/github/connect` (X-Internal-Knowledge-Key gated): seeds a GitHub
  App installation (customer_source_mapping + `installation:<id>` token row,
  validated by a dry-run mint) and enqueues its historical backfill, which the
  deployed BackfillWorker drains. This is the invokable equivalent of the
  `scripts.github_seed_token` + `scripts.backfill` runbook, so a downstream
  consumer (research-os) can backfill a repo the moment a team claims the
  installation instead of running the manual runbook. The seed logic is
  extracted to `kb.github_seed.seed_github_installation`, shared by the
  endpoint and the CLI.

### Changed

- The turn-1 pre-fan-out dump is rendered as compact JSON instead of
  `indent=2`. Indentation is input tokens carrying no information, re-sent on
  every turn, and it was un-budgeted: the budget counts `content` only, so
  whitespace rode on top of the cap rather than inside it.
- `SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET` lowered 40,000 → 18,000 and made
  env-overridable. This is the highest-leverage knob on provider quota
  because the turn-1 dump is charged on every turn, while a completion is
  charged once. Cerebras enforces 250,000 tokens-per-minute at the
  **organization** level — shared by every deployment of this engine — and
  reserves `input + max_completion_tokens` before running a request, which
  admitted only ~2-3 concurrent searches before `429 token_quota_exceeded`.
  This is a payload cap, not a recall cap: `SEARCH_AGENT_VECTOR_TOP_K` and
  `_BM25_TOP_K` are deliberately unchanged, so the candidate pool and the
  recall ceiling are untouched and only the post-fusion payload shrinks.
  Measured in production, ~280 candidates were rendered to return 10-16
  results.
- Query synthesis now advertises Gemini 3.6 Flash and Gemini 3.5 Flash Lite,
  keeps the previous picker IDs as compatibility aliases, and uses each new
  model's supported thinking level without retired sampling controls.

### Fixed

- Entity extraction no longer silently returns nothing on reasoning models.
  `max_tokens` bounds reasoning *and* visible content together, so the old
  600-token cap was consumed by the reasoning trace and truncated the JSON
  mid-object; the caller swallowed the parse error and searched with zero
  entity anchors while still reporting `state:"ok"`. Raised to 2000 and made
  it env-overridable, since the right value follows the model.
- Extraction failure logs now carry `finish_reason` and `content_len`, which
  distinguish "the model emitted bad JSON" from "we cut the model off".
- Retrieval trace ids no longer collide. They were a bare millisecond
  timestamp, so the several concurrent `/retrieve` calls a single search fans
  out into shared one id — making interleaved log lines unattributable and
  letting their R2 trace blobs overwrite each other at the same key.
- Claude Code session extraction now sends gateway-configured model aliases
  over the proxy's OpenAI-compatible transport while preserving direct
  Anthropic routing, preventing gateway URLs from becoming
  `/v1/v1/messages` and leaving finalized sessions in the ingestion DLQ.
- Retrieval queries now return citable pre-fan-out evidence with low
  confidence when the gateway exhausts its providers on a transient timeout,
  rate limit, or server error. Permanent provider failures and responses
  without citable evidence continue to fail closed, while phase-specific
  deadlines keep the fallback inside the MCP request envelope.
- Gatherer responses now honor `top_k_related`, including returning no related
  entity payload when callers set it to zero.
- Gateway-routed retrieval entity extraction and gatherer turns now make a
  single client attempt, so a failed provider chain is not replayed after the
  LiteLLM gateway has exhausted its configured failover routes. Direct
  self-hosted provider calls retain their normal transient retries.
- Probe MCP retrieval calls now preserve upstream and transport diagnostics,
  use phase-specific HTTP timeouts with enough read headroom for the search
  agent, and return failures with the MCP error flag plus a trace ID.
- Large MCP retrieval responses now emit one compact JSON payload, enforce a
  24 KB hard wire limit, and reject oversized `get_source(mode="full")` calls
  with bounded-mode guidance instead of allowing responses up to the retrieval
  service's 100 MB ceiling.

## [0.1.0] - 2026-06-18

Initial public release of the open-source community edition.

### Added

- Self-hosted, single-tenant knowledge engine: ingestion, worker, retrieval,
  synthesis, and MCP services.
- Source connectors for GitHub, Slack, Linear, Notion, and Sentry, plus a
  custom-ingest API.
- Hybrid retrieval (vector + BM25 + graph) fused via RRF, exposed as raw chunk
  retrieval (`/retrieve`) and LLM-synthesized cited answers (`/query`).
- Knowledge-page synthesis (optional `cron` profile).
- MCP server exposing `search_knowledge`, `query_knowledge`, and `get_source`
  (static and OAuth auth modes).
- Turnkey self-hosting via Docker Compose and a community Helm chart, backed by
  Postgres (pgvector) and S3/R2/MinIO object storage.

[Unreleased]: https://github.com/prbe-ai/prbe-knowledge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/prbe-ai/prbe-knowledge/releases/tag/v0.1.0

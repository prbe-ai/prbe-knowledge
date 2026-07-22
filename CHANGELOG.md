# Changelog

All notable changes to the Probe knowledge engine are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- Query synthesis now advertises Gemini 3.6 Flash and Gemini 3.5 Flash Lite,
  keeps the previous picker IDs as compatibility aliases, and uses each new
  model's supported thinking level without retired sampling controls.

### Fixed

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

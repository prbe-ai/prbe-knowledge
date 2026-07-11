# Changelog

All notable changes to the Probe knowledge engine are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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

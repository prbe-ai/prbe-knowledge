# Probe knowledge engine

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](pyproject.toml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Probe is a self-hosted knowledge engine. It ingests your team's activity from
GitHub, Slack, Linear, Notion, Sentry, and custom sources; normalizes, chunks,
and embeds it into Postgres; and serves hybrid retrieval (vector + keyword +
graph) plus LLM-synthesized, cited answers and auto-generated knowledge pages.
This repository is the open-source community edition: a single-tenant engine you
run yourself, with no control plane and no calls out to Probe's hosted service.

## 30-second quickstart

```bash
cp .env.example .env
# In .env, fill three things:
#   GOOGLE_API_KEY        — embeddings (Gemini)
#   ANTHROPIC_API_KEY     — or OPENAI_API_KEY, for answer synthesis
#   KNOWLEDGE_API_TOKEN   — your query bearer token (openssl rand -hex 32)

make up        # build images + start the full stack (or: docker compose up)
make health    # confirm ingestion + retrieval are live
```

Then ask a question:

```bash
curl -X POST http://localhost:8081/query \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "what changed last week?"}'
```

`make up` brings up Postgres (pgvector), MinIO (object store), a one-shot DB
migration, then the ingestion, retrieval, and worker services. In single-tenant
community mode everything is scoped to `DEFAULT_CUSTOMER_ID` (defaults to
`default`) and gated by `KNOWLEDGE_API_TOKEN`. See
[docs/self-hosting.md](docs/self-hosting.md) for the full setup.

> Apple Silicon note: the Gemini embedding SDK's native `grpcio` wheel can crash
> the ingestion/worker containers with `SIGILL` (exit 132) on arm64. Run on
> `linux/amd64` — `export DOCKER_DEFAULT_PLATFORM=linux/amd64` before `make up`,
> or deploy on an amd64 host. amd64 is unaffected. See
> [docs/self-hosting.md](docs/self-hosting.md#prerequisites).

## How it works

```
                    ┌──────────────────────────────────────────────┐
   data sources     │           Probe knowledge engine             │
 ─────────────────  │                                              │
  GitHub  Slack     │   ┌─────────────┐         ┌──────────────┐   │
  Linear  Notion ──▶│   │  ingestion  │  enqueue│    worker     │   │
  Sentry  custom    │   │  :8080      │────────▶│ normalize ·   │   │
                    │   │  webhooks + │         │ chunk · embed │   │
                    │   │  custom API │         └──────┬───────┘   │
                    │   └──────┬──────┘                │           │
                    │          │ raw payload           │ docs +    │
                    │          ▼                        ▼ chunks +  │
                    │   ┌──────────────┐        ┌──────────────┐   │
                    │   │ object store │        │  Postgres 16  │   │
                    │   │ (S3/R2/MinIO)│        │  + pgvector   │   │
                    │   │ raw archives │        │  docs·chunks· │   │
                    │   └──────────────┘        │  graph·queue  │   │
                    │                           └──────┬───────┘   │
                    │   ┌─────────────┐                │           │
   agents / apps ──▶│   │  retrieval  │◀───────────────┘           │
   ◀── answers      │   │  :8081      │  vector + BM25 + graph      │
                    │   │  /query     │  → RRF fusion → synthesis   │
                    │   │  /retrieve  │                             │
                    │   └─────────────┘                             │
                    │   ┌─────────────┐                             │
                    │   │ synthesis   │  nightly: cluster activity   │
                    │   │ (cron)      │  into knowledge pages        │
                    │   └─────────────┘                             │
                    └──────────────────────────────────────────────┘
```

- **Ingestion** (`:8080`) takes signed provider webhooks at
  `POST /webhooks/{source}` (verified in-process by each connector) plus a
  custom-ingest API, persists the raw payload to object storage, and enqueues
  work. `GET /health` for liveness.
- **Worker** drains the queue: normalize → chunk → embed → upsert documents,
  chunks, and the entity graph into Postgres.
- **Retrieval** (`:8081`) serves `POST /retrieve` (raw chunks: vector + BM25 +
  graph fused via RRF) and `POST /query` (retrieval + LLM-synthesized cited
  answer). `GET /health` for liveness.
- **Synthesis** (optional `cron` profile) periodically clusters recent activity
  into knowledge pages.

Full design: [docs/phase0-design.md](docs/phase0-design.md) ·
storage/data flow: [docs/storage-architecture.md](docs/storage-architecture.md) ·
how retrieval works: [docs/retrieval-architecture.md](docs/retrieval-architecture.md).

## Repo layout

```
prbe-knowledge/
├── services/
│   ├── ingestion/      webhook + custom-ingest API (:8080)
│   │   └── handlers/   one connector per source
│   ├── worker/         queue drain: normalize · chunk · embed
│   ├── retrieval/      /retrieve + /query (:8081)
│   ├── mcp/            MCP server: agent tool surface (:8084)
│   ├── synthesis/      knowledge-page generation
│   └── ...
├── shared/             config, db, storage, models, enums
├── db/
│   └── schema.sql      canonical Postgres schema (source of truth)
├── scripts/
│   └── migrate.py      generic DB bootstrap (schema.sql + alembic)
├── deploy/helm/        community Helm chart (probe-knowledge)
├── docker-compose.yml  turnkey local/self-host stack
├── Makefile            up · down · health · query · seed
└── docs/
    ├── self-hosting.md prerequisites, compose, Helm, Postgres, providers
    ├── connectors.md   per-source setup (GitHub App, webhook secrets)
    ├── phase0-design.md
    ├── storage-architecture.md   how data is stored at rest
    └── retrieval-architecture.md how a query becomes a cited answer
```

## Documentation

- [Self-hosting](docs/self-hosting.md) — Docker Compose, Helm, Postgres, model
  providers, object storage, backups.
- [Connectors](docs/connectors.md) — per-source setup, including how to register
  your own GitHub App and where each webhook signing secret comes from.
- [Contributing](CONTRIBUTING.md) — dev setup, gates, PR norms.
- [Design](docs/phase0-design.md) · [Storage architecture](docs/storage-architecture.md) · [Retrieval architecture](docs/retrieval-architecture.md)

### MCP server

The agent tool surface (`search_knowledge`, `query_knowledge`, `get_source`)
ships in-repo at `services/mcp/` (vendored from the former standalone
`prbe-knowledge-mcp`). It proxies to the retrieval service over HTTP
(`KNOWLEDGE_QUERY_URL`) and has two auth modes via `MCP_AUTH_MODE`:

- **`static`** (community self-host) — one shared bearer (`MCP_API_TOKEN`)
  scoped to `DEFAULT_CUSTOMER_ID`. This is the `docker compose` default; point
  your client at `http://localhost:8084/mcp` with `Authorization: Bearer
  $MCP_API_TOKEN`.
- **`oauth`** (Probe-hosted) — validates OAuth 2.1 JWTs against the issuer JWKS
  (`MCP_OAUTH_*`).

Mode auto-resolves from the env (static when only `MCP_API_TOKEN` is set, oauth
when a JWKS URL is set); set `MCP_AUTH_MODE` to force one.

## Contributing & community

- [Contributing](CONTRIBUTING.md) — dev setup, gates, PR norms.
- [Code of Conduct](CODE_OF_CONDUCT.md) — expectations for participation.
- [Security policy](SECURITY.md) — how to report a vulnerability privately.
- [Changelog](CHANGELOG.md) — notable changes per release.

## License

[AGPL-3.0](LICENSE). Copyright © 2026 prbe-ai.

This repository is the canonical engine and the open-source **community
edition** — a single-tenant engine you run yourself. Probe's hosted
multi-tenant control plane (per-tenant routing, gateway trust, usage
metering) is intentionally dormant in this build, not removed; the
community edition makes no calls out to Probe's hosted service.

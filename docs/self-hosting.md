# Self-hosting Probe

The community edition runs as a single tenant: no control plane, no dashboard,
and no egress to Probe's hosted service. Everything is scoped to
`DEFAULT_CUSTOMER_ID` (default `default`) and gated by `KNOWLEDGE_API_TOKEN`.
Postgres row-level security stays on; in single-tenant mode it is trivially
satisfied by the one configured tenant.

## Prerequisites

- **Docker + Docker Compose** (the simplest path), or a Kubernetes cluster with
  Helm 3 for the production path.
- A **Postgres 16** with the `vector`, `pg_trgm`, and `btree_gin` extensions —
  the bundled `pgvector/pgvector:pg16` image ships all three. (Compose provides
  this for you.)
- A **`GOOGLE_API_KEY`** for embeddings (Gemini) and **one LLM key**
  (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) for answer synthesis.

> **Apple Silicon / arm64 caveat.** The Gemini embedding SDK's native `grpcio`
> wheel can crash the ingestion and worker containers with `SIGILL` (exit code
> 132) on arm64. Run the stack on **`linux/amd64`**:
>
> ```bash
> export DOCKER_DEFAULT_PLATFORM=linux/amd64
> make up
> ```
>
> or deploy on an amd64 host. amd64 is unaffected. This only matters for the
> Python services that load the Gemini SDK (ingestion + worker); the database
> and object store are fine on any architecture.

## Path 1: Docker Compose (simplest)

```bash
cp .env.example .env
# Fill: GOOGLE_API_KEY, one LLM key, KNOWLEDGE_API_TOKEN.
make up        # or: docker compose up -d --build
make health
```

The Compose stack starts:

| Service        | Role                                                              |
|----------------|-------------------------------------------------------------------|
| `postgres`     | `pgvector/pgvector:pg16` with the required extensions             |
| `minio`        | bundled S3-compatible object store (console on `:9001`)           |
| `createbuckets`| one-shot: creates the default tenant's bucket                     |
| `migrate`      | one-shot: runs `scripts/migrate.py` (schema + alembic stamp)      |
| `ingestion`    | webhook + custom-ingest API on `:8080`                            |
| `retrieval`    | `/retrieve` + `/query` on `:8081`                                 |
| `worker`       | drains the ingestion queue (normalize · chunk · embed)            |
| `cron`         | optional, behind the `cron` profile: nightly knowledge-page synth |

The `migrate` job runs `scripts/migrate.py`, which:

1. Installs a minimal `neon_auth` shim (gives one nullable FK a target; unused
   in single-tenant mode).
2. On a **fresh** database, applies `db/schema.sql` and stamps the alembic head.
3. On an **existing** database, runs `alembic upgrade head` (incremental).

Re-run it any time with `make migrate` (idempotent).

The nightly synthesis trigger is **not** part of the default `up`. Fire it once
with `docker compose --profile cron up cron`, or schedule it externally (cron /
Kubernetes CronJob) against `python -m services.synthesis.nightly_trigger`.

## Path 2: Helm (production)

A community Helm chart lives at [`deploy/helm/`](../deploy/helm) (chart name
`probe-knowledge`). It renders the same single image into the ingestion,
retrieval, and worker roles — single-tenant self-host mode, no control plane.
Point it at your own Postgres and object store via values, supply the same
secrets described below, and install with `helm install`. See the chart's
`values.yaml` for the full surface.

## Choosing Postgres

**Bundled (Compose).** Nothing to do — Compose sets `DATABASE_URL` and
`DATABASE_URL_SYNC` to the bundled `pgvector/pgvector:pg16`.

**Managed / external.** Point the engine at your own Postgres 16 and make sure
the three extensions are installed (`CREATE EXTENSION` for `vector`, `pg_trgm`,
`btree_gin`). Set both DSNs:

```bash
DATABASE_URL=postgresql://user:pass@host:5432/prbe_knowledge          # asyncpg (app)
DATABASE_URL_SYNC=postgresql+psycopg://user:pass@host:5432/prbe_knowledge  # psycopg (alembic)
```

The async URL drives the services; the `_SYNC` URL drives alembic during
migration. Run `scripts/migrate.py` (or `make migrate`) once against the new
database to apply the schema.

## Choosing LLM / embedding providers

Embeddings always use Gemini, so `GOOGLE_API_KEY` is required. Answer synthesis
and routing use one LLM provider — set at least one of `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`.

Two ways to reach providers:

- **Direct (default).** Set the provider keys above; the engine calls each
  provider SDK directly.
- **Gateway.** Route every LLM and embedding call through a LiteLLM-compatible
  gateway by setting `LLM_GATEWAY_URL` (and `LLM_GATEWAY_KEY`). Useful for
  central key management, rate limiting, and spend tracking.

## Object storage

The engine stores raw webhook/ingest payloads (for replay and debugging) in an
S3-compatible object store, configured with the `R2_*` variables — these work
for S3, Cloudflare R2, and MinIO alike:

```bash
R2_ENDPOINT_URL=http://minio:9000     # S3/R2 endpoint
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_REGION=auto
R2_BUCKET_PREFIX=prbe-knowledge       # bucket is <prefix>-<customer_id>
```

Compose points these at the bundled MinIO and pre-creates the default tenant's
bucket. For S3 or R2, set the endpoint, credentials, and region to your provider
and create the `<R2_BUCKET_PREFIX>-<DEFAULT_CUSTOMER_ID>` bucket (e.g.
`prbe-knowledge-default`).

## Token encryption

Once you connect any OAuth/webhook source, set `TOKEN_ENCRYPTION_KEY` — connector
tokens are encrypted at rest with Fernet. Generate one with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Backups

Two things hold state; back up both:

- **Postgres** — all structured state (documents, chunks, embeddings, the entity
  graph, the ingestion queue). Use your normal Postgres backup tooling
  (`pg_dump`, managed snapshots, PITR). This is the system of record for
  retrieval.
- **Object store** — raw payload archives. These exist for replay and debugging;
  losing them does not lose retrievable knowledge, but back them up if you rely
  on re-processing source events.

The schema can always be rebuilt from `db/schema.sql` via `scripts/migrate.py`,
so a fresh database plus a Postgres restore is enough to recover.

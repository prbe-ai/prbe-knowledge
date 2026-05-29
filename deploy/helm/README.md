# probe-knowledge Helm chart (community edition)

Deploys the open-source **Probe knowledge engine** to Kubernetes: the
**ingestion** API, the **retrieval** API, and the background **worker** — all
from one image, in single-tenant self-host mode.

What this chart does **not** include (by design): the control plane, the
dashboard, any license / pairing / heartbeat, and the MCP server. A community
deployment never talks to `*.prbe.ai`.

## Architecture

One image runs every role; only the container `command` differs:

| Role       | Command                                                              | Port |
|------------|----------------------------------------------------------------------|------|
| ingestion  | `uvicorn services.ingestion.main:app --host :: --port 8080`          | 8080 |
| retrieval  | `uvicorn services.retrieval.main:app --host :: --port 8081`          | 8081 |
| worker     | `python -m services.ingestion.worker`                                | —    |
| migrate    | `python scripts/migrate.py` (one-shot, pre-install/pre-upgrade hook) | —    |

The migration runs as a Helm **pre-install,pre-upgrade hook Job** (hook-weight
`-5`) so the schema is ready before the Deployments roll. `scripts/migrate.py`
applies `db/schema.sql` and stamps the alembic head on a fresh DB, or runs
`alembic upgrade head` on an existing one.

## Prerequisites

- Kubernetes 1.23+ and Helm 3.
- **A container image.** No public image is published yet, so build and push
  one from the repo root:

  ```sh
  docker build -t <your-registry>/prbe-knowledge:0.1.0 \
    -f services/ingestion/Dockerfile .
  docker push <your-registry>/prbe-knowledge:0.1.0
  ```

  Then set `image.repository` / `image.tag` to match.

- **Postgres 16** reachable from the cluster, with the extensions `vector`
  (pgvector), `pg_trgm`, and `btree_gin`. The migrate job creates these (they
  ship in `db/schema.sql`) if the DB role is allowed to `CREATE EXTENSION`;
  otherwise create them once as a superuser first. (The bundled
  `pgvector/pgvector:pg16` option below already has them available.)

- **An S3-compatible object store** (Cloudflare R2, AWS S3, or MinIO) for raw
  payload archives, plus an access key / secret.

- **API keys**: a `GOOGLE_API_KEY` (Gemini embeddings) and at least one LLM key
  (Anthropic by default, or OpenAI), plus a Fernet `TOKEN_ENCRYPTION_KEY` and a
  static `KNOWLEDGE_API_TOKEN` bearer for the retrieval API.

  ```sh
  # Fernet key for connector tokens at rest:
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  # Bearer token for /query + /retrieve:
  openssl rand -hex 32
  ```

## Install — bring-your-own Postgres (recommended)

Create a values overlay (keep it out of git — it holds secrets):

```yaml
# my-values.yaml
image:
  repository: <your-registry>/prbe-knowledge
  tag: "0.1.0"

postgresql:
  external:
    url: postgresql://prbe:secret@my-pg:5432/prbe_knowledge
    urlSync: postgresql+psycopg://prbe:secret@my-pg:5432/prbe_knowledge

config:
  defaultCustomerId: default
  objectStore:
    endpointUrl: https://<accountid>.r2.cloudflarestorage.com
    region: auto
    bucketPrefix: prbe-knowledge

secrets:
  r2AccessKeyId: "<r2-access-key-id>"
  r2SecretAccessKey: "<r2-secret-access-key>"
  googleApiKey: "<gemini-api-key>"
  anthropicApiKey: "<anthropic-api-key>"
  tokenEncryptionKey: "<fernet-key>"
  knowledgeApiToken: "<openssl-rand-hex-32>"
```

Install:

```sh
helm install probe-knowledge ./deploy/helm \
  -n probe-knowledge --create-namespace \
  -f my-values.yaml
```

### Using your own pre-created Secret

Set `secrets.create=false` and `secrets.existingSecret=<name>`. The Secret must
contain the keys this chart references: `DATABASE_URL`, `DATABASE_URL_SYNC`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `GOOGLE_API_KEY`,
`TOKEN_ENCRYPTION_KEY`, `KNOWLEDGE_API_TOKEN`, and any optional provider /
connector keys you use (e.g. `ANTHROPIC_API_KEY`, `GITHUB_APP_PRIVATE_KEY`).

## Install — bundled Postgres (kick-the-tires only)

For a quick local trial you can let the chart run a single-replica
`pgvector/pgvector:pg16` StatefulSet. **Not for production.** When enabled it
overrides `postgresql.external.*` and the DSNs are derived automatically.

```sh
helm install probe-knowledge ./deploy/helm \
  -n probe-knowledge --create-namespace \
  --set postgresql.bundled.enabled=true \
  --set image.repository=<your-registry>/prbe-knowledge \
  --set image.tag=0.1.0 \
  -f my-values.yaml   # still needs the object-store + API-key secrets
```

## Connectors (optional)

To ingest from GitHub / Slack / Linear / Notion / Sentry, fill the matching
keys under `secrets.connectors` (each is wired in only when non-empty), point
the source's webhook at the ingestion Service / ingress, and the engine will
verify and process inbound events in-process — no gateway required.

## Exposing the API

`ingress.enabled=false` by default; both APIs are `ClusterIP` only. To expose
the retrieval API publicly:

```yaml
ingress:
  enabled: true
  className: nginx
  retrievalHost: knowledge.example.com
  # ingestionHost: ingest.example.com   # optional, for inbound webhooks
  tls:
    - secretName: knowledge-tls
      hosts: [knowledge.example.com]
```

## Common operations

```sh
# Upgrade (re-runs the migrate hook, then rolls the Deployments):
helm upgrade probe-knowledge ./deploy/helm -n probe-knowledge -f my-values.yaml

# Tail a role:
kubectl -n probe-knowledge logs deploy/probe-knowledge-retrieval -f

# Uninstall (does not delete an external DB or object store):
helm uninstall probe-knowledge -n probe-knowledge
```

## Key values

| Value | Default | Purpose |
|-------|---------|---------|
| `image.repository` / `image.tag` | `ghcr.io/prbe-ai/prbe-knowledge` / appVersion | Engine image (build it yourself for now) |
| `config.defaultCustomerId` | `default` | Single-tenant id (RLS on, trivially satisfied) |
| `config.objectStore.*` | — | S3/R2/MinIO endpoint, region, bucket prefix |
| `postgresql.external.url` / `.urlSync` | — | BYO Postgres DSNs (asyncpg / psycopg) |
| `postgresql.bundled.enabled` | `false` | Run a convenience pgvector StatefulSet |
| `secrets.*` | placeholders | Provider keys + tokens (or use `existingSecret`) |
| `migrate.enabled` | `true` | Run the pre-install/upgrade migrate hook |
| `{ingestion,retrieval,worker}.replicas` | `1` | Per-role replica counts |
| `ingress.enabled` | `false` | Expose the HTTP APIs |

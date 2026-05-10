"""Fixtures for the directed-phrase generation eval.

Five fixtures:
  - prod_repo_overview: real wiki:repo:prbe_knowledge body from prod
  - prod_runbook_data_backfills: real wiki:runbook:data_backfills body from prod
  - synthetic_tiny: very short page (stress-test minimum-info case)
  - synthetic_dense_code: code-heavy runbook (stress-test code/text mix)
  - synthetic_ambiguous: a runbook with cross-cutting concerns

Each fixture also carries `engineer_queries`: 3 paraphrased symptom queries
an engineer would type when needing this page. Retrieval-fitness scoring
embeds these and measures cosine similarity to the model-emitted phrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fixture:
    fixture_id: str
    title: str
    body: str
    engineer_queries: list[str] = field(default_factory=list)
    note: str = ""


_PROD_REPO_BODY = """`prbe-knowledge` is the core knowledge ingestion and retrieval service for Probe. It parses raw data from integrations (GitHub, Slack, Notion, etc.), extracts structure and edges, and powers intelligent search queries.

## Core Architecture and Data Flow

### 1. Ingestion Pipeline
Ingests data via API endpoints and webhooks, processing documents and chunks into the PostgreSQL database. The chunker processes text into 512-token windows with a 64-token overlap. Uses `pgvector` for vector similarity and full-text search (BM25) on text. Primary owner: [[person:richardwei6]].

- **Embeddings**: Currently migrating from OpenAI to `gemini-embedding-2-preview`. Ingestion dual-writes newly ingested chunks to `chunks.embedding_v2` (backed by an HNSW index), while the query path temporarily remains on OpenAI pending a full historical backfill.
- **GitHub Integration**: Ingests `release` and `commit_comment` webhooks as Document nodes. Extracts both `position` (diff hunk offset) and `line` (file blob line number) from `commit_comment` events.
- **Notion Integration**: Integrates via webhooks and REST API (v2026-03-11), handling page content, property changes, databases, and `data_source` events.
- **Slack Backfills**: Handles historical syncs robustly, shielding termination release logic so SIGTERMs cleanly revert backfill state to 'pending'.

### 2. Code Graph
Maintains structural representations of connected GitHub repositories. Extracts symbol-level structure (Module, Class, Function) using Tree-sitter for multiple languages. Derives structural edges (CALLS, IMPORTS, INHERITS).

### 3. Inferred Edges
An asynchronous pipeline that uses an LLM (Claude Haiku) to discover semantic cross-source edges between documents (e.g., a Slack thread `DISCUSSES` a GitHub PR). Edges are stored with justifications. Processed via an asynchronous drain worker.

### 4. Retrieval System
Handles search queries by routing through an LLM-based query classifier. Combines multiple strategies fused with Reciprocal Rank Fusion (RRF):
  - **Vector**: Standard embedding similarity search.
  - **BM25**: Full-text keyword search using `to_tsquery`. BM25 execution is dynamically skipped when the query contains a known lookup identifier (UUID, ticket ref) and the residual string lacks sufficient topical signal.
  - **Graph**: Explores local neighborhoods using extracted entities. Code graph results are heavily penalized (0.3x score multiplier) to prevent common code identifiers ("session", "tenant") from overpowering non-code semantic matches.
  - **Inferred Edges**: A fourth retrieval channel that walks LLM-derived Doc-Doc semantic edges.
  - **Directed Vectors**: An HNSW lookup against document-level trigger phrases (both human-pinned and LLM-generated).
  - **ID Lookup**: Exact-ID searches for UUIDs or ticket codes.

### 5. Wiki Synthesis
An automated system that analyzes the ingested knowledge base and generates high-level wiki pages for the customer's dashboard.

### 6. Cron Jobs
Runs nightly operations on a consolidated Fly app (`prbe-knowledge-cron`):
- Leiden community detection for graph clustering.
- Nightly wiki synthesis trigger.

## Runbooks
- **[[runbook:adding_fly_app]]**: Procedures for bootstrapping a new Fly app and syncing secrets.
- **[[runbook:data_backfills]]**: Manually executing historical backfills for code graph and inferred edges.
- **[[runbook:synth_generation]]**: Commands for running the synthetic narrative pipeline."""


_PROD_BACKFILLS_BODY = """There are several manual scripts to run backfills for existing customers. These are generally run via a Fly SSH console into the worker machine (`flyctl ssh console -a prbe-knowledge-worker`).

## Embeddings Migration Backfill (Gemini)
To backfill `chunks.embedding_v2` with `gemini-embedding-2-preview` for historical chunks during the embedding migration, you can run the script via SSH console. It is idempotent, batches up to `--cost-cap`, and utilizes concurrent sub-batching to maximize throughput against the Gemini API.

To run a single process:
```bash
flyctl ssh console --app prbe-knowledge-ingestion
.venv/bin/python -m scripts.backfill_embedding_v2 --cost-cap 100
```

To run multiple parallel workers to speed up the backfill (using `--workers` and `--worker-id` for disjoint partitioning):
```bash
flyctl ssh console --app prbe-knowledge-worker
cd /app
nohup python -u -m scripts.backfill_embedding_v2 --workers 4 --worker-id 0 > /tmp/bf0.log 2>&1 < /dev/null &
```

Verify completion with:
```sql
SELECT COUNT(*) FROM chunks WHERE embedding_v2 IS NULL;
```

## Code Graph Backfill
For tenants who connected GitHub prior to code graph incremental push support, or to do a hard refresh.
```bash
uv run python -m scripts.code_graph_backfill_existing --customer-id <customer_id>
```
To test safely, append `--dry-run`.
(Alternatively, you can reindex from the Dashboard via `POST /api/code-graph/reindex`).

## Inferred Edges Backfill
To populate cross-source AI-inferred edges for historical data:
```bash
uv run python -m scripts.inferred_edges_backfill_existing --customer-id <customer_id> --days 7
```
This enqueues documents into `inferred_edges_queue` which the `prbe-knowledge-side-worker` will drain.

### Retrying Failed Inferred Edges
If bundles fail due to exhaustion of rate-limit retries or extraction errors, you can reset their queue rows and clear low-quality existing edges so the worker will reprocess them:

```sql
DELETE FROM graph_edges WHERE customer_id='<id>' AND extractor_id='inferred_edges:v1' AND confidence != 'EXTRACTED';
UPDATE inferred_edges_queue SET attempts = 0, processing_started_at = NULL, error = NULL
 WHERE customer_id='<id>' AND extractor_id='inferred_edges:v1' AND done_at IS NULL AND attempts >= 3;
```

## Wiki Synthesis Catchup
To force re-evaluation of historical events for wiki generation under a new prompt/threshold:
```bash
uv run python -m scripts.wiki_synthesis_catchup <customer_id> --reset-terminal
```"""


_SYNTHETIC_TINY_BODY = """## Restart Postgres replica

If the read replica falls behind by more than 5 minutes, restart it:

```bash
flyctl machines restart <machine-id> -a prbe-knowledge-replica
```

Watch lag drop in the dashboard."""


_SYNTHETIC_DENSE_CODE_BODY = """## TLS certificate rotation

Triggered when ACME renewal fails or a domain is added.

### 1. Inspect current cert
```bash
openssl s_client -connect prbe.ai:443 -servername prbe.ai </dev/null 2>/dev/null \\
  | openssl x509 -noout -dates -issuer -subject
```

### 2. Force re-issuance via Fly
```bash
flyctl certs delete prbe.ai --yes
flyctl certs add prbe.ai --app prbe-knowledge-retrieval
flyctl certs show prbe.ai --app prbe-knowledge-retrieval
```

Wait for `Status: Verified`. ACME challenge is HTTP-01 over port 80; if you see `dns-01: pending`, the wildcard issuance is still spinning. Don't proceed until the leaf cert is issued.

### 3. Verify clients see the new cert
```bash
curl -vI https://prbe.ai 2>&1 | grep -E '^(\\*  expire date|\\* subject)'
```

### 4. Roll Fly machines that cached the old cert
```bash
flyctl machines list -a prbe-knowledge-retrieval --json | jq -r '.[].id' | \\
  xargs -I{} flyctl machines restart {} -a prbe-knowledge-retrieval
```

### Failure modes

* `acme: rate limit exceeded` — Let's Encrypt has a 5-cert-per-week-per-domain limit. Wait it out or use the staging endpoint.
* `dns: no address record` — apex CNAME is missing. Confirm with `dig prbe.ai +short`.
* Cert valid in browser but app rejects: Python `ssl.SSLCertVerificationError` usually means the cert chain is incomplete. Check `flyctl certs show` for the chain.
"""


_SYNTHETIC_AMBIGUOUS_BODY = """## On-call escalation policy

When something is on fire, this is who responds.

### Tiers

* **Tier 1 — automated alerts**: Opsgenie pings the rotating on-call. Acknowledge within 5 min.
* **Tier 2 — customer-reported**: Reported via Intercom or shared Slack channel. The on-call triages and either fixes or escalates.
* **Tier 3 — exec escalation**: For data loss, security incidents, or > 1h customer-visible outage. Page Richard directly. Page Mahit if Richard is unreachable for > 15 min.

### Acknowledgment

Acknowledge in Opsgenie AND post in `#incidents` Slack with the format:
> [INC-NNN] <one-line summary> — investigating

If the incident touches customer data, also post in `#customers` (no PII).

### Investigation

Pull the relevant traces from Honeycomb (`service.name = "..."`). Check Sentry for unhandled exceptions in the same window. If multiple services are affected, look at Fly machine status across the app set.

For database problems: check Neon console for slow queries, lock timeouts, or replication lag. Don't run `EXPLAIN ANALYZE` against prod under load — use a read replica.

### Comms

Status updates every 15 min in `#incidents` until mitigated. After mitigation:
1. Open a Linear ticket tagged `incident` with the timeline.
2. Close the Opsgenie alert.
3. Schedule a blameless postmortem within 5 business days.

### Common false alarms

* "Replication lag spike" — usually a single long-running query holding a snapshot. Check `pg_stat_activity`.
* "OpenAI 429" — embed_many's recursive split-retry handles this. Don't page unless the dashboard shows sustained > 5min of 429s.
* "Health check fail on a single machine" — Fly auto-replaces. Only intervene if > 2 of N machines are flapping."""


FIXTURES: list[Fixture] = [
    Fixture(
        fixture_id="prod_repo_overview",
        title="prbe-knowledge",
        body=_PROD_REPO_BODY,
        engineer_queries=[
            "BM25 keyword search overpowered by code identifiers",
            "embedding migration dual write rollback",
            "wiki page not surfacing for symptom-style queries",
        ],
        note="Real prod page; describes the entire knowledge service. Hard for an LLM to pick distinct triggers because everything overlaps.",
    ),
    Fixture(
        fixture_id="prod_runbook_data_backfills",
        title="Data Backfills",
        body=_PROD_BACKFILLS_BODY,
        engineer_queries=[
            "tenant code graph stale need reindex",
            "inferred edges queue stuck retry rate limit",
            "embedding_v2 column null for historical chunks",
        ],
        note="Real prod runbook; clear procedural intent. Should produce sharp symptom-style triggers.",
    ),
    Fixture(
        fixture_id="synthetic_tiny",
        title="Restart Postgres replica",
        body=_SYNTHETIC_TINY_BODY,
        engineer_queries=[
            "read replica lagging far behind primary",
            "fly postgres replica falling behind",
            "stale data on read replica",
        ],
        note="Tiny page, single procedure. Tests whether models gracefully avoid padding.",
    ),
    Fixture(
        fixture_id="synthetic_dense_code",
        title="TLS certificate rotation",
        body=_SYNTHETIC_DENSE_CODE_BODY,
        engineer_queries=[
            "ssl certificate expired browser warning",
            "let's encrypt rate limit hit cant renew",
            "fly app rejecting TLS connections",
        ],
        note="Code-heavy. Tests whether models extract intent from procedural shell snippets without regurgitating commands as triggers.",
    ),
    Fixture(
        fixture_id="synthetic_ambiguous",
        title="On-call escalation policy",
        body=_SYNTHETIC_AMBIGUOUS_BODY,
        engineer_queries=[
            "incident response who do I page",
            "production outage escalation procedure",
            "customer reported bug how to triage",
        ],
        note="Cross-cutting policy doc. Tests whether models resist generating overly-broad triggers that would match unrelated runbooks.",
    ),
]

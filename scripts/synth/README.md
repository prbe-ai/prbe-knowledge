# `scripts/synth` — Synthetic Corpus Generator

Generate realistic synthetic Slack/Notion/GitHub/Linear/Sentry events keyed to a
real repo, ingest them through the production worker into the knowledge stack,
and run eval queries against them. Built for offline eval of the retrieval
pipeline without paying real-user privacy or labelling costs.

The tool sits across three layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│  scripts/synth (this directory)                                      │
│  ┌────────────────┐   ┌────────────────┐   ┌─────────────────────┐   │
│  │  WorldModel    │ → │  Archetypes    │ → │  Output wrappers    │   │
│  │  (extracted    │   │  (templated +  │   │  (slack/notion/git  │   │
│  │   from repo)   │   │   LLM-driven)  │   │   /linear/sentry)   │   │
│  └────────────────┘   └────────────────┘   └─────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼ webhook-shape JSON
┌──────────────────────────────────────────────────────────────────────┐
│  Production ingestion pipeline (services/ingestion)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐    │
│  │  R2 bucket   │←→│  ingestion_  │→ │  worker (normalize +     │    │
│  │  payload     │  │  queue       │  │  embed) writes to        │    │
│  │  archive     │  │              │  │  documents/chunks/graph_*│    │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Retrieval (services/retrieval)                                      │
│  /retrieve  → vector + BM25 chunks                                   │
│  /query     → synthesized answer with citations                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Three modes, three reasons to run

| Mode | API keys needed | Writes to disk | Writes to DB+R2 | Use when |
|---|---|---|---|---|
| `--mock-llm` | none | `./out/` only | no | Smoke-test the pipeline; require pre-recorded fixtures for plot archetypes |
| Real LLM (default) | Anthropic + OpenAI | `./out/` only | no | Generate the eval corpus locally, inspect artifacts |
| `--integrate` (no flag) | Anthropic + OpenAI | yes | yes | Drive synth data through the full prod ingestion stack so retrieval works |
| `--record-llm` | Anthropic + OpenAI | yes | no | Refresh `tests/fixtures/synth_llm/` after prompt changes |

`--mock-llm` and `--integrate` compose. `--record-llm` is mutually exclusive with `--mock-llm`.

## Quickstart — local dev with the full stack

### 1. Bring up Postgres + MinIO (R2 stand-in)

```bash
docker compose up -d postgres minio
```

### 2. Run database migrations (handles the local `neon_auth` shim)

```bash
export PATH="/opt/homebrew/opt/libpq/bin:$PATH"   # or wherever your psql lives
scripts/neon-migrate.sh local
```

This script is idempotent and does the right thing whether the DB is fresh or
already migrated.

### 3. Generate a Fernet key for token encryption

```bash
TOKEN_KEY=$(.venv/bin/python -c "from shared.encryption import generate_key; print(generate_key())")
```

The same key has to be used by `synth init` (encrypts `integration_tokens.access_token_encrypted`)
and the worker (decrypts on read). Save it somewhere stable for the duration of
the run, e.g. `echo "$TOKEN_KEY" > /tmp/synth-token-key`.

### 4. Write a profile YAML

```yaml
# ~/synth-profiles/smoke.yaml
customer_id: cust-eval-prbe-01     # MUST start with cust-eval- or cust-synth-
preset: tiny_test                  # see scripts/synth/presets/ for shipped presets
seed: 7                            # determinism contract: same seed → byte-identical output
repos:
  - url: github.com/prbe-ai/prbe-knowledge
    local_path: /Users/you/Desktop/prbe/prbe-knowledge   # path to a real git clone
world_model:
  min_commits_per_persona: 2
  topic_pool_lookback_days: 30
```

Available presets:
- `tiny_test` — STANDUP=1, ON_CALL=1, INCIDENT=2, LAUNCH=1, BIG_REFACTOR=1 (~$0.50–1.00 cold)
- `incident_only` — INCIDENT=8, others=0 (~$1–2 cold)

### 5. Bootstrap the tenant

```bash
TOKEN_ENCRYPTION_KEY="$TOKEN_KEY" \
DATABASE_URL="postgresql://prbe:prbe@localhost:5432/prbe_knowledge" \
R2_ENDPOINT_URL="http://localhost:9000" \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
R2_REGION=auto \
.venv/bin/python -m scripts.synth init --profile ~/synth-profiles/smoke.yaml
```

Creates the `customers` row (with a deterministic placeholder `api_key_hash`),
ensures the per-tenant R2 bucket exists, and writes Fernet-encrypted
`integration_tokens` stubs for slack + notion. Idempotent — safe to re-run.

### 6. Generate + push to R2 + queue

```bash
TOKEN_ENCRYPTION_KEY="$TOKEN_KEY" \
DATABASE_URL="postgresql://prbe:prbe@localhost:5432/prbe_knowledge" \
R2_ENDPOINT_URL="http://localhost:9000" \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
R2_REGION=auto \
ANTHROPIC_API_KEY="sk-ant-..." \
OPENAI_API_KEY="sk-..." \
.venv/bin/python -m scripts.synth run \
  --profile ~/synth-profiles/smoke.yaml \
  --integrate \
  --output-dir /tmp/wm-integrate
```

Walks the local repo to build a WorldModel, runs templated + LLM-driven
archetype builders, validates each scenario, wraps each emitted SynthDoc as a
webhook-shape envelope, and:

- writes a local mirror to `/tmp/wm-integrate/raw/<source>/<event_id>.json`
- pushes the same envelope bytes to R2 at `raw/<source>/<customer_id>/synth/<event_id>.json`
- batch-inserts a row into `ingestion_queue` per doc

### 7. Drain the queue with the production worker

```bash
TOKEN_ENCRYPTION_KEY="$TOKEN_KEY" \
DATABASE_URL="postgresql://prbe:prbe@localhost:5432/prbe_knowledge" \
DATABASE_URL_SYNC="postgresql+psycopg://prbe:prbe@localhost:5432/prbe_knowledge" \
R2_ENDPOINT_URL="http://localhost:9000" \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
R2_REGION=auto \
INTERNAL_KNOWLEDGE_API_KEY="dev-internal-$(uuidgen | tr -d -)" \
ANTHROPIC_API_KEY="sk-ant-..." \
OPENAI_API_KEY="sk-..." \
.venv/bin/python -m services.ingestion.worker
```

The worker is a long-running service — `Ctrl-C` after the queue is empty
(check via `SELECT status, COUNT(*) FROM ingestion_queue WHERE customer_id='cust-eval-prbe-01' GROUP BY status;`).
It normalizes each envelope through the same connector handlers prod uses,
chunks the document body, calls OpenAI for `text-embedding-3-large`, and
writes to `documents`, `chunks` (with embeddings), `graph_nodes`,
`graph_edges`, `acl_snapshots`.

### 8. Query via the retrieval API

```bash
TOKEN_ENCRYPTION_KEY="$TOKEN_KEY" \
INTERNAL_KNOWLEDGE_API_KEY="dev-internal-..." \
DATABASE_URL="postgresql://prbe:prbe@localhost:5432/prbe_knowledge" \
DATABASE_URL_SYNC="postgresql+psycopg://prbe:prbe@localhost:5432/prbe_knowledge" \
R2_ENDPOINT_URL="http://localhost:9000" \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
R2_REGION=auto \
ANTHROPIC_API_KEY="sk-ant-..." \
OPENAI_API_KEY="sk-..." \
.venv/bin/uvicorn services.retrieval.main:app --port 8001
```

Then curl with service-to-service auth:

```bash
# /retrieve — vector + BM25 chunks, no synthesis
curl -s -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KEY" \
  -H "X-Prbe-Customer: cust-eval-prbe-01" \
  -d '{"query": "what was the recent on-call status", "top_k": 5}'

# /query — full RAG with synthesized answer + citations
curl -s -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KEY" \
  -H "X-Prbe-Customer: cust-eval-prbe-01" \
  -d '{"query": "summarize the recent on-call handoffs", "top_k": 5}'
```

## Inspecting the output

### Local artifacts (`--output-dir`)

```
<out>/
├── manifest.json              # run summary: archetypes_executed, totals, warnings
├── docs_index.jsonl           # one line per emitted SynthDoc, sorted by occurred_at
├── profile.yaml               # frozen copy of the resolved profile (with preset merged)
├── world_model.json           # full WorldModel snapshot
├── company_context.json       # CompanyContext used by LLM planner/writer
├── warnings.log               # Pass 1 validator violations
├── scenarios/                 # one JSON per ScenarioSpec (incl. plot eval_questions)
│   └── scn-<archetype>-<title>-<date>.json
├── questions.jsonl            # eval questions with cross-source evidence_doc_keys
└── raw/<source>/              # webhook-shape envelopes (slack/notion/github/linear/sentry)
    └── <event_id>.json
```

### Database queries

```bash
# All docs
PGPASSWORD=prbe /opt/homebrew/opt/libpq/bin/psql -h localhost -p 5432 -U prbe -d prbe_knowledge -x -c "
  SELECT source_system, title, body_size_bytes, body_preview
  FROM documents WHERE customer_id='cust-eval-prbe-01' ORDER BY created_at;
"

# Full body via chunks (joins skip the index=-1 metadata chunk)
PGPASSWORD=prbe /opt/homebrew/opt/libpq/bin/psql -h localhost -p 5432 -U prbe -d prbe_knowledge -x -c "
  SELECT d.source_system, d.title,
         string_agg(c.content, E'\n---\n' ORDER BY c.chunk_index) AS body
  FROM documents d
  JOIN chunks c ON c.doc_id = d.doc_id AND c.customer_id = d.customer_id
  WHERE d.customer_id='cust-eval-prbe-01' AND c.chunk_index >= 0
  GROUP BY d.doc_id, d.source_system, d.title, d.created_at
  ORDER BY d.created_at;
"

# Queue health
PGPASSWORD=prbe /opt/homebrew/opt/libpq/bin/psql -h localhost -p 5432 -U prbe -d prbe_knowledge -c "
  SELECT status, COUNT(*) FROM ingestion_queue
  WHERE customer_id='cust-eval-prbe-01' GROUP BY status;
"
```

## Cleanup

```bash
TOKEN_ENCRYPTION_KEY="$TOKEN_KEY" \
DATABASE_URL="postgresql://prbe:prbe@localhost:5432/prbe_knowledge" \
R2_ENDPOINT_URL="http://localhost:9000" \
R2_ACCESS_KEY_ID=minioadmin R2_SECRET_ACCESS_KEY=minioadmin R2_REGION=auto \
.venv/bin/python -m scripts.synth clean --customer cust-eval-prbe-01
```

Drops every row across `CUSTOMER_OWNED_TABLES` (14 tables — `ingestion_queue`,
`chunks`, `documents`, `graph_*`, `acl_*`, etc.) plus all R2 keys under
`raw/<source>/<customer_id>/synth/`. The `customers` row is preserved as a
"tenant exists" marker so re-init doesn't race.

**Hard guard:** refuses any `customer_id` that doesn't start with `cust-eval-` or
`cust-synth-`. This is intentional — synth never touches real customer data.

## Seeding a real-shape tenant (Plan 4 V1)

For demoing Probe to a customer who hasn't connected real sources yet.
Admin-triggered, costs ~$0 per seed (replays a canonical corpus, no LLM at
seed time). The customer can never trigger this themselves.

### One-time setup: record the canonical corpus

(Already done if `scripts/synth/canonical/v1/raw/` is committed in this repo.
Re-run only if you need to regenerate after a model upgrade or to capture
new archetypes.)

Choose or copy a profile from `scripts/synth/profiles/` and adjust its
`customer_id` to `cust-eval-canonical-v1` before running:

````bash
python -m scripts.synth init --profile scripts/synth/profiles/<canonical-profile>.yml
python -m scripts.synth run \
    --profile scripts/synth/profiles/<canonical-profile>.yml \
    --integrate --record-llm --archetypes standup,oncall \
    --output-dir scripts/synth/canonical/v1/
python -m scripts.synth clean --customer cust-eval-canonical-v1
git add scripts/synth/canonical/v1/raw/
git commit -m "chore(synth): refresh canonical v1 corpus"
````

### Per-customer seed: two valid paths

**Path 1 — opt-in flag (recommended for any customer you'll seed more than once):**

````bash
python -m scripts.synth allow-seed --customer cust-prbe-acme-co
python -m scripts.synth seed --customer cust-prbe-acme-co
````

The first command sets `customers.metadata.allow_synth_seed = true`. Once
the flag is set, future re-seeds are a single `synth seed` call with no
prompt.

**Path 2 — escape hatch (one-off, no DB state change):**

````bash
python -m scripts.synth seed --customer cust-prbe-acme-co --allow-non-sandbox
# Prompts: type "cust-prbe-acme-co" to confirm.
````

Use this for one-time seeds where you don't want to leave the metadata flag
behind.

### Cleanup caveat (V1)

`synth clean --customer cust-prbe-acme-co` will refuse the customer (existing
prefix gate stays — V1 does not extend `clean_tenant`). To wipe a real-shape
tenant, you currently have to either:
- Drop the customer row by hand via SQL (cascade deletes all DB rows;
  R2 objects are NOT cascade-deleted and will remain as orphans under
  `raw/<source>/<customer_id>/synth/*.json`. Clean them up via the
  MinIO/R2 console, or with:
  `aws s3 rm s3://<bucket>/raw/ --recursive --include "*/<customer_id>/synth/*"`).
- Wait for V2's surgical cleanup (`synth seed clear`) which will remove
  only synth-tagged rows.

For a customer that has connected real sources after being seeded, **do
not** wipe — synth and real data are intermixed and there's no surgical
removal in V1. This is a known V1 limitation; V2 adds per-row provenance
tagging.

### Gate failures

`synth seed` exits non-zero on:
- exit 2: customer not found, no path satisfied, or confirm mismatch
- exit 1: canonical fixtures missing

In all cases, no R2 or DB writes happen.

## Customer ID conventions

| Prefix | Used by | Notes |
|---|---|---|
| `cust-eval-*` | offline eval datasets | Plan 3 default. Eval pipeline points at these tenants. |
| `cust-synth-*` | scratch / dev tenants | One-off experiments, throwaway data. |
| anything else | refused | `synth init`/`run`/`clean` all reject. |

A tenant created by `synth init` has:
- `display_name = synth-<customer_id>`
- `api_key_hash` = sha256 of `synth-stub-no-bearer-<customer_id>` (no real bearer auth path resolves to it)
- `integration_tokens.access_token_encrypted` = Fernet-encrypted `synth-stub` (decryptable but useless against real Slack/Notion APIs)
- A per-tenant R2 bucket: `prbe-knowledge-<customer_id>`

## Modes deep dive

### `--mock-llm`

Replaces all real LLM clients with `MockLlmClient` reading from
`tests/fixtures/synth_llm/<provider>/<sha256-of-prompt>.json`. Templated
archetypes (`STANDUP_UPDATE`, `ON_CALL_HANDOFF`) work without fixtures because
they don't call the LLM. Plot archetypes (`INCIDENT`, `LAUNCH`, `BIG_REFACTOR`)
will FixtureNotFoundError per scenario unless the fixtures exist — useful for
deterministic unit tests, useless for fresh prompt iteration.

### Real LLM (default)

Real Anthropic + OpenAI calls. Wraps each LLM client in a `CachingLlmClient` so
re-runs with the same prompt + temperature get cache hits (free). Plot
archetypes call:
- planner: `claude-opus-4-7` for the structured `ScenarioSpec` (1 call/scenario)
- writer: `claude-sonnet-4-6` for each per-source doc body (1 call/doc/source)
- validator pass 2: `claude-haiku-4-5-20251001` for cross-source consistency check (1 call/scenario)

`tiny_test` preset: ~7–10 LLM calls/scenario × 5 scenarios ≈ 50–80 cold calls,
clamps to ~$0.50–1.00 on first run, $0 on cached re-runs.

### `--record-llm`

Real LLM calls AND writes the responses to the fixture store. Run this after
prompt changes to refresh `tests/fixtures/synth_llm/`. Without it, `--mock-llm`
runs against stale fixtures.

## Plot archetypes vs templated archetypes

| Type | Examples | Driver | Cost |
|---|---|---|---|
| Templated | `STANDUP_UPDATE`, `ON_CALL_HANDOFF` | Hand-written builders, deterministic from `(world, seed)` | $0 |
| Plot | `INCIDENT`, `LAUNCH`, `BIG_REFACTOR` | LLM Planner → ScenarioSpec → LLM Writer per source | ~$0.10–0.30/scenario |

Templated archetypes always succeed (no LLM, no validation drops). Plot
archetypes can fail Pass 1 validation when the LLM mentions kebab-cased tokens
not in the WorldModel allowlist (`auto-scaling`, `well-known`, etc.) —
**Plan 3 currently drops the scenario** when this happens. The Plan 3 spec
called for a regen loop here that hasn't been implemented yet (deferred).

If you see plot scenarios consistently dropping in real-LLM mode, your options
are: (a) loosen the validator's third-party allowlist, (b) demote Pass 1
unknown-name violations to warnings for plot archetypes, or (c) implement the
regen loop. See `docs/superpowers/plans/2026-05-02-synth-plan-3-narrative-layer.md`
for the original design.

## Key environment variables

| Var | Required for | Notes |
|---|---|---|
| `DATABASE_URL` | `synth init`/`run --integrate`/`clean`, worker, retrieval | asyncpg URL: `postgresql://...` |
| `DATABASE_URL_SYNC` | alembic migrations | psycopg URL: `postgresql+psycopg://...` |
| `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_REGION` | `synth run --integrate`, worker, retrieval | local MinIO: `http://localhost:9000`, `minioadmin`/`minioadmin`/`auto` |
| `TOKEN_ENCRYPTION_KEY` | `synth init`, worker, retrieval | Fernet key. `python -c "from shared.encryption import generate_key; print(generate_key())"`. **Must be the same key across init + worker.** |
| `INTERNAL_KNOWLEDGE_API_KEY` | worker, retrieval | Service-to-service shared secret. Any uuid works for local dev. |
| `ANTHROPIC_API_KEY` | `synth run` (real LLM), worker (Claude Code path), retrieval | `sk-ant-...` |
| `OPENAI_API_KEY` | worker (embeddings), retrieval (router) | `sk-...` |

## Common gotchas

**`'Pool' object has no attribute 'transaction'`** — using a stale `synth clean`.
Update to commit `9ac6c4c+`.

**`temperature is deprecated for this model`** — using a stale Anthropic
client. Update to commit `e005c80+`.

**`null value in column "api_key_hash" of relation "customers"`** — using a
stale `init_tenant`. Update to commit `89a27ba+`.

**`TOKEN_ENCRYPTION_KEY is not set` (worker DLQs everything)** — same Fernet
key not exported in both the `synth init` shell AND the worker shell. Generate
once, reuse everywhere.

**Notion docs ingest with empty title + body** — using a stale notion handler.
Update to commit `93e83eb+`. Synth's notion wrapper inlines `entity.body_markdown`
because the prod handler would otherwise call Notion's API (which 401s without
a real OAuth token).

**Plot archetypes all drop in real-LLM mode** — known Plan 3 issue, regen loop
not implemented. See "Plot archetypes vs templated archetypes" above.

**`column "occurred_at" does not exist` on ingestion_queue** — using a stale
`IngestionWriter`. Update to commit `e005c80+`.

**psql command not found** — install via `brew install libpq && export PATH="/opt/homebrew/opt/libpq/bin:$PATH"`,
or use the in-container psql: `docker exec -it prbe-knowledge-postgres psql -U prbe -d prbe_knowledge`.

## Where things live

```
scripts/synth/
├── README.md                  # this file
├── cli.py                     # `synth init / run / clean / extract / allow-seed / seed` argparse + main
├── bootstrap.py               # init_tenant + clean_tenant (DB + R2 customer lifecycle)
├── seed.py                    # Plan 4: synth allow-seed + synth seed gate stack and orchestration
├── profile.py                 # YAML loader + Profile dataclass
├── scenarios.py               # async run_scenarios — yields (ScenarioSpec, SynthDoc)
├── validator.py               # Pass 1 (name allowlist) + combined Pass 2 wrapper
├── world_model.py             # WorldModel extracted from real repo signals
├── company_context.py         # CompanyContext stub or LLM-inferred
├── ownership.py               # CODEOWNERS-derived OwnershipIndex
├── archetypes/
│   ├── library.py             # ARCHETYPE_LIBRARY + BUILDERS + PLOT_BUILDERS registries
│   ├── standup.py, oncall.py  # Templated archetypes
│   ├── incident.py, launch.py, big_refactor.py   # Plot archetypes (LLM-driven)
│   └── plot_base.py           # Shared planner-prompt assembly + cast picking
├── llm/
│   ├── base.py                # Provider enum + LlmRequest/LlmResponse + Protocol
│   ├── anthropic_client.py, gemini_client.py     # Real LLM clients
│   ├── mock_client.py         # Fixture-replay client
│   ├── cache.py               # PromptCache (DiskCache wrapper)
│   ├── planner.py             # LLMPlanner — single LLM call → ScenarioSpec
│   ├── writer.py              # LLMWriter — per-doc text generation
│   └── validator_pass2.py     # Cheap LLM consistency check across docs in scenario
├── output/
│   ├── writer.py              # IngestionWriter (local + R2 + queue)
│   ├── eval_artifacts.py      # manifest, docs_index, scenarios/, questions.jsonl
│   ├── slack.py, notion.py, github.py, linear.py, sentry.py   # Source wrappers
│   └── base.py                # SynthDoc dataclass
└── presets/
    ├── loader.py              # apply_preset()
    ├── tiny_test.yaml, incident_only.yaml
    └── prompts/               # Planner + writer prompt templates per archetype
```

## Where things are not

The synth tool does **not**:
- Mutate prod customer data (prefix guard refuses any non-`cust-eval-`/`cust-synth-` tenant)
- Call real Slack/Notion/GitHub/Linear/Sentry APIs (envelopes are pre-shaped to mimic webhooks)
- Hit real LLM APIs without your explicit env-var keys
- Persist outside the dev `customers` row + per-tenant R2 bucket

If you want to seed a real-shape customer with synth data for demo/onboarding
purposes, that's the Plan 4 design — see `docs/superpowers/specs/2026-05-04-synthetic-tenant-seeding-design.md`.

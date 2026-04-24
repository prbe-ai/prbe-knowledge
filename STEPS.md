# STEPS — connect + verify end-to-end

This is the runbook for taking `prbe-knowledge` from a cold repo to "real
Slack/GitHub/Linear/Notion/Sentry webhooks flowing into our own tenant and
`/query` returning relevant results."

Sections:

1. [Local verification](#1-local-verification) — 5 min, no external accounts
2. [Configure `.env.local`](#2-configure-envlocal) — keys + secrets
3. [Register the 5 source apps](#3-register-the-5-source-apps) — PRBE's workspace
4. [Bootstrap the PRBE tenant](#4-bootstrap-the-prbe-tenant)
5. [Run the services locally](#5-run-the-services-locally)
6. [Forward webhooks with a tunnel](#6-forward-webhooks-with-a-tunnel)
7. [Connect each source via OAuth](#7-connect-each-source-via-oauth)
8. [Verify ingestion](#8-verify-ingestion)
9. [Run a real query](#9-run-a-real-query)
10. [Ship to Fly + Neon staging](#10-ship-to-fly--neon-staging)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Local verification

This confirms the whole pipeline works on your laptop with no external accounts.

```bash
# From repo root:
docker compose up -d                        # Postgres + MinIO
scripts/neon-migrate.sh local               # applies schema

# Install deps + run tests
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -q
```

Expected: **66 tests pass**. If anything fails, see §11 before proceeding.

---

## 2. Configure `.env.local`

Create `.env.local` in the repo root. The services auto-load it on boot.

```bash
# Generate a Fernet key for OAuth token encryption:
.venv/bin/python -c "from shared.encryption import generate_key; print(generate_key())"

# Generate an admin API key (for the internal provisioning dashboard):
.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(48))"
```

```ini
# .env.local
ENVIRONMENT=local
LOG_LEVEL=DEBUG

DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge
R2_ENDPOINT_URL=http://localhost:9000
R2_ACCESS_KEY_ID=minioadmin
R2_SECRET_ACCESS_KEY=minioadmin

TOKEN_ENCRYPTION_KEY=<paste the Fernet key you just generated>
ADMIN_API_KEY=<paste the admin API key you just generated>

OPENAI_API_KEY=sk-...                       # embeddings
ANTHROPIC_API_KEY=sk-ant-...                # router (optional for Phase 0 smoke)

# Per-source (fill in after §3):
SLACK_CLIENT_ID=
SLACK_CLIENT_SECRET=
SLACK_SIGNING_SECRET=

GITHUB_APP_ID=
GITHUB_APP_PRIVATE_KEY=                     # full PEM, use """ quoting if needed
GITHUB_WEBHOOK_SECRET=

LINEAR_CLIENT_ID=
LINEAR_CLIENT_SECRET=
LINEAR_WEBHOOK_SECRET=

NOTION_CLIENT_ID=
NOTION_CLIENT_SECRET=

SENTRY_CLIENT_ID=
SENTRY_CLIENT_SECRET=
SENTRY_WEBHOOK_SECRET=
```

Without OpenAI/Anthropic keys, embeddings and the Haiku router run in stub mode
(deterministic hash vectors, empty entity extraction). Ingestion still works and
the smoke test still passes — you just won't get semantic ranking.

---

## 3. Register the 5 source apps

Do this once per PRBE workspace. You'll end up with a `client_id`, `client_secret`,
and (where applicable) `signing_secret` for each, pasted into `.env.local`.

### Slack (15 min)

1. https://api.slack.com/apps → **Create New App** → From scratch → "PRBE Knowledge (dev)"
2. **OAuth & Permissions** → scopes:
   - Bot token: `channels:history`, `channels:read`, `groups:history`, `groups:read`, `users:read`, `team:read`
3. **Event Subscriptions** → enable:
   - Request URL: `https://<your-tunnel>.ngrok.app/webhooks/slack` (fill in after §6)
   - Subscribe to bot events: `message.channels`, `message.groups`
4. **Basic Information** → copy:
   - Client ID → `SLACK_CLIENT_ID`
   - Client Secret → `SLACK_CLIENT_SECRET`
   - Signing Secret → `SLACK_SIGNING_SECRET`

### GitHub (20 min)

1. Org settings → **Developer settings** → **GitHub Apps** → **New GitHub App**: "PRBE Knowledge (dev)"
2. **Webhook URL**: `https://<tunnel>/webhooks/github`, generate a **Webhook secret** → `GITHUB_WEBHOOK_SECRET`
3. **Permissions** (read-only across the board):
   - Repository: Contents, Issues, Metadata, Pull requests
   - Organization: Members
4. **Subscribe to events**: `push`, `pull_request`, `issues`, `pull_request_review`
5. Generate a **private key** (PEM file) → paste contents into `GITHUB_APP_PRIVATE_KEY`
6. **App ID** → `GITHUB_APP_ID`
7. Install the app on your org.

After installing the App, grab the `installation_id` from the post-install
redirect URL — e.g. `.../settings/installations/87654321` → `87654321`. Then
seed PRBE so it can mint installation tokens on demand:

```bash
.venv/bin/python -m scripts.github_seed_token \
  --customer prbe-internal \
  --installation-id 87654321
```

Backfill and CODEOWNERS hydration mint fresh installation tokens via the
App private key whenever they need to call GitHub — there's no refresh cron
for GitHub because installation tokens are ~1h-lived and always reissued
from the App JWT.

### Linear (10 min)

1. Linear settings → **API** → **OAuth applications** → **Create new** → "PRBE Knowledge (dev)"
2. **Callback URL**: `https://<tunnel>/oauth/linear/callback`
3. **Webhook URL**: `https://<tunnel>/webhooks/linear`, copy the **webhook signing secret** → `LINEAR_WEBHOOK_SECRET`
4. **Client ID / Secret** → `LINEAR_CLIENT_ID` / `LINEAR_CLIENT_SECRET`

### Notion (10 min)

1. https://www.notion.so/profile/integrations → **+ New integration** → "PRBE Knowledge (dev)"
2. **Type**: Public. **Capabilities**: Read content, Read user information.
3. **Redirect URI**: `https://<tunnel>/oauth/notion/callback`
4. Copy **OAuth client ID / client secret** → `NOTION_CLIENT_ID` / `NOTION_CLIENT_SECRET`
5. **Webhooks**: enable, URL `https://<tunnel>/webhooks/notion` (Notion's official webhook support is beta — if unavailable, skip; the connector also accepts synthetic polls).

### Sentry (10 min)

1. Sentry org → **Settings** → **Integrations** → **Internal Integration** → **Create New Integration**: "PRBE Knowledge (dev)"
2. **Webhook URL**: `https://<tunnel>/webhooks/sentry`
3. **Permissions**: Issue: Read, Event: Read, Project: Read.
4. **Webhooks**: issue, event_alert.
5. **Client ID / Secret / Webhook secret** → `SENTRY_CLIENT_ID` / `SENTRY_CLIENT_SECRET` / `SENTRY_WEBHOOK_SECRET`

---

## 4. Bootstrap the PRBE tenant

```bash
.venv/bin/python -m scripts.bootstrap_customer \
  --id prbe-internal \
  --display-name "PRBE Internal" \
  --redirect-uri "https://<tunnel>/oauth"
```

This:
- Inserts a `customers` row
- Creates the tenant's MinIO bucket (`prbe-knowledge-prbe-internal`)
- Prints an API key (store it — used by Phase 1 agent clients) and the 5 install URLs

The install URLs also get served by the ingestion service — see §7.

---

## 5. Run the services locally

Three processes. Easiest: three terminal tabs.

```bash
# Tab 1 — ingestion (webhooks + OAuth)
.venv/bin/uvicorn services.ingestion.main:app --reload --port 8080

# Tab 2 — worker (drains ingestion_queue)
.venv/bin/python -m services.ingestion.worker

# Tab 3 — retrieval (/query)
.venv/bin/uvicorn services.retrieval.main:app --reload --port 8081
```

Health check:
```bash
curl localhost:8080/health
curl localhost:8081/health
```

---

## 6. Forward webhooks with a tunnel

Sources need a public HTTPS URL to deliver webhooks. Any tunnel works.

### ngrok (simplest)

```bash
ngrok http 8080
```

Copy the `https://<...>.ngrok.app` URL. Put that `<tunnel>` value into every
callback/webhook URL you configured in §3.

### Cloudflare tunnel (longer-lived)

```bash
cloudflared tunnel --url http://localhost:8080
```

---

## 7. Connect each source via OAuth

Open the install URLs that `bootstrap_customer` printed — one at a time — and
approve each. The `/oauth/{source}/callback` endpoint handles the code exchange,
encrypts the token with your Fernet key, and persists it to `integration_tokens`.

Verify:
```bash
.venv/bin/python -c "
import asyncio
from shared.db import init_pool, raw_conn
from shared.config import get_settings

async def main():
    await init_pool(get_settings())
    async with raw_conn() as c:
        rows = await c.fetch(\"SELECT source_system, status, scope FROM integration_tokens WHERE customer_id='prbe-internal'\")
        for r in rows: print(dict(r))

asyncio.run(main())
"
```

Expect one row per source you've connected, all `status='active'`.

---

## 8. Verify ingestion

Send a test event from each source. Easy triggers:

- **Slack**: post `hello from prbe` in any channel the integration is in
- **GitHub**: push a dummy commit to any monitored repo
- **Linear**: create a throwaway issue
- **Notion**: edit any page the integration has access to
- **Sentry**: manually `sentry-cli send-event -m "test"` or trigger a real error

Watch the worker terminal — you should see `normalizer.start` + `normalizer.done`
log lines. Then count what landed:

```bash
docker compose exec postgres psql -U prbe -d prbe_knowledge -c "
  SELECT source_system, count(*)
  FROM documents WHERE customer_id='prbe-internal'
  GROUP BY 1 ORDER BY 1;
"
```

If a source is missing rows:
1. Check `ingestion_queue` — rows with `status='pending'` mean the worker isn't keeping up or hasn't picked them up yet.
2. Rows with `status='dlq'` have a populated `error` column — fix the bug.
3. No rows at all → webhook isn't reaching you. Test the tunnel with `curl` + a signed fixture.

---

## 9. Run a real query

```bash
curl -s localhost:8081/query \
  -H 'content-type: application/json' \
  -d '{"query": "which services has alice deployed this week",
       "customer_id": "prbe-internal",
       "top_k": 10}' | jq
```

You should get back 10 chunks, each with a score + `retriever_scores` breakdown
showing vector/bm25/graph contributions, and per-stage timing in `timing_ms`.

Per Phase 0 success criteria: `/query` p95 should be **<2s warm / <3s cold** against
a 10K chunk corpus. Measure by running the query 20x; the first is cold, warm
from there.

---

## 10. Ship to Fly + Neon staging

### Prerequisites

- Fly account with a payment method
- Neon account (already have the project from scaffold)

### One-time setup

```bash
# 0. Provision the 3 Fly apps
flyctl apps create prbe-knowledge-ingestion --org prbe
flyctl apps create prbe-knowledge-retrieval --org prbe
flyctl apps create prbe-knowledge-worker    --org prbe

# 1. Put secrets on all three apps
for app in prbe-knowledge-ingestion prbe-knowledge-retrieval prbe-knowledge-worker; do
  flyctl secrets set --app $app \
    DATABASE_URL="$(security find-generic-password -a staging -s neon-prbe-knowledge -w)" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    TOKEN_ENCRYPTION_KEY="$TOKEN_ENCRYPTION_KEY" \
    ADMIN_API_KEY="$ADMIN_API_KEY" \
    R2_ENDPOINT_URL="$R2_ENDPOINT_URL" \
    R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    SLACK_SIGNING_SECRET="$SLACK_SIGNING_SECRET" \
    SLACK_CLIENT_ID="$SLACK_CLIENT_ID" \
    SLACK_CLIENT_SECRET="$SLACK_CLIENT_SECRET" \
    GITHUB_APP_ID="$GITHUB_APP_ID" \
    GITHUB_APP_PRIVATE_KEY="$GITHUB_APP_PRIVATE_KEY" \
    GITHUB_WEBHOOK_SECRET="$GITHUB_WEBHOOK_SECRET" \
    LINEAR_CLIENT_ID="$LINEAR_CLIENT_ID" \
    LINEAR_CLIENT_SECRET="$LINEAR_CLIENT_SECRET" \
    LINEAR_WEBHOOK_SECRET="$LINEAR_WEBHOOK_SECRET" \
    NOTION_CLIENT_ID="$NOTION_CLIENT_ID" \
    NOTION_CLIENT_SECRET="$NOTION_CLIENT_SECRET" \
    SENTRY_CLIENT_ID="$SENTRY_CLIENT_ID" \
    SENTRY_CLIENT_SECRET="$SENTRY_CLIENT_SECRET" \
    SENTRY_WEBHOOK_SECRET="$SENTRY_WEBHOOK_SECRET"
done

# 2. Migrate the Neon staging branch
scripts/neon-migrate.sh staging

# 3. Add FLY_API_TOKEN to GitHub → Settings → Secrets → Actions
#    (generates an API token via: flyctl auth token)
```

### Deploy

```bash
flyctl deploy -c fly.ingestion.toml
flyctl deploy -c fly.retrieval.toml
flyctl deploy -c fly.worker.toml
```

Or let the `deploy.yml` workflow handle it on merges to `main`.

### After deploy

Update every webhook / callback URL in §3 from your local tunnel to the Fly
hostname (`https://prbe-knowledge-ingestion.fly.dev`). OAuth tokens don't need
to be reissued — they're the same across environments (we use shared Neon).

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TokenMissing` in worker logs for a source | OAuth not completed for that source | Re-hit `/oauth/<source>/install?customer_id=...` |
| Webhooks returning 401 | Signing secret mismatch | Re-copy from source console → `.env.local`/Fly secrets, restart |
| `ingestion_queue` rows stuck in `processing` | Worker crashed mid-row | `python -m scripts.cron_stuck_queue_reclaim` |
| `/query` returns 0 chunks | No data ingested yet, or customer_id mismatch | Check `SELECT count(*) FROM chunks WHERE customer_id=...`; verify queue drained |
| Embeddings never populated | `OPENAI_API_KEY` missing → stub vectors only | Set key; existing chunks need a re-embedding pass (Phase 1 TODO) |
| OAuth callback `invalid redirect_uri` | `redirect_uri` passed at install ≠ what source has registered | Make them identical, including scheme + path |
| `TenantIsolationError` at runtime | Code path querying graph tables outside `with_tenant()` | Wrap the query in `with_tenant(customer_id)` |
| RLS test passes locally but "data visible everywhere" in staging | App connected as superuser / BYPASSRLS role | Use a non-super role in prod — Neon's default `neondb_owner` is fine |
| `failed_chunks` rows piling up | Chunks exceeding embedding context length | Look at `content_preview`; Phase 1 introduces structural chunking |

### Common dev-setup foot-guns

- **Fernet key rotation**: change `TOKEN_ENCRYPTION_KEY` and every stored token becomes undecryptable. Rotation is a Tier 7 TODO — for now, treat the key as permanent per environment.
- **ngrok session rotation**: free-tier URLs change on every restart. Save cycles with a paid subdomain or Cloudflare tunnel.
- **Slack app install scope**: installing the app on a workspace ≠ adding the bot to a channel. Add `@prbe-knowledge` to each channel you want ingested.
- **GitHub Apps vs OAuth Apps**: this connector expects a GitHub App (not a Personal Access Token). App installations ≠ OAuth user tokens.
- **Python 3.14 local**: Python 3.12 is what CI + Fly run. 3.14 mostly works; some asyncio edge cases differ. If you hit one, match CI's 3.12.

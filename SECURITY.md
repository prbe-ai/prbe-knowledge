# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in this project,
please report it privately. **Do not open a public issue, pull request, or
discussion for security reports.**

Email **security@prbe.ai** with:

- a description of the issue and its impact,
- steps to reproduce (a proof of concept, if you have one), and
- any affected versions or configuration.

Please give us a reasonable window to investigate and ship a fix before any
public disclosure. We aim to acknowledge a report within **3 business days**
and to provide a remediation timeline within **10 business days**. We follow a
**90-day** coordinated-disclosure window by default and will keep you updated
as we work through a fix.

## Supported versions

Security fixes are applied to the latest released version on the default
branch. Older versions are not maintained; please upgrade to the latest
release before reporting.

## Scope

This repository is the knowledge engine. Reports about a hosted deployment you
do not operate should go to the operator of that deployment. Configuration
secrets (API keys, tokens, encryption keys) are supplied at runtime via
environment variables and are never committed to this repository — see
`.env.example`.

## Hardening a self-hosted deployment

The engine ships with development-friendly defaults. Before exposing a
standalone (community) deployment to any untrusted network:

- **Replace every placeholder secret.** The Helm chart's `CHANGEME` values
  (`knowledgeApiToken`, `tokenEncryptionKey`, `googleApiKey`, R2 credentials)
  must be replaced — the app refuses to boot with them outside
  `ENVIRONMENT=local`. The docker-compose stack runs as `ENVIRONMENT=local`
  and skips that check: its defaults (`local-mcp-token`,
  `local-internal-key`, `minioadmin`, Postgres `prbe:prbe`) are for local
  development only. If you repurpose the compose file for a server, change
  them all and set a non-local `ENVIRONMENT`.
- **Do not expose the ingestion service (port 8080) publicly.** In
  standalone mode `/webhooks/{source}` authenticates each request with the
  provider's webhook signature, which requires the matching secret to be
  configured (`GITHUB_WEBHOOK_SECRET`, `SLACK_SIGNING_SECRET`,
  `LINEAR_WEBHOOK_SECRET`, `NOTION_WEBHOOK_VERIFICATION_TOKEN`,
  `SENTRY_WEBHOOK_SECRET`). Sources without a public webhook surface
  (`code_graph`, `custom_ingest`, `wiki`) reject all standalone webhook
  traffic. Agent-session sources (`claude_code`, `codex`, `manual_upload`)
  have no signature scheme — keep 8080 reachable only from networks you
  trust (compose publishes it on the host; firewall it).
- **Note that `ENVIRONMENT=local` disables webhook signature checks** for
  sources whose secret is unset ("dev bypass"). Never run a network-exposed
  deployment as `local`.
- **Custom ingest and retrieval auth.** `/api/custom-ingest/documents`,
  `/query`, and `/retrieve` accept the static `KNOWLEDGE_API_TOKEN` bearer
  in standalone mode; generate it with `openssl rand -hex 32` and rotate it
  if leaked. The MCP service's `MCP_API_TOKEN` is an equivalent shared
  bearer — generate it the same way.
- **Run Postgres with the non-superuser app role.** A superuser
  `DATABASE_URL` bypasses row-level security; the boot log warns with
  `db.superuser_in_managed_env` outside `local`.

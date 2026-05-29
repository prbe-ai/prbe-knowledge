# Connectors

Probe ingests from these sources:

| Source           | Kind            | Setup needed                                  |
|------------------|-----------------|-----------------------------------------------|
| `github`         | signed webhook  | your own GitHub App + webhook secret          |
| `slack`          | signed webhook  | Slack app signing secret                      |
| `linear`         | signed webhook  | Linear webhook secret                         |
| `notion`         | signed webhook  | Notion verification token                     |
| `sentry`         | signed webhook  | Sentry webhook secret                         |
| `custom_ingest`  | API             | push arbitrary documents (see below)          |
| `claude_code`    | API             | Claude Code session ingest                    |
| `granola`        | API             | Granola meeting-notes ingest                  |
| `manual_upload`  | API             | upload a file/document directly               |
| `wiki`           | internal        | generated knowledge pages                     |

## How webhook intake works (self-host)

In single-tenant community mode there is **no gateway**. Each provider signs its
webhooks; the engine verifies the signature **in-process** using that provider's
signing secret and scopes the event to `DEFAULT_CUSTOMER_ID`. If the secret is
missing or the signature is invalid, the request is rejected.

All providers POST to the same path on the ingestion service:

```
POST http://<your-host>:8080/webhooks/{source}
```

where `{source}` is `github`, `slack`, `linear`, `notion`, or `sentry`. Expose
`:8080` to the internet (reverse proxy / ingress) so providers can reach it, and
register that public URL in each provider's webhook settings.

Set the matching secret in `.env` for every source you enable:

| Source  | `.env` variable                       |
|---------|---------------------------------------|
| GitHub  | `GITHUB_WEBHOOK_SECRET`               |
| Slack   | `SLACK_SIGNING_SECRET`                |
| Linear  | `LINEAR_WEBHOOK_SECRET`               |
| Notion  | `NOTION_WEBHOOK_VERIFICATION_TOKEN`   |
| Sentry  | `SENTRY_WEBHOOK_SECRET`               |

## GitHub (bring your own GitHub App)

Unlike the other sources, GitHub needs a small **GitHub App** that you create and
own. The engine mints installation access tokens locally from the App's private
key — no Probe credentials involved.

### 1. Create the App

GitHub → **Settings → Developer settings → GitHub Apps → New GitHub App**.

- **Webhook URL:** `https://<your-host>/webhooks/github`
- **Webhook secret:** generate a strong random string; you'll put this in
  `.env` as `GITHUB_WEBHOOK_SECRET`.

### 2. Permissions (read-only)

Under **Repository permissions**, grant **Read** to:

- **Contents** — repository files/commits
- **Pull requests**
- **Issues**

### 3. Subscribe to webhook events

At minimum subscribe to the events you want ingested, e.g.:

- **Push**
- **Pull request** (and review/comment events)
- **Issues** (and issue comment events)

### 4. Private key + IDs into `.env`

After creating the App, generate a private key (downloads a `.pem`). Set:

```bash
GITHUB_APP_ID=123456                  # the numeric App ID
GITHUB_APP_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\n...   # full PEM contents, keep the newlines
GITHUB_WEBHOOK_SECRET=...             # the webhook secret from step 1
```

`GITHUB_APP_PRIVATE_KEY` is the **contents** of the `.pem` file (not a path).
Preserve the line breaks — in a `.env` file, paste the multi-line key as-is or
encode the newlines as `\n` consistently with how your loader reads them.

### 5. Install the App

Install the App on the org/repos you want indexed. Pushes, PRs, and issues then
flow to `POST /webhooks/github`; the engine verifies the signature, mints an
installation token from your private key to fetch any extra content it needs, and
ingests.

## Slack

Create a Slack app, enable Event Subscriptions pointed at
`https://<your-host>/webhooks/slack`, and copy the app's **Signing Secret** into
`SLACK_SIGNING_SECRET`. Signatures are verified using the
`X-Slack-Signature` + `X-Slack-Request-Timestamp` headers.

## Linear

Create a Linear webhook pointed at `https://<your-host>/webhooks/linear` and copy
its signing secret into `LINEAR_WEBHOOK_SECRET`. Signatures arrive in the
`linear-signature` header.

## Notion

Configure a Notion integration webhook pointed at
`https://<your-host>/webhooks/notion`. On first setup Notion sends a one-time
payload containing a `verification_token`; set that value as
`NOTION_WEBHOOK_VERIFICATION_TOKEN`.

## Sentry

Create a Sentry internal integration with a webhook pointed at
`https://<your-host>/webhooks/sentry` and copy its client/webhook secret into
`SENTRY_WEBHOOK_SECRET`.

## Custom ingestion

To push your own documents (no provider webhook), use the custom-ingest API on
the ingestion service. It accepts a batch of documents and queues one per row for
the normal normalize → chunk → embed pipeline:

```
POST http://<your-host>:8080/api/custom-ingest/documents
```

This endpoint is service-to-service: it is gated by an internal key and a tenant
header, not by `KNOWLEDGE_API_TOKEN`. For local single-tenant use, send the
`INTERNAL_KNOWLEDGE_API_KEY` (set in the Compose `.env`) and the tenant id:

```bash
curl -X POST http://localhost:8080/api/custom-ingest/documents \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: $DEFAULT_CUSTOMER_ID" \
  -H "Content-Type: application/json" \
  -d '{
        "source_key": "my-docs",
        "documents": [
          {"id": "seed:1", "title": "Hello Probe", "body": "Probe is a self-hosted knowledge engine."}
        ]
      }'
```

Each document requires `id` and `body`; `title`, `type`, `url`, `metadata`, and
`acl` are optional. The envelope requires a `source_key` naming the logical
source and a non-empty `documents` array (up to 100 per request).

## Token encryption

Any connector that stores credentials (OAuth tokens) encrypts them at rest with
Fernet. Set `TOKEN_ENCRYPTION_KEY` before connecting a source — see
[self-hosting.md](self-hosting.md#token-encryption).

---
title: "Auth service"
slug: auth
wiki_type: service_card
last_updated: 2026-04-18T08:02:19Z
owners: [person:maison]
---

# Auth service

The auth service wraps every ingestion handler. It enforces a per-customer
token, rate-limits unauthenticated calls, and surfaces 401s with a
structured body.

## Recent changes

- 2026-04-18 — [[person:maison]] tightened the OAuth refresh window to
  60 seconds ahead of expiry to prevent intermittent 401s during
  long-running ingestion drains
  ([PR #43](https://github.com/prbe-ai/prbe-knowledge/pull/43)).
- 2026-04-05 — Initial OAuth middleware added.

## Sources

- [PR #43: fix(auth): refresh OAuth tokens before 60s expiry window](https://github.com/prbe-ai/prbe-knowledge/pull/43)

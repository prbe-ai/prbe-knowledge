---
title: "Wiki bootstrap from historical sources"
slug: wiki-bootstrap
wiki_type: feature
last_updated: 2026-04-30T17:55:32Z
owners: [person:richard]
contributors: [person:richard, person:maison, person:janedoe]
---

# Wiki bootstrap from historical sources

A per-source bootstrap crawler that walks each connected source's API
recency-first and writes wiki pages directly. Closes the gap where new
customers start with an empty wiki because the queue only contains
events ingested AFTER they connected sources.

The plan is captured in `docs/wiki-bootstrap-plan.md`.

## Status

In flight. Lane B (the typed-link extractor) shipped in
[PR #45](https://github.com/prbe-ai/prbe-knowledge/pull/45). Lane D (the
GitHub crawler) is the first concrete crawler; eval fixtures pre-built
in [PR #46](https://github.com/prbe-ai/prbe-knowledge/pull/46). The
admin-facing trigger button shipped in
[prbe-backend PR #12](https://github.com/prbe-ai/prbe-backend/pull/12).

## Trigger surfaces

- Dashboard "Rebuild wiki" admin button.
- OAuth-callback per-source hook.
- `scripts/wiki_bootstrap.py` for ops.

## Sources

- [Issue #21: feature: wiki bootstrap from historical sources](https://github.com/prbe-ai/prbe-knowledge/issues/21)
- [PR #45: wiki: ship typed-link extractor + persister](https://github.com/prbe-ai/prbe-knowledge/pull/45)
- [PR #46: wiki: bootstrap GitHub crawler eval corpus](https://github.com/prbe-ai/prbe-knowledge/pull/46)
- [prbe-backend PR #12: feat(dashboard): wire wiki bootstrap admin button](https://github.com/prbe-ai/prbe-backend/pull/12)

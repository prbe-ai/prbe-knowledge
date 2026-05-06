---
title: "We picked pgvector over Pinecone"
slug: pgvector-over-pinecone
wiki_type: decision
last_updated: 2026-04-15T10:00:00Z
contributors: [person:richard, person:maison]
related: [service_card:retrieval, vendor:pgvector, vendor:pinecone, vendor:neon]
---

# We picked pgvector over Pinecone

In April 2026 the team committed to [[vendor:pgvector]] inside our existing
[[vendor:neon|Neon]] Postgres instance as the Phase 0 vector store, rather
than standing up a separate [[vendor:pinecone|Pinecone]] cluster.

[[person:richard]] proposed the decision in
[PR #42](https://github.com/prbe-ai/prbe-knowledge/pull/42), closing the
discussion that started in
[issue #18](https://github.com/prbe-ai/prbe-knowledge/issues/18).
[[person:maison|reviewer]] approved the migration on 2026-04-15.

## Rationale

- One less moving part: no separate vector DB to operate.
- Significantly lower monthly cost at our query volume.
- Strong existing Postgres ops muscle memory on the team.
- Hybrid retrieval (BM25 + dense) is straightforward when both indexes
  live in the same database.

## Trade-offs accepted

- pgvector lacks Pinecone's sharding-out-of-the-box; we will revisit at
  >50M vectors.
- IVFFlat tuning is on us.

## Sources

- [PR #42: decision: pick pgvector over Pinecone for vector store](https://github.com/prbe-ai/prbe-knowledge/pull/42)
- [Issue #18: Decision needed: vector store for Phase 0](https://github.com/prbe-ai/prbe-knowledge/issues/18)

"""LLM-driven wiki synthesis loop.

Triages incoming documents (Haiku) and rewrites wiki pages from clusters of
important events (Sonnet). Drains `wiki_synthesis_queue` rows enqueued by
`Normalizer._persist`. Wakes on `pg_notify('wiki_synthesize', customer_id)`
or on a periodic defensive tick.

See plan: docs/wiki/synthesis.md (this Phase 2 work).
"""

"""Inferred-edges test fixtures.

Deferred follow-up (tracked here so it doesn't get lost):
- TODO(integration): write tests/test_inferred_edges_pipeline.py — full
  end-to-end test exercising ingest -> normalizer enqueues
  inferred_edges_queue row -> side worker dequeues, builds bundle,
  calls extractor -> INFERRED edges visible in graph_edges. Requires
  docker-compose Postgres and either a fake LLM client or a recorded
  cassette so it runs in CI without an API key. Each stage has unit
  coverage today; the full path is verified by manual smoke test on
  first staging deploy until the integration test lands.
"""

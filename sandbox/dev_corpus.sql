-- Tiny, agent-VISIBLE dev corpus for IMPLEMENT mode (entrypoint.sh precedence #2).
-- Purpose: give the agent's /retrieve a few BM25-able rows so it can iterate and the
-- smoke test exercises a real (non-empty) path. This is NOT the eval corpus — the
-- held-out grade corpus (real gemini-embedding-2 vectors, Mahit's content) is injected
-- separately at grade time as /grade/corpus.sql.gz and is never visible to the agent.
--
-- embedding_v2 is left NULL on purpose: hand-writing 3072-dim halfvecs is meaningless,
-- and the vector channel simply returns nothing for these rows. BM25 (content_tsv,
-- generated) carries the signal — enough for the agent to see /retrieve return docs.
--
-- Tenant key = customer_id = the slug 'eval-tenant' (data-plane convention). The smoke
-- test and the grader both send X-Prbe-Customer: eval-tenant. A mismatch here would
-- silently zero recall, so keep these three literals in lockstep.

INSERT INTO customers (customer_id, display_name, api_key_hash, status)
VALUES ('eval-tenant', 'Eval Tenant', 'dev-corpus-not-a-real-key', 'active')
ON CONFLICT (customer_id) DO NOTHING;

INSERT INTO documents
    (doc_id, version, customer_id, source_system, source_id, source_url,
     doc_type, content_hash, title, body_preview,
     created_at, updated_at, valid_from, acl)
VALUES
    ('slack:C100:1000', 1, 'eval-tenant', 'slack', 'C100:1000',
     'https://example.test/slack/C100/1000', 'message', 'devhash-1',
     'Rotating the LiteLLM master key',
     'How we rotate the LiteLLM master key and resync k8s secrets.',
     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '{}'),
    ('github:pr:100', 1, 'eval-tenant', 'github', 'pr:100',
     'https://example.test/github/pr/100', 'pull_request', 'devhash-2',
     'Add pgvector HNSW index to chunks',
     'PR adding the HNSW index on chunks.embedding_v2 for vector recall.',
     '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z', '{}')
ON CONFLICT DO NOTHING;

INSERT INTO chunks
    (chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
     token_count, first_seen_version, last_seen_version)
VALUES
    ('slack:C100:1000#0', 'slack:C100:1000', 'eval-tenant', 0,
     'To rotate the LiteLLM master key, update litellm.env and run the k8s secrets sync script, then restart the gateway.',
     'devhash-1', 24, 1, 1),
    ('github:pr:100#0', 'github:pr:100', 'eval-tenant', 0,
     'This PR adds a pgvector HNSW index over chunks.embedding_v2 (halfvec cosine) to speed up vector retrieval recall.',
     'devhash-2', 22, 1, 1),
    ('github:pr:100#1', 'github:pr:100', 'eval-tenant', 1,
     'Benchmarks show recall at 10 improves once the HNSW index replaces the brute-force scan over embeddings.',
     'devhash-3', 20, 1, 1)
ON CONFLICT DO NOTHING;

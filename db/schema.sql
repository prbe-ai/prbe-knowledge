-- prbe-knowledge Phase 0 schema.
-- Canonical reference. Alembic's initial migration executes this file verbatim.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Apache AGE was evaluated and is not available on Neon Scale tier.
-- Graph is modeled as relational tables (graph_nodes + graph_edges) below,
-- with RLS for tenant isolation.

-- ---------------------------------------------------------------------------
-- customers: tenant registry (parent of all customer-scoped tables)
-- ---------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id          TEXT PRIMARY KEY,
    display_name         TEXT NOT NULL,
    api_key_hash         TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'active',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata             JSONB NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- customer_source_mapping: resolve an incoming webhook's source-side
-- workspace/team/org id to the owning customer.
-- Populated at OAuth install time via Connector.identify_workspaces().
-- ---------------------------------------------------------------------------
CREATE TABLE customer_source_mapping (
    source_system   TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    external_name   TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_system, external_id)
);
CREATE INDEX idx_customer_source_mapping_customer
    ON customer_source_mapping (customer_id, source_system);

-- ---------------------------------------------------------------------------
-- documents: canonical normalized form, one row per version.
-- Temporal columns:
--   valid_from         — when this version became the live version
--   valid_to           — when it stopped being live (NULL = still live)
--   supersedes_doc_id  — chain pointer to the version that replaced it
--   deleted_at         — source-side deletion tombstone (no chunks should be live)
-- Full body content lives in chunks.content (inline). No documents.body column.
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
    doc_id               TEXT NOT NULL,
    version              INT  NOT NULL,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,

    source_system        TEXT NOT NULL,
    source_id            TEXT NOT NULL,
    source_url           TEXT NOT NULL,

    doc_class            TEXT NOT NULL DEFAULT 'raw_source',
    doc_type             TEXT NOT NULL,
    content_type         TEXT NOT NULL DEFAULT 'text/plain',
    language             TEXT,

    content_hash         TEXT NOT NULL,
    title                TEXT,
    body_preview         TEXT,
    body_size_bytes      INT  NOT NULL DEFAULT 0,
    body_token_count     INT  NOT NULL DEFAULT 0,
    author_id            TEXT,

    created_at           TIMESTAMPTZ NOT NULL,
    updated_at           TIMESTAMPTZ NOT NULL,
    valid_from           TIMESTAMPTZ NOT NULL,
    valid_to             TIMESTAMPTZ,
    deleted_at           TIMESTAMPTZ,
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    parent_doc_id        TEXT,
    supersedes_doc_id    TEXT,

    acl                  JSONB NOT NULL,
    metadata             JSONB NOT NULL DEFAULT '{}',
    entities             JSONB NOT NULL DEFAULT '[]',
    attachments          JSONB NOT NULL DEFAULT '[]',
    doc_references       JSONB NOT NULL DEFAULT '[]',

    ingestion_event_id   BIGINT,
    normalizer_version   TEXT NOT NULL DEFAULT 'v1',

    compiled_from_doc_ids TEXT[] DEFAULT NULL,
    compilation_model    TEXT DEFAULT NULL,
    compiled_at          TIMESTAMPTZ DEFAULT NULL,
    compile_trigger      TEXT DEFAULT NULL,

    PRIMARY KEY (doc_id, version)
);

CREATE INDEX idx_documents_customer_source ON documents (customer_id, source_system, source_id);
CREATE INDEX idx_documents_customer_updated ON documents (customer_id, updated_at DESC);
CREATE INDEX idx_documents_customer_class ON documents (customer_id, doc_class, doc_type);
-- Fast "latest version" lookup per (customer_id, doc_id).
CREATE INDEX idx_documents_live ON documents (customer_id, doc_id) WHERE valid_to IS NULL;
CREATE INDEX idx_documents_fts_title_preview ON documents
    USING GIN (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body_preview,'')));
CREATE INDEX idx_documents_entities ON documents USING GIN (entities jsonb_path_ops);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata jsonb_path_ops);

-- ---------------------------------------------------------------------------
-- chunks: content-addressable retrieval units.
-- Identity is (doc_id, content_hash) — a chunk with the same content across
-- doc versions is ONE row with its temporal validity extended, not N rows.
-- That keeps embedding cost bounded on doc edits (only added content is re-embedded).
--
-- Temporal columns:
--   valid_from          — when this chunk first appeared in any version of the doc
--   valid_to            — when it stopped being in the live version (NULL = still live)
--   first_seen_version  — document version that first introduced this chunk
--   last_seen_version   — most recent document version that still contained it
-- ---------------------------------------------------------------------------
CREATE TABLE chunks (
    chunk_id             TEXT PRIMARY KEY,
    doc_id               TEXT NOT NULL,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,

    chunk_index          INT  NOT NULL,
    content              TEXT NOT NULL,
    content_hash         TEXT NOT NULL,
    token_count          INT  NOT NULL,

    embedding            halfvec(3072) NOT NULL,
    embedding_model      TEXT NOT NULL DEFAULT 'openai/text-embedding-3-large',
    embedding_dim        INT  NOT NULL DEFAULT 3072,
    chunker_version      TEXT NOT NULL DEFAULT 'naive-v1',

    first_seen_version   INT  NOT NULL,
    last_seen_version    INT  NOT NULL,
    valid_from           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to             TIMESTAMPTZ,

    metadata             JSONB NOT NULL DEFAULT '{}',

    UNIQUE (doc_id, content_hash)
    -- No FK to documents(doc_id, version). A chunk can span multiple versions
    -- (first_seen_version..last_seen_version), so pinning the FK to a specific
    -- version would cascade-delete live chunks if an old doc version ever
    -- gets hand-deleted by a retention job. The customer_id CASCADE above
    -- handles the tenant-delete path, which is the only real delete in
    -- normal operation.
);

-- halfvec_cosine_ops: pgvector HNSW indexes halfvec up to 4000 dims.
CREATE INDEX idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX idx_chunks_customer       ON chunks (customer_id);
CREATE INDEX idx_chunks_doc            ON chunks (doc_id);
CREATE INDEX idx_chunks_doc_live       ON chunks (doc_id) WHERE valid_to IS NULL;
CREATE INDEX idx_chunks_doc_hash       ON chunks (doc_id, content_hash);
CREATE INDEX idx_chunks_fts_content    ON chunks USING GIN (to_tsvector('english', content));

-- ---------------------------------------------------------------------------
-- acl_snapshots: temporal ACL truth.
-- Phase 0 INGESTS + MAINTAINS this. Phase 0 does NOT enforce at query time.
-- Phase 1 flips enforcement on with no backfill required.
-- ---------------------------------------------------------------------------
CREATE TABLE acl_snapshots (
    snapshot_id          BIGSERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system        TEXT NOT NULL,

    principal_type       TEXT NOT NULL,
    principal_id         TEXT NOT NULL,

    resource_type        TEXT NOT NULL,
    resource_id          TEXT NOT NULL,

    permission           TEXT NOT NULL,
    valid_from           TIMESTAMPTZ NOT NULL,
    valid_to             TIMESTAMPTZ,
    metadata             JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_acl_principal ON acl_snapshots (customer_id, principal_id, valid_from DESC);
CREATE INDEX idx_acl_resource ON acl_snapshots (customer_id, resource_type, resource_id, valid_from DESC);

-- ---------------------------------------------------------------------------
-- ingestion_queue: backpressure buffer between webhook handler and worker.
-- Fast path inserts here and returns 200. Worker drains with SKIP LOCKED.
-- ---------------------------------------------------------------------------
CREATE TABLE ingestion_queue (
    queue_id             BIGSERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system        TEXT NOT NULL,
    source_event_id      TEXT NOT NULL,
    payload_s3_key       TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending',
    attempts             INT  NOT NULL DEFAULT 0,
    error                TEXT,
    enqueued_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at           TIMESTAMPTZ,
    heartbeat_at         TIMESTAMPTZ,
    completed_at         TIMESTAMPTZ,
    UNIQUE (customer_id, source_system, source_event_id)
);
CREATE INDEX idx_queue_pending ON ingestion_queue (status, enqueued_at) WHERE status = 'pending';
CREATE INDEX idx_queue_processing ON ingestion_queue (status, heartbeat_at) WHERE status = 'processing';
CREATE INDEX idx_queue_customer_status ON ingestion_queue (customer_id, status, enqueued_at);

-- ---------------------------------------------------------------------------
-- backfill_state: pagination cursor per (customer, source). Resumable.
-- ---------------------------------------------------------------------------
CREATE TABLE backfill_state (
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system        TEXT NOT NULL,
    last_cursor          TEXT,
    status               TEXT NOT NULL DEFAULT 'idle',
    last_error           TEXT,
    events_enqueued      INT  NOT NULL DEFAULT 0,
    started_at           TIMESTAMPTZ,
    heartbeat_at         TIMESTAMPTZ,
    last_progress_at     TIMESTAMPTZ,
    completed_at         TIMESTAMPTZ,
    PRIMARY KEY (customer_id, source_system)
);
CREATE INDEX idx_backfill_state_pending ON backfill_state (status, started_at)
    WHERE status = 'pending';
CREATE INDEX idx_backfill_state_running ON backfill_state (status, heartbeat_at)
    WHERE status = 'running';

-- ---------------------------------------------------------------------------
-- integration_tokens: per-customer per-source OAuth credentials.
-- ---------------------------------------------------------------------------
CREATE TABLE integration_tokens (
    customer_id              TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system            TEXT NOT NULL,
    access_token_encrypted   TEXT NOT NULL,
    refresh_token_encrypted  TEXT,
    expires_at               TIMESTAMPTZ,
    scope                    TEXT,
    webhook_secret           TEXT,
    status                   TEXT NOT NULL DEFAULT 'active',
    last_refresh_at          TIMESTAMPTZ,
    last_refresh_error       TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (customer_id, source_system)
);
CREATE INDEX idx_tokens_refresh_errors ON integration_tokens (status, last_refresh_error)
    WHERE last_refresh_error IS NOT NULL;

-- ---------------------------------------------------------------------------
-- failed_chunks: audit of embedding batch rejects.
-- ---------------------------------------------------------------------------
CREATE TABLE failed_chunks (
    failed_chunk_id      BIGSERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    doc_id               TEXT NOT NULL,
    doc_version          INT  NOT NULL,
    chunk_index          INT  NOT NULL,
    content_preview      TEXT,
    error                TEXT NOT NULL,
    failed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_failed_chunks_customer ON failed_chunks (customer_id, failed_at DESC);

-- ---------------------------------------------------------------------------
-- ingestion_events: replay / debug log
-- ---------------------------------------------------------------------------
CREATE TABLE ingestion_events (
    event_id             BIGSERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system        TEXT NOT NULL,
    event_type           TEXT NOT NULL,
    source_event_id      TEXT,
    payload_s3_key       TEXT NOT NULL,
    status               TEXT NOT NULL,
    retry_count          INT  NOT NULL DEFAULT 0,
    error                TEXT,
    doc_ids_produced     TEXT[],
    received_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at         TIMESTAMPTZ,
    normalizer_version   TEXT,
    UNIQUE (customer_id, source_system, source_event_id)
);

CREATE INDEX idx_events_customer_source_status ON ingestion_events (customer_id, source_system, status);
CREATE INDEX idx_events_customer_received ON ingestion_events (customer_id, received_at DESC);

-- ---------------------------------------------------------------------------
-- audit_log: append-only per-tenant (for enterprise audit in Phase 2+)
-- ---------------------------------------------------------------------------
CREATE TABLE audit_log (
    audit_id             BIGSERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    actor_id             TEXT NOT NULL,
    action               TEXT NOT NULL,
    resource_type        TEXT,
    resource_id          TEXT,
    details              JSONB NOT NULL DEFAULT '{}',
    occurred_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_audit_customer ON audit_log (customer_id, occurred_at DESC);

-- ---------------------------------------------------------------------------
-- Graph: relational model with RLS tenant isolation.
-- ---------------------------------------------------------------------------
CREATE TABLE graph_nodes (
    node_id       BIGSERIAL PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label         TEXT NOT NULL,
    canonical_id  TEXT NOT NULL,
    properties    JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (customer_id, label, canonical_id)
);

CREATE INDEX idx_graph_nodes_customer_label ON graph_nodes (customer_id, label);
CREATE INDEX idx_graph_nodes_props ON graph_nodes USING GIN (properties jsonb_path_ops);

CREATE TABLE graph_edges (
    edge_id       BIGSERIAL PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    edge_type     TEXT NOT NULL,
    from_node_id  BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    to_node_id    BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    properties    JSONB NOT NULL DEFAULT '{}',
    valid_from    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to      TIMESTAMPTZ,
    UNIQUE (customer_id, edge_type, from_node_id, to_node_id)
);

CREATE INDEX idx_graph_edges_customer_type ON graph_edges (customer_id, edge_type);
CREATE INDEX idx_graph_edges_from ON graph_edges (customer_id, from_node_id, edge_type);
CREATE INDEX idx_graph_edges_to ON graph_edges (customer_id, to_node_id, edge_type);

-- RLS: tenant isolation enforced at the DB level.
-- Application sets `SET app.current_customer_id = '<id>'` at the start of each tx.
-- FORCE is required so the policy applies to the table owner too.
ALTER TABLE graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY;
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON graph_nodes
    USING (customer_id = current_setting('app.current_customer_id', true));

CREATE POLICY tenant_isolation ON graph_edges
    USING (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- query_cache: router (Haiku) output cache keyed by query hash. 1-hour TTL.
-- ---------------------------------------------------------------------------
CREATE TABLE query_cache (
    cache_key            TEXT PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    query_text_hash      TEXT NOT NULL,
    entities             JSONB NOT NULL,
    expansions           JSONB NOT NULL,
    expires_at           TIMESTAMPTZ NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_query_cache_expiry ON query_cache (expires_at);
CREATE INDEX idx_query_cache_customer ON query_cache (customer_id, query_text_hash);

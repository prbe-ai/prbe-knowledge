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
--
-- Bridges to Neon Auth (Better Auth) Organization plugin via organization_id:
--   * Each team-managed tenant maps 1:1 to a neon_auth.organization row
--   * NULL organization_id is permitted for legacy admin-key-managed tenants
--     pre-Phase-9 migration; new tenants always have one
--   * ON DELETE RESTRICT — the dashboard soft-deletes via status='deleted'
--     and an offline reaper handles hard-delete; Better Auth's
--     organization.delete is blocked while a customer references the org
-- status:
--   'active'   — normal operation
--   'deleted'  — soft-deleted; service layer filters all reads/writes
-- ---------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id          TEXT PRIMARY KEY,
    display_name         TEXT NOT NULL,
    api_key_hash         TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'active',
    organization_id      UUID REFERENCES neon_auth.organization(id) ON DELETE RESTRICT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata             JSONB NOT NULL DEFAULT '{}'
);

-- One customer per organization (where the link is set).
CREATE UNIQUE INDEX customers_organization_id_unique
    ON customers (organization_id)
    WHERE organization_id IS NOT NULL;

-- Hot path filter: dashboard + retrieval skip soft-deleted tenants.
CREATE INDEX idx_customers_active
    ON customers (customer_id)
    WHERE status = 'active';

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

    ingestion_event_id   BIGINT,  -- FK added at bottom (ingestion_events is defined later)
    normalizer_version   TEXT NOT NULL DEFAULT 'v1',

    compiled_from_doc_ids TEXT[] DEFAULT NULL,
    compilation_model    TEXT DEFAULT NULL,
    compiled_at          TIMESTAMPTZ DEFAULT NULL,
    compile_trigger      TEXT DEFAULT NULL,

    -- PK includes customer_id so tenants ingesting the same source identity
    -- (e.g. the same Slack workspace replayed under a different customer)
    -- don't collide on doc_id and silently drop writes via ON CONFLICT.
    PRIMARY KEY (customer_id, doc_id, version)
);

CREATE INDEX idx_documents_customer_source ON documents (customer_id, source_system, source_id);
CREATE INDEX idx_documents_customer_updated ON documents (customer_id, updated_at DESC);
CREATE INDEX idx_documents_customer_class ON documents (customer_id, doc_class, doc_type);
-- Composite + partial for the deterministic list pipeline (and aggregates).
-- Matches: WHERE customer_id=? AND source_system=? AND doc_type=? AND valid_to IS NULL
-- ORDER BY updated_at DESC.
CREATE INDEX idx_documents_customer_source_doctype_updated
    ON documents (customer_id, source_system, doc_type, updated_at DESC)
    WHERE valid_to IS NULL;
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
    chunk_id             TEXT NOT NULL,
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
    -- 'content' = body chunk (default for all rows pre-0018).
    -- 'metadata' = synthetic per-document chunk holding title/repo/author/url
    --   text, generated at ingestion. Embedded + FTS-indexed for the search
    --   path to rank metadata-keyed queries; the list path's representative
    --   chunk filters to kind='content' so list responses always show body.
    kind                 TEXT NOT NULL DEFAULT 'content',

    -- PK includes customer_id so tenants ingesting overlapping source content
    -- can't collide on chunk_id (which is derived from doc_id + content_hash).
    PRIMARY KEY (customer_id, chunk_id),
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
-- One metadata chunk per doc; partial index serves backfill idempotency check.
CREATE INDEX idx_chunks_metadata_kind  ON chunks (customer_id, doc_id) WHERE kind = 'metadata';

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
    metadata             JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT acl_snapshots_assertion_unique UNIQUE (
        customer_id, source_system,
        principal_type, principal_id,
        resource_type, resource_id,
        permission, valid_from
    )
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
    -- Legacy single-payload column. Kept dormant after migration 0026 so
    -- old in-flight rows during the cutover deploy don't crash; new code
    -- only reads payload_s3_keys. A follow-up PR drops payload_s3_key
    -- once enough deploy cycles have passed.
    payload_s3_key       TEXT,
    -- Coalesced array of every R2 path written for this row. For most
    -- connectors this is a single-element array; for claude_code, every
    -- batch for the same session_id appends here via _enqueue's UPSERT.
    payload_s3_keys      TEXT[] NOT NULL DEFAULT '{}',
    status               TEXT NOT NULL DEFAULT 'pending',
    attempts             INT  NOT NULL DEFAULT 0,
    error                TEXT,
    priority             SMALLINT NOT NULL DEFAULT 100,
    -- Monotonic counter, bumped on every UPSERT into the row. Worker
    -- captures it on claim and CAS-commits on it, so any batch landing
    -- mid-Phase-A triggers a clean re-claim with the extended array.
    version              INT NOT NULL DEFAULT 0,
    enqueued_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at           TIMESTAMPTZ,
    heartbeat_at         TIMESTAMPTZ,
    completed_at         TIMESTAMPTZ,
    UNIQUE (customer_id, source_system, source_event_id)
);
CREATE INDEX idx_queue_pending_priority ON ingestion_queue (priority DESC, enqueued_at) WHERE status = 'pending';
CREATE INDEX idx_queue_processing ON ingestion_queue (status, heartbeat_at) WHERE status = 'processing';
CREATE INDEX idx_queue_customer_status ON ingestion_queue (customer_id, status, enqueued_at);

-- ---------------------------------------------------------------------------
-- manual_uploads: dashboard-originated file upload audit and cleanup state.
--
-- Original bytes are staged in R2, text is extracted into a raw payload and
-- queued like any other source, then the worker deletes the staged original
-- after documents/chunks are persisted successfully.
-- ---------------------------------------------------------------------------
CREATE TABLE manual_uploads (
    upload_id           TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    filename            TEXT NOT NULL,
    content_type        TEXT NOT NULL DEFAULT 'application/octet-stream',
    file_size_bytes     BIGINT NOT NULL DEFAULT 0,
    file_sha256         TEXT NOT NULL,
    staging_object_key  TEXT,
    payload_object_key  TEXT,
    uploaded_by         TEXT,
    uploaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status              TEXT NOT NULL,
    parse_engine        TEXT,
    parse_error         TEXT,
    extracted_chars     INT NOT NULL DEFAULT 0,
    doc_id              TEXT,
    indexed_at          TIMESTAMPTZ,
    original_deleted_at TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT manual_uploads_status_check CHECK (
        status IN ('queued', 'indexed', 'failed_parse', 'failed_ingest')
    )
);
CREATE INDEX idx_manual_uploads_customer_uploaded
    ON manual_uploads (customer_id, uploaded_at DESC);
CREATE INDEX idx_manual_uploads_customer_status
    ON manual_uploads (customer_id, status, uploaded_at DESC);
CREATE INDEX idx_manual_uploads_doc
    ON manual_uploads (customer_id, doc_id)
    WHERE doc_id IS NOT NULL;

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
-- integration_tokens: per-customer per-source credentials.
--
-- For non-device sources (slack/linear/github/notion/granola), one row per
-- (customer, source) — enforced by the partial unique index where device_id
-- IS NULL. Device-scoped sources (claude_code) can have many rows per
-- (customer, source), keyed by device_id.
-- ---------------------------------------------------------------------------
CREATE TABLE integration_tokens (
    token_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    device_id                TEXT,
    device_metadata          JSONB,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX integration_tokens_unique_per_source
    ON integration_tokens (customer_id, source_system)
    WHERE device_id IS NULL;
CREATE UNIQUE INDEX integration_tokens_unique_per_device
    ON integration_tokens (customer_id, source_system, device_id)
    WHERE device_id IS NOT NULL;
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
-- Functional indexes for the list pipeline's loose-match entity filter.
-- Equality arms (= canonical_id, = properties->>'name') hit these; the
-- suffix-LIKE arm accepts seq-scan-of-subset (graph_nodes filtered by
-- (customer_id, label) is small).
CREATE INDEX idx_graph_nodes_lower_canonical ON graph_nodes (customer_id, label, LOWER(canonical_id));
CREATE INDEX idx_graph_nodes_lower_props_name ON graph_nodes (customer_id, label, LOWER(properties ->> 'name'));
-- Alphanumeric-normalized variants for the regex_replace match arms in
-- _entity_match_clause (PR #18). Strip non-[a-z0-9] before comparing so
-- "external investigations" ↔ "external-investigations" hits the same
-- index path as the LOWER() variants above.
CREATE INDEX idx_graph_nodes_alnum_canonical
    ON graph_nodes (customer_id, label, regexp_replace(LOWER(canonical_id), '[^a-z0-9]+', '', 'g'));
CREATE INDEX idx_graph_nodes_alnum_props_name
    ON graph_nodes (customer_id, label, regexp_replace(LOWER(properties ->> 'name'), '[^a-z0-9]+', '', 'g'));

CREATE TABLE graph_edges (
    edge_id       BIGSERIAL PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    edge_type     TEXT NOT NULL,
    from_node_id  BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    to_node_id    BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    properties    JSONB NOT NULL DEFAULT '{}',
    valid_from    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to      TIMESTAMPTZ,
    source_system TEXT,
    UNIQUE (customer_id, edge_type, from_node_id, to_node_id)
);

CREATE INDEX idx_graph_edges_customer_type ON graph_edges (customer_id, edge_type);
CREATE INDEX idx_graph_edges_from ON graph_edges (customer_id, from_node_id, edge_type);
CREATE INDEX idx_graph_edges_to ON graph_edges (customer_id, to_node_id, edge_type);

-- Per-node provenance: which source system(s) asserted this node. A node
-- touched by multiple connectors must survive disconnection of any single
-- one; this table is the join target for that cleanup logic.
CREATE TABLE graph_node_provenance (
    node_id        BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    customer_id    TEXT   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system  TEXT   NOT NULL,
    first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_id, source_system)
);

CREATE INDEX idx_provenance_customer_source
    ON graph_node_provenance (customer_id, source_system);

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
-- usage_events: per-tenant audit trail of /retrieve, /query, /sources calls.
-- Written from a post-response BackgroundTask in services/retrieval/middleware.py;
-- read by the dashboard's /query/usage page via /usage/feed, /usage/stats,
-- /usage/search. RLS-isolated like graph_nodes / graph_edges.
-- ---------------------------------------------------------------------------
CREATE TABLE usage_events (
    event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    caller_kind     TEXT NOT NULL,
    caller_subject  TEXT,
    event_type      TEXT NOT NULL,
    request_id      UUID,
    endpoint        TEXT NOT NULL,
    summary         TEXT,
    status          TEXT NOT NULL,
    error_class     TEXT,
    latency_ms      INT,
    result_count    INT,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_usage_events_customer_time
    ON usage_events (customer_id, occurred_at DESC);
CREATE INDEX idx_usage_events_customer_type_time
    ON usage_events (customer_id, event_type, occurred_at DESC);
-- 'simple' (not 'english') so user search terms aren't stemmed/stop-worded.
CREATE INDEX idx_usage_events_search
    ON usage_events USING gin (to_tsvector('simple', summary));

ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_events FORCE ROW LEVEL SECURITY;

CREATE POLICY usage_events_tenant_isolation ON usage_events
    USING (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- query_traces: full request/response payload log per retrieval call.
-- Sister table to usage_events. usage_events stores thin metrics; this
-- stores the parsed request body and response body so we can evaluate
-- retrieval effectiveness — zero-result rate, score distributions,
-- retrieve->get_source click-through, etc. Written from the same
-- middleware BackgroundTask chain. response_truncated is a separate
-- boolean (not a JSONB sentinel) so consumers can distinguish a stub
-- row from a real response that happens to contain a `_truncated` key.
-- request_id is plain BTREE (NOT UNIQUE) — clients may supply
-- X-Request-Id and a UNIQUE constraint would silently drop legitimate
-- retries that we'd want to study.
-- ---------------------------------------------------------------------------
CREATE TABLE query_traces (
    trace_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id          UUID NOT NULL,
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    occurred_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type          TEXT NOT NULL,
    schema_version      SMALLINT NOT NULL DEFAULT 1,
    request             JSONB NOT NULL,
    response            JSONB NOT NULL,
    response_size_bytes INT NOT NULL,
    response_truncated  BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_query_traces_customer_time
    ON query_traces (customer_id, occurred_at DESC);
CREATE INDEX idx_query_traces_request_id
    ON query_traces (request_id);

ALTER TABLE query_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_traces FORCE ROW LEVEL SECURITY;

CREATE POLICY query_traces_tenant_isolation ON query_traces
    USING (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- Late-bound FKs: targets defined later in this file than their source tables.
-- ---------------------------------------------------------------------------
ALTER TABLE documents
    ADD CONSTRAINT documents_ingestion_event_id_fkey
    FOREIGN KEY (ingestion_event_id)
    REFERENCES ingestion_events(event_id)
    ON DELETE SET NULL;

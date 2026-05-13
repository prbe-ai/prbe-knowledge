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
    metadata             JSONB NOT NULL DEFAULT '{}',
    -- Per-tenant feature toggles (added by migration 0023). Read by
    -- shared.customer_prefs for the wiki-generation gate. Schema-on-read
    -- bool keys; missing keys resolve to False on every reader.
    preferences          JSONB NOT NULL DEFAULT '{}'
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
-- Trigram GIN for the id_lookup retriever's leading-wildcard LIKE arms
-- (`source_id LIKE '%:<id>'`, `doc_id LIKE '%:<id>'`). Btree can't help
-- here; without these the planner seq-scans documents filtered only by
-- customer_id. See migration 0055.
CREATE INDEX idx_documents_source_id_trgm ON documents USING GIN (source_id gin_trgm_ops);
CREATE INDEX idx_documents_doc_id_trgm ON documents USING GIN (doc_id gin_trgm_ops);

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
    -- Stage 1 of the Gemini embedding migration: every newly-ingested
    -- chunk also gets a gemini-embedding-2-preview vector written here
    -- alongside the OpenAI vector above. NULLABLE -- a Gemini API outage
    -- during dual-write leaves these NULL for affected rows; the Stage 2
    -- backfill sweeps them up. The query path keeps reading `embedding`
    -- until Stage 4 cuts over.
    embedding_v2         halfvec(3072) NULL,
    embedding_v2_model   TEXT NULL,
    embedding_v2_dim     INT  NULL,
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

    -- Materialized to_tsvector so BM25 (`ts_rank_cd` + bitmap recheck) reads
    -- the precomputed lexeme array instead of re-tokenizing `content` on
    -- every candidate row. See migration 0062 + services/retrieval/retrievers/
    -- bm25.py for the perf rationale.
    content_tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

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
-- Stage 3 of the Gemini embedding migration: parallel HNSW over the
-- gemini-embedding-2-preview vectors, used by the query path after the
-- Stage 4 cutover. Defaults (m=16, ef_construction=64) match the v1 index
-- above so any retrieval tuning translates 1:1; written without explicit
-- WITH (...) for symmetry with the v1 line.
CREATE INDEX idx_chunks_embedding_v2_hnsw ON chunks USING hnsw (embedding_v2 halfvec_cosine_ops);
CREATE INDEX idx_chunks_customer       ON chunks (customer_id);
CREATE INDEX idx_chunks_doc            ON chunks (doc_id);
CREATE INDEX idx_chunks_doc_live       ON chunks (doc_id) WHERE valid_to IS NULL;
CREATE INDEX idx_chunks_doc_hash       ON chunks (doc_id, content_hash);
-- TEMPORARY: scheduled for removal in the contract-phase migration that
-- follows 0062. Kept during the EXPAND window so old retrieval pods
-- running pre-0062 binaries (which BM25 against `to_tsvector('english',
-- content)`) still hit a real index during the rolling deploy. Once the
-- new code is fully rolled out, the cleanup PR drops this.
CREATE INDEX idx_chunks_fts_content    ON chunks USING GIN (to_tsvector('english', content));
-- New BM25 index over the stored content_tsv column (migration 0062).
-- Becomes the sole BM25 index after the contract-phase cleanup PR drops
-- the expression-based one above.
CREATE INDEX idx_chunks_content_tsv    ON chunks USING GIN (content_tsv);
-- One metadata chunk per doc; partial index serves backfill idempotency check.
CREATE INDEX idx_chunks_metadata_kind  ON chunks (customer_id, doc_id) WHERE kind = 'metadata';

-- ---------------------------------------------------------------------------
-- directed_vectors: per-document trigger phrases used as a doc-level
-- retrieval booster. Engineer-pinned (source='human') phrases are authored
-- via wiki frontmatter `directed:` blocks; LLM-generated (source='llm')
-- phrases come from synthesis. The retriever (services/retrieval/retrievers/
-- directed.py) HNSW-searches `embedding` and reports one hit per matched
-- doc; fusion folds the cosine-distance signal into the doc score
-- (services/retrieval/fusion.py). Phrase text NEVER reaches the agent —
-- only the owning doc's content chunks are returned.
--
-- No FK to documents — same rationale chunks uses (PK includes version,
-- but a directed_vector is doc-level not version-level). Tenant cascade
-- flows through customer_id REFERENCES customers ON DELETE CASCADE.
-- ---------------------------------------------------------------------------
CREATE TABLE directed_vectors (
    vector_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    doc_id           TEXT NOT NULL,
    embedding        halfvec(3072) NOT NULL,
    source_text      TEXT NOT NULL,
    source           TEXT NOT NULL,
    -- LLM-generated rows carry the run that produced them; older runs'
    -- rows are deleted on regen. Human pins set this to NULL.
    synthesis_run_id BIGINT NULL,
    content_hash     BYTEA NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_dv_source CHECK (source IN ('human','llm')),
    CONSTRAINT ck_dv_run_for_llm CHECK (
        (source = 'llm'   AND synthesis_run_id IS NOT NULL) OR
        (source = 'human' AND synthesis_run_id IS NULL)
    ),
    CONSTRAINT uq_dv_doc_hash UNIQUE (customer_id, doc_id, content_hash)
);

CREATE INDEX idx_directed_vectors_embedding_hnsw
    ON directed_vectors USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX idx_directed_vectors_customer_doc
    ON directed_vectors (customer_id, doc_id);

ALTER TABLE directed_vectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE directed_vectors FORCE ROW LEVEL SECURITY;
CREATE POLICY directed_vectors_tenant_isolation ON directed_vectors
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

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
-- code_repo_state: per-(customer, repo, file) extraction cache for the
-- code_graph connector. Push events short-circuit on content_hash match so
-- steady-state pushes do zero re-embedding. Survives across worker restarts.
-- ---------------------------------------------------------------------------
CREATE TABLE code_repo_state (
    customer_id            TEXT NOT NULL,
    repo                   TEXT NOT NULL,
    file_path              TEXT NOT NULL,
    content_hash           TEXT NOT NULL,
    language               TEXT NOT NULL,
    symbol_count           INT  NOT NULL DEFAULT 0,
    last_extracted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_extractor_version TEXT NOT NULL,
    PRIMARY KEY (customer_id, repo, file_path)
);
ALTER TABLE code_repo_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_repo_state FORCE ROW LEVEL SECURITY;
CREATE POLICY code_repo_state_tenant_isolation ON code_repo_state
    USING (customer_id = current_setting('app.current_customer_id', true));

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
    -- Lane A (surprise-score, migration 0054_graph_node_degree_community):
    -- materialized degree maintained on edge insert/delete in
    -- graph_writer.upsert_edges and cross_repo_deps deletes; community_id
    -- populated by the nightly Leiden cron in services/community/leiden.py.
    -- Both feed the surprise_score() boost in the graph retriever
    -- (services/retrieval/surprise.py).
    degree        INT NOT NULL DEFAULT 0,
    community_id  INT,
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
-- Lane A: partial index on community_id for cross-community surprise-score lookups.
CREATE INDEX idx_graph_nodes_customer_community
    ON graph_nodes (customer_id, community_id) WHERE community_id IS NOT NULL;

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
    -- Three tiers: 'EXTRACTED' (deterministic AST), 'INFERRED' (PR-B
    -- proposer/promoter), 'AMBIGUOUS' (unresolved call sites awaiting
    -- promotion). Retrieval defaults to dropping AMBIGUOUS.
    confidence    TEXT NOT NULL DEFAULT 'EXTRACTED'
        CONSTRAINT graph_edges_confidence_check
        CHECK (confidence IN ('EXTRACTED', 'INFERRED', 'AMBIGUOUS')),
    -- Lane B (inferred-edges, migration 0055_inferred_edge_metadata):
    -- provenance for LLM-inferred edges. Both NULL for deterministically-
    -- extracted edges (back-compat with all existing call sites). When set,
    -- identify which prompt version produced this edge and when, so future
    -- prompt v2 can DELETE rows with extractor_id = 'inferred_edges:v1' and
    -- re-extract via the backfill script.
    extractor_id  TEXT,
    extracted_at  TIMESTAMPTZ,
    UNIQUE (customer_id, edge_type, from_node_id, to_node_id)
);

CREATE INDEX idx_graph_edges_customer_type ON graph_edges (customer_id, edge_type);
CREATE INDEX idx_graph_edges_from ON graph_edges (customer_id, from_node_id, edge_type);
CREATE INDEX idx_graph_edges_to ON graph_edges (customer_id, to_node_id, edge_type);
CREATE INDEX idx_graph_edges_confidence
    ON graph_edges (customer_id, edge_type, confidence);
-- Lane B: partial index for prompt-version invalidation queries.
CREATE INDEX idx_graph_edges_customer_extractor
    ON graph_edges (customer_id, extractor_id) WHERE extractor_id IS NOT NULL;

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
-- uploaded_at / counters / ix_usage_events_pending: outbox shape for the
-- data-plane telemetry uploader (migration 0065, option A — one table).
-- uploaded_at NULL = "needs flushing"; counters holds token/usage counts
-- ({} until a follow-up threads real counts); the partial index is the
-- uploader's drain query.
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
    metadata        JSONB NOT NULL DEFAULT '{}',
    uploaded_at     TIMESTAMPTZ,
    counters        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_usage_events_customer_time
    ON usage_events (customer_id, occurred_at DESC);
CREATE INDEX idx_usage_events_customer_type_time
    ON usage_events (customer_id, event_type, occurred_at DESC);
-- 'simple' (not 'english') so user search terms aren't stemmed/stop-worded.
CREATE INDEX idx_usage_events_search
    ON usage_events USING gin (to_tsvector('simple', summary));
CREATE INDEX ix_usage_events_pending
    ON usage_events (customer_id, occurred_at)
    WHERE uploaded_at IS NULL;

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
-- wiki_synthesis_queue / wiki_synthesis_runs
--
-- Internal queue tables for the LLM-Wiki synthesis cron (services/synthesis/).
-- `Normalizer._persist` enqueues one row per persisted document; the cron
-- drains them, runs Haiku triage + Sonnet synthesis, and writes wiki pages
-- via build_normalization_result.
--
-- Convention for internal queue tables (matches ingestion_queue,
-- backfill_state): NO row-level security. Tenant scoping for per-customer
-- operations is enforced by application code (`with_tenant(customer_id)`
-- + explicit WHERE customer_id = $1). The cron's cross-customer
-- `SELECT DISTINCT customer_id` in `_tick` would silently return zero
-- rows under FORCE RLS without a tenant GUC; see migration
-- 20260503_0034_wiki_synthesis_no_rls.py for the rationale.
-- ---------------------------------------------------------------------------
CREATE TABLE wiki_synthesis_queue (
    queue_id                BIGSERIAL PRIMARY KEY,
    customer_id             TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    doc_id                  TEXT NOT NULL,
    doc_version             INT  NOT NULL,
    source_system           TEXT NOT NULL,
    doc_type                TEXT NOT NULL,
    enqueued_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Source-side timestamp (Slack ts, GitHub created_at, Linear updatedAt,
    -- Granola startedAt, Notion last_edited_time, fallback documents.created_at).
    -- Populated by Normalizer at insert. The wiki agent reads triaged events
    -- ordered by source_ts ASC to walk the day in time order.
    source_ts               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status                  TEXT NOT NULL DEFAULT 'pending',
    triage_score            REAL,
    triage_error            TEXT,
    triage_completed_at     TIMESTAMPTZ,
    attempts                INT NOT NULL DEFAULT 0,
    synthesis_run_id        BIGINT,
    synthesis_completed_at  TIMESTAMPTZ,
    synthesis_error         TEXT,
    -- Stamped at claim time. ReclaimLoop sweeps stale rows back to the
    -- prior state if attempts < cap, else to terminal 'failed' so ops
    -- can investigate via the dashboard. See migration 0038.
    heartbeat_at            TIMESTAMPTZ,
    -- DLQ surface for unrecoverable failures: triage batch crash, agent
    -- halt (turn cap, stall, compactor failure, Gemini outage). Admin
    -- reset (POST .../dlq/reset) flips rows back to pending or triaged.
    dlq_reason              TEXT,
    dlq_at                  TIMESTAMPTZ,
    CONSTRAINT uq_wsq_customer_doc_version UNIQUE (customer_id, doc_id, doc_version),
    CONSTRAINT ck_wsq_status CHECK (status IN (
        'pending','triaging','triaged','rejected',
        'synthesizing','done','failed',
        'synthesis_skipped','dlq'
    ))
);

CREATE INDEX idx_wsq_drain
    ON wiki_synthesis_queue (customer_id, status, enqueued_at);

-- Cursor index for the wiki agent's next_events() pagination. The
-- agent reads triaged events ordered by source_ts ASC, queue_id ASC,
-- skipping rows already applied or skipped in the current run.
CREATE INDEX ix_wsq_drain_cursor
    ON wiki_synthesis_queue (customer_id, status, source_ts, queue_id);

-- Reclaim only ever scans rows in 'triaging'/'synthesizing' — partial
-- index keeps the sweep cheap as the queue's done/rejected tail grows.
CREATE INDEX idx_wsq_heartbeat_reclaim
    ON wiki_synthesis_queue (heartbeat_at)
    WHERE status IN ('triaging', 'synthesizing');

CREATE TABLE wiki_synthesis_runs (
    run_id          BIGSERIAL PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    -- Discriminator: which worker wrote this run row. Triage and
    -- synthesis each open their own run per drain; the status endpoint
    -- filters stage='synthesis' for `last_run_pages_*`.
    stage           TEXT NOT NULL DEFAULT 'synthesis',
    -- Per-source discriminator for bootstrap runs ('slack', 'github',
    -- 'linear', etc.). NULL for daily-replay (kind='wake'/'scheduled').
    source          TEXT,
    -- Phase 2 fan-out target. NULL for Phase 1 rows (one per source per
    -- trigger). Phase 2 rows carry a target like 'owner/repo' (GitHub),
    -- 'channel_id' (Slack), etc. The orchestrator's post-Phase-1 hook
    -- inserts these by querying the source's BackfillFanout discoverer.
    target          TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    events_total    INT NOT NULL DEFAULT 0,
    events_triaged  INT NOT NULL DEFAULT 0,
    events_kept     INT NOT NULL DEFAULT 0,
    pages_updated   INT NOT NULL DEFAULT 0,
    pages_created   INT NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'running',
    error           TEXT,
    CONSTRAINT ck_wsr_kind CHECK (kind IN ('onboarding','wake','scheduled','bootstrap')),
    CONSTRAINT ck_wsr_stage CHECK (stage IN ('triage','synthesis')),
    CONSTRAINT ck_wsr_status CHECK (status IN ('pending','running','complete','failed','partial','cancelled'))
);

CREATE INDEX idx_wsr_customer
    ON wiki_synthesis_runs (customer_id, started_at DESC);

CREATE INDEX idx_wsr_stage_started
    ON wiki_synthesis_runs (customer_id, stage, started_at DESC);

CREATE INDEX idx_wsr_kind_source
    ON wiki_synthesis_runs (customer_id, kind, source, started_at DESC);

CREATE INDEX idx_wsr_kind_source_target
    ON wiki_synthesis_runs (customer_id, kind, source, target, started_at DESC);

-- ---------------------------------------------------------------------------
-- wiki_links / wiki_timeline_entries / wiki_raw_data
--
-- Bootstrap-era extensions for the wiki page graph. All three tables share
-- the wiki_synthesis_queue precedent (migration 0034): NO row-level security.
-- Tenant scoping is application-enforced (explicit WHERE customer_id = $1).
-- See migration 20260506_0043_wiki_bootstrap_schema.py for rationale.
-- ---------------------------------------------------------------------------
CREATE TABLE wiki_links (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    src_wiki_type   TEXT NOT NULL,
    src_slug        TEXT NOT NULL,
    dst_wiki_type   TEXT NOT NULL,
    dst_slug        TEXT NOT NULL,
    -- Optional relation verb extracted from `[[type:slug|verb]]` markdown
    -- syntax or a frontmatter field name. Empty string when the link is a
    -- bare `[[type:slug]]` mention.
    link_type       TEXT NOT NULL DEFAULT '',
    -- ~80 chars surrounding the link site in the source markdown. Useful
    -- for backlink rendering ("mentioned in: '... [[person:X|works_at]]
    -- the auth migration ...'").
    context         TEXT NOT NULL DEFAULT '',
    -- Where the link came from. 'markdown' = body inline; 'frontmatter' =
    -- YAML field; 'manual' = admin / migration-set.
    link_source     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_wiki_links_source CHECK (link_source IN ('markdown','frontmatter','manual')),
    -- The parser slices a ~80-char window for `context`. Cap of 200 leaves
    -- 2.5x headroom for multibyte characters and future window adjustments
    -- without a re-migration. Pure "don't misuse this field" guard rail —
    -- context is not in the unique key, so this isn't btree protection.
    CONSTRAINT ck_wiki_links_context_len CHECK (length(context) <= 200),
    CONSTRAINT uq_wiki_links UNIQUE NULLS NOT DISTINCT
        (customer_id, src_wiki_type, src_slug,
         dst_wiki_type, dst_slug, link_type, link_source)
);

CREATE INDEX ix_wiki_links_from
    ON wiki_links (customer_id, src_wiki_type, src_slug);

CREATE INDEX ix_wiki_links_to
    ON wiki_links (customer_id, dst_wiki_type, dst_slug);

CREATE TABLE wiki_timeline_entries (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    wiki_type       TEXT NOT NULL,
    slug            TEXT NOT NULL,
    -- Day-bucket the source event falls under (typically the source
    -- system's own timestamp). The dashboard renders the timeline
    -- grouped by entry_date DESC.
    entry_date      DATE NOT NULL,
    source          TEXT NOT NULL,
    -- One-line headline for the timeline UI. Capped at 1000 chars because
    -- this column participates in `uq_wiki_timeline_dedup`, and Postgres
    -- btree keys are limited to ~2704 bytes per row. The other columns
    -- in the unique total ~150 bytes; 1000 leaves comfortable headroom
    -- and is well above any sensible "one-line audit entry" length.
    -- Long-form expansion goes in `detail` (uncapped, not in the unique).
    summary         TEXT NOT NULL,
    detail          TEXT NOT NULL DEFAULT '',
    -- Optional source-side ref (Slack thread_ts, GitHub PR number, ...).
    -- Lets the dashboard deep-link from a timeline entry to its source.
    source_ref      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_wiki_timeline_summary_len CHECK (length(summary) <= 1000),
    CONSTRAINT uq_wiki_timeline_dedup UNIQUE
        (customer_id, wiki_type, slug, entry_date, summary)
);

CREATE INDEX ix_wiki_timeline_page
    ON wiki_timeline_entries (customer_id, wiki_type, slug, entry_date DESC);

CREATE INDEX ix_wiki_timeline_date
    ON wiki_timeline_entries (customer_id, entry_date DESC);

CREATE TABLE wiki_raw_data (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    wiki_type       TEXT NOT NULL,
    slug            TEXT NOT NULL,
    source          TEXT NOT NULL,
    -- Source-system identifier (Slack thread_ts, GitHub PR id, Linear
    -- issue id, ...). Together with (customer_id, wiki_type, slug,
    -- source) this is the dedup key — re-bootstrap of the same source
    -- doesn't duplicate raw rows.
    source_ref      TEXT NOT NULL,
    data            JSONB NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_wiki_raw_data UNIQUE (customer_id, wiki_type, slug, source, source_ref)
);

CREATE INDEX ix_wiki_raw_data_page
    ON wiki_raw_data (customer_id, wiki_type, slug, fetched_at DESC);

CREATE INDEX ix_wiki_raw_data_source
    ON wiki_raw_data (customer_id, source, source_ref);

-- ---------------------------------------------------------------------------
-- Custom Ingest Tokens (migration 0046)
-- Self-serve bearer tokens for the Custom Ingest API. Customers mint a
-- token from the dashboard; that token authenticates writes to the
-- Custom Ingest endpoint without dragging the user through full OAuth.
-- ---------------------------------------------------------------------------
CREATE TABLE custom_ingest_tokens (
    token_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    token_hash          TEXT NOT NULL UNIQUE,
    token_prefix        TEXT NOT NULL,
    created_by_user_id  TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at        TIMESTAMPTZ,
    revoked_at          TIMESTAMPTZ
);

CREATE INDEX ix_custom_ingest_tokens_customer_active
    ON custom_ingest_tokens (customer_id, revoked_at);

-- RLS enabled (not FORCE'd) so the SECURITY DEFINER verifier path
-- bypasses cleanly via owner privileges. Matches the integration_tokens
-- convention.
ALTER TABLE custom_ingest_tokens ENABLE ROW LEVEL SECURITY;

CREATE POLICY custom_ingest_tokens_tenant_isolation ON custom_ingest_tokens
    FOR ALL
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- SECURITY DEFINER lookup-and-touch. Runs as the function OWNER (the
-- migration role); because the table is ENABLE'd but not FORCE'd, the
-- owner is naturally exempt from RLS — the verifier path can't know
-- the tenant until *after* the lookup. Throttles last_used_at to one
-- update per 5 minutes to keep verification cheap on hot paths.
CREATE OR REPLACE FUNCTION verify_and_touch_custom_ingest_token(p_token_hash text)
RETURNS TABLE(token_id uuid, customer_id text)
-- search_path: ag_catalog FIRST — prbe-knowledge tables live there (AGE
-- extension hijack at migrate time prepended ag_catalog to search_path
-- during CREATE EXTENSION age, so `custom_ingest_tokens` actually resides
-- in ag_catalog, not public). The original `SET search_path = public`
-- (migration 0046) made the function body's `UPDATE custom_ingest_tokens`
-- raise UndefinedTableError; see migration 0066_fix_custom_ingest_search_path.
LANGUAGE plpgsql SECURITY DEFINER SET search_path = ag_catalog, "$user", public AS $$
BEGIN
    RETURN QUERY
    UPDATE custom_ingest_tokens
       SET last_used_at = now()
     WHERE token_hash = p_token_hash
       AND revoked_at IS NULL
       AND (last_used_at IS NULL OR last_used_at < now() - interval '5 minutes')
     RETURNING custom_ingest_tokens.token_id, custom_ingest_tokens.customer_id;
    IF NOT FOUND THEN
        RETURN QUERY
            SELECT t.token_id, t.customer_id
              FROM custom_ingest_tokens t
             WHERE t.token_hash = p_token_hash
               AND t.revoked_at IS NULL;
    END IF;
END $$;

REVOKE ALL ON FUNCTION verify_and_touch_custom_ingest_token(text) FROM PUBLIC;

-- ---------------------------------------------------------------------------
-- inferred_edges_queue (Lane B, migration 0055_inferred_edge_metadata):
-- side-queue worker drains this. One row per (customer, anchor_doc_id) is
-- enqueued by the main worker after successful normalize+write. The
-- side-worker claims via FOR UPDATE SKIP LOCKED, builds a bundle of related
-- content, calls the LLM extractor, and upserts INFERRED/AMBIGUOUS edges
-- into graph_edges (stamped with extractor_id + extracted_at).
-- ---------------------------------------------------------------------------
CREATE TABLE inferred_edges_queue (
    id                      BIGSERIAL PRIMARY KEY,
    customer_id             TEXT NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
    anchor_doc_id           TEXT NOT NULL,
    enqueued_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_started_at   TIMESTAMPTZ,
    processing_worker_id    TEXT,
    attempts                INT NOT NULL DEFAULT 0,
    extractor_id            TEXT NOT NULL,
    done_at                 TIMESTAMPTZ,
    error                   TEXT
);

-- Partial index: drain queries filter on pending rows only. As done/failed
-- rows accumulate the tail stays outside this index.
CREATE INDEX idx_inferred_edges_queue_pending
    ON inferred_edges_queue (enqueued_at)
    WHERE processing_started_at IS NULL AND done_at IS NULL;

ALTER TABLE inferred_edges_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE inferred_edges_queue FORCE ROW LEVEL SECURITY;
CREATE POLICY inferred_edges_queue_tenant_isolation
    ON inferred_edges_queue
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- mcp_oauth_* — OAuth 2.1 provider tables for the MCP server.
--
-- prbe-backend acts as the OAuth issuer for prbe-knowledge-mcp; customer AI
-- agents (Claude Desktop, Cursor, etc.) register dynamically via RFC 7591
-- and present issued JWTs to the MCP endpoint. The session is the
-- persistent identity for a grant — refresh tokens are rotating tickets
-- within a session.
--
-- Sources: db/migrations/versions/20260425_0009_mcp_oauth.py and
--          db/migrations/versions/20260429_0027_mcp_oauth_sessions.py
-- (kept in sync here so a fresh-DB provision via schema.sql + stamp head
-- lands these tables instead of leaving the dashboard's /mcp/connections
-- 500-ing on UndefinedTableError.)
--
-- No RLS: these are global per-customer rows scoped by user_id /
-- customer_id columns and accessed only by the issuer code path.
-- ---------------------------------------------------------------------------

CREATE TABLE mcp_oauth_clients (
    client_id                  TEXT PRIMARY KEY,
    client_name                TEXT NOT NULL,
    redirect_uris              TEXT[] NOT NULL,
    grant_types                TEXT[] NOT NULL
                               DEFAULT ARRAY['authorization_code','refresh_token'],
    response_types             TEXT[] NOT NULL DEFAULT ARRAY['code'],
    token_endpoint_auth_method TEXT NOT NULL DEFAULT 'none',
    software_id                TEXT,
    software_version           TEXT,
    scope                      TEXT NOT NULL DEFAULT 'mcp:read',
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE mcp_oauth_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id       TEXT NOT NULL
                    REFERENCES mcp_oauth_clients(client_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    customer_id     TEXT NOT NULL
                    REFERENCES customers(customer_id) ON DELETE CASCADE,
    scope           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at      TIMESTAMPTZ
);
CREATE INDEX mcp_oauth_sessions_user
    ON mcp_oauth_sessions(user_id, customer_id)
    WHERE revoked_at IS NULL;

CREATE TABLE mcp_oauth_codes (
    code                  TEXT PRIMARY KEY,
    client_id             TEXT NOT NULL
                          REFERENCES mcp_oauth_clients(client_id) ON DELETE CASCADE,
    user_id               TEXT NOT NULL,
    customer_id           TEXT NOT NULL
                          REFERENCES customers(customer_id) ON DELETE CASCADE,
    redirect_uri          TEXT NOT NULL,
    code_challenge        TEXT NOT NULL,
    code_challenge_method TEXT NOT NULL,
    scope                 TEXT NOT NULL,
    issued_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at            TIMESTAMPTZ NOT NULL,
    used_at               TIMESTAMPTZ
);
CREATE INDEX mcp_oauth_codes_expires_at ON mcp_oauth_codes(expires_at);

CREATE TABLE mcp_oauth_refresh_tokens (
    token_id     TEXT PRIMARY KEY,
    client_id    TEXT NOT NULL
                 REFERENCES mcp_oauth_clients(client_id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL,
    customer_id  TEXT NOT NULL
                 REFERENCES customers(customer_id) ON DELETE CASCADE,
    scope        TEXT NOT NULL,
    issued_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL,
    revoked_at   TIMESTAMPTZ,
    session_id   UUID NOT NULL
                 REFERENCES mcp_oauth_sessions(id) ON DELETE CASCADE
);
CREATE INDEX mcp_oauth_refresh_tokens_user
    ON mcp_oauth_refresh_tokens(user_id, customer_id)
    WHERE revoked_at IS NULL;
CREATE INDEX mcp_oauth_refresh_tokens_session_active
    ON mcp_oauth_refresh_tokens(session_id, issued_at DESC)
    WHERE revoked_at IS NULL;

-- ---------------------------------------------------------------------------
-- Late-bound FKs: targets defined later in this file than their source tables.
-- ---------------------------------------------------------------------------
ALTER TABLE documents
    ADD CONSTRAINT documents_ingestion_event_id_fkey
    FOREIGN KEY (ingestion_event_id)
    REFERENCES ingestion_events(event_id)
    ON DELETE SET NULL;

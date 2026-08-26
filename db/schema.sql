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
    -- shared.customer_prefs for per-feature gating. Schema-on-read
    -- bool keys; missing keys resolve to False on every reader.
    preferences          JSONB NOT NULL DEFAULT '{}',
    -- Per-tenant R2 bucket name. Added by migration 0073, locked NOT NULL
    -- by 0075. Every INSERT path (the CP→DP mirror, self-host's
    -- seed-customer Helm job) writes this; the runtime never falls back
    -- to a computed value.
    r2_bucket            TEXT NOT NULL
);

-- BEFORE INSERT trigger that fills r2_bucket from customer_id when NULL
-- (Postgres DEFAULTs can't reference other columns). Production callers
-- supply r2_bucket explicitly via the CP→DP mirror; the trigger only
-- fires for test fixtures and self-host installs that forget to set it.
-- On the DP, ``customer_id`` IS the slug (see prbe-backend's
-- apps/control_plane/routers/me/provision.py), so ``prbe-<customer_id>``
-- == the canonical new-policy bucket name.
CREATE OR REPLACE FUNCTION customers_fill_r2_bucket() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.r2_bucket IS NULL THEN
        NEW.r2_bucket := 'prbe-' || NEW.customer_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER customers_fill_r2_bucket_trg
    BEFORE INSERT ON customers
    FOR EACH ROW
    EXECUTE FUNCTION customers_fill_r2_bucket();

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
-- system_settings: global (non-tenant) operational config, keyed by name.
-- Introduced by alembic 0025; folded into schema.sql here so a fresh DB (built
-- from this file + stamped to head) carries it. The `ingestion_killswitch` row
-- is read on every webhook/ingest; a missing table makes the reader fail OPEN
-- (ingestion enabled), so the drift was silent until a truly-fresh deploy.
-- ---------------------------------------------------------------------------
CREATE TABLE system_settings (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    description TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by  TEXT
);
INSERT INTO system_settings (key, value, description)
VALUES (
    'ingestion_killswitch',
    '{"enabled": true, "reason": null}'::jsonb,
    'Master switch for all plugin ingestion. Set value->>enabled to false to halt webhooks globally.'
)
ON CONFLICT (key) DO NOTHING;

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

    -- Materialized, weighted tsvector over the title (migration 0099).
    -- setweight 'A' against chunks.content_tsv's unweighted 'D' is what gave
    -- the old ts_rank_cd ranker its ~10x title advantage for free.
    -- Indexed TWICE over: verbatim, and with path/filename punctuation
    -- flattened to spaces, because Postgres' `english` parser emits
    -- `model.ckpt` as ONE `file` lexeme while query tokenizers split on
    -- alphanumeric runs -- a verbatim-only index misses exactly the filename
    -- case this exists for.
    title_tsv            tsvector GENERATED ALWAYS AS (
                             setweight(
                                 to_tsvector(
                                     'english',
                                     coalesce(title, '') || ' ' ||
                                     translate(coalesce(title, ''), './\-_:', '      ')
                                 ),
                                 'A'
                             )
                         ) STORED,
    -- The title+body_preview vector, MATERIALIZED rather than left as an
    -- expression index. Under FORCE RLS the planner will not use an expression
    -- index (matching one means evaluating the expression before the security
    -- qual), so grounding's fts arm rebuilt this tsvector for every live
    -- document in the tenant -- 1052 ms/probe. A stored column is an ordinary
    -- column reference and indexes normally. See migration 0109.
    title_preview_tsv    tsvector GENERATED ALWAYS AS (
                             to_tsvector(
                                 'english'::regconfig,
                                 coalesce(title, '') || ' ' ||
                                 coalesce(body_preview, '')
                             )
                         ) STORED,
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

    -- migration 0082 (post-approval draft gating for generated artifacts).
    -- New generated-artifact writes set 'draft'; the review approve path
    -- flips to 'approved' atomically. Existing rows backfill to 'approved'.
    visibility           TEXT NOT NULL DEFAULT 'approved',

    -- PK includes customer_id so tenants ingesting the same source identity
    -- (e.g. the same Slack workspace replayed under a different customer)
    -- don't collide on doc_id and silently drop writes via ON CONFLICT.
    PRIMARY KEY (customer_id, doc_id, version),
    CONSTRAINT documents_visibility_chk CHECK (visibility IN ('draft','approved'))
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
-- Reaching a doc's CHILDREN. Retiring a session's superseded units filters on
-- parent_doc_id, which idx_documents_live cannot satisfy (its leading column
-- after customer_id is doc_id), so without this the retire scans every live
-- document for the tenant. See migration 0105.
CREATE INDEX idx_documents_parent_live ON documents (customer_id, parent_doc_id) WHERE valid_to IS NULL;
CREATE INDEX idx_documents_title_preview_tsv ON documents USING GIN (title_preview_tsv);
CREATE INDEX idx_documents_entities ON documents USING GIN (entities jsonb_path_ops);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata jsonb_path_ops);
-- Trigram GIN for the id_lookup retriever's leading-wildcard LIKE arms
-- (`source_id LIKE '%:<id>'`, `doc_id LIKE '%:<id>'`). Btree can't help
-- here; without these the planner seq-scans documents filtered only by
-- customer_id. See migration 0055.
CREATE INDEX idx_documents_source_id_trgm ON documents USING GIN (source_id gin_trgm_ops);
CREATE INDEX idx_documents_doc_id_trgm ON documents USING GIN (doc_id gin_trgm_ops);

-- Trigram index on the document TITLE (migration 0089). grounding.py's
-- doc_title subtask ORs a trgm predicate against an FTS one, and an OR is only
-- as indexable as its worst branch -- without this the whole predicate falls to
-- a scan. This index is the reason schema.sql drift matters: it was added by a
-- migration, never backported here, and every database born fresh afterwards
-- silently lacked it while reporting alembic head (measured: 677 ms on the
-- plane missing it vs 77 ms on the plane that had it).
CREATE INDEX idx_documents_title_trgm ON documents USING GIN (title gin_trgm_ops)
    WHERE valid_to IS NULL;

-- GIN over the weighted title tsvector (migration 0099).
CREATE INDEX idx_documents_title_tsv ON documents USING GIN (title_tsv);
-- Partial index keeps the doc-type listing path from scanning draft rows
-- once visibility='draft' artifacts start appearing. See migration 0082.
CREATE INDEX IF NOT EXISTS documents_visibility_approved_idx
    ON documents (customer_id, doc_type) WHERE visibility = 'approved';

-- Covers both aggregates behind the /knowledge stats header (migration 0111):
-- the per-source document count/MAX(ingested_at) by full partial-index scan,
-- and the live-document side of the chunk count by (customer_id, doc_id)
-- prefix. Every column those two read is here, so both go index-only and the
-- 10,494-page heap scan the count used to do disappears.
CREATE INDEX IF NOT EXISTS idx_documents_stats_live
    ON documents (customer_id, doc_id, source_system, ingested_at)
    WHERE valid_to IS NULL AND deleted_at IS NULL;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

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

    -- Legacy OpenAI vector. NULLABLE post-cutover (2026-05-14, mig 0071):
    -- new rows leave this NULL and only populate embedding_v2 below. Kept
    -- so the eval harness can still read pre-cutover OpenAI baselines for
    -- apples-to-apples comparisons. No production code path reads this.
    embedding            halfvec(3072) NULL,
    embedding_model      TEXT NULL,
    embedding_dim        INT  NULL,
    -- Production embedding column (gemini-embedding-2). Every newly-ingested
    -- chunk's vector is written here. NULLABLE only so embed-failure paths
    -- can record a chunk for later backfill without blocking the txn.
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

    -- Denormalized copy of the owning document's title (migration 0100).
    -- NOT a convenience: BM25 matches titles, and a cross-table
    -- `chunks.content OR documents.title` predicate cannot be served by any
    -- single index -- it seq-scanned 4.4 GB and timed out at 30s in
    -- production. With the title on the chunk the whole query is single-table
    -- and pg_search answers it from one index (measured 191 ms).
    -- Kept in sync by two triggers below, not by application code.
    title                TEXT NOT NULL DEFAULT '',

    -- migration 0082 (post-approval draft gating for generated artifacts).
    -- Tracks the visibility of the chunk's owning document version so retrieval
    -- can default-filter draft chunks without joining documents.
    visibility           TEXT NOT NULL DEFAULT 'approved',

    -- PK includes customer_id so tenants ingesting overlapping source content
    -- can't collide on chunk_id (which is derived from doc_id + content_hash).
    PRIMARY KEY (customer_id, chunk_id),
    UNIQUE (doc_id, content_hash),
    CONSTRAINT chunks_visibility_chk CHECK (visibility IN ('draft','approved'))
    -- No FK to documents(doc_id, version). A chunk can span multiple versions
    -- (first_seen_version..last_seen_version), so pinning the FK to a specific
    -- version would cascade-delete live chunks if an old doc version ever
    -- gets hand-deleted by a retention job. The customer_id CASCADE above
    -- handles the tenant-delete path, which is the only real delete in
    -- normal operation.
);

-- halfvec_cosine_ops: pgvector HNSW indexes halfvec up to 4000 dims.
-- Production retrieval index over gemini-embedding-2 vectors. The legacy
-- v1 HNSW index over `embedding` was dropped in migration 0071 (no
-- production reader after the cutover).
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
-- Live chunks for one tenant (migration 0111). idx_chunks_doc_live is partial
-- on valid_to but carries no customer_id, and idx_chunks_customer carries
-- customer_id but every version -- so the stats count used to BitmapAnd the two
-- and read 148,808 buffers. This one is correct on both axes.
CREATE INDEX IF NOT EXISTS idx_chunks_stats_live
    ON chunks (customer_id, doc_id) WHERE valid_to IS NULL;

-- Single-column uniqueness on chunk_id (migration 0101). chunk_id is already
-- unique in practice (`{doc_id}:{prefix}{content_hash[:16]}`); this enforces
-- it. Was also a pg_search key_field requirement before 0.23.4 relaxed it.
CREATE UNIQUE INDEX chunks_chunk_id_unique ON chunks (chunk_id);

-- pg_search BM25 index (migration 0100). Guarded because the extension ships
-- in `prbe-postgres` but NOT in the `pgvector/pgvector` image used for local
-- dev, and schema.sql has to bootstrap both. Without the guard every fresh
-- local database fails here; with it, local gets a working schema minus BM25
-- and the retrieval tests skip loudly (see tests/retrieval/conftest.py).
--
-- pg_search permits exactly ONE `USING bm25` index per relation, so this is
-- the only one -- adding a second raises
-- "a relation may only have one `USING bm25` index".
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search') THEN
        CREATE EXTENSION IF NOT EXISTS pg_search;
        EXECUTE $ix$
            CREATE INDEX IF NOT EXISTS idx_chunks_bm25_v2
            ON chunks USING bm25 (
                chunk_id, content, title, customer_id, doc_id, kind,
                chunk_index, first_seen_version, last_seen_version, visibility
            )
            WITH (
                key_field=chunk_id,
                -- source_code, not the default: the default splits on `/` but
                -- NOT on `.`, so `checkpoints/model.ckpt` indexes as
                -- {checkpoints, model.ckpt} and a search for `model` or `ckpt`
                -- finds nothing. Same case migration 0099 handles on the
                -- tsvector side by storing a punctuation-flattened copy.
                text_fields='{"title": {"tokenizer": {"type": "source_code"}}}'
            )
        $ix$;
    END IF;
END
$$;

-- Title sync (migration 0100). The obligation a denormalized column takes on.
-- Enforced in the database because the application is not the only writer:
-- backfill scripts, migrations and manual SQL all insert chunks.
CREATE OR REPLACE FUNCTION chunks_sync_title_from_document()
RETURNS trigger AS $fn$
BEGIN
    UPDATE chunks c
    SET title = coalesce(NEW.title, '')
    WHERE c.customer_id = NEW.customer_id
      AND c.doc_id = NEW.doc_id
      AND NEW.version BETWEEN c.first_seen_version AND c.last_seen_version
      AND c.title IS DISTINCT FROM coalesce(NEW.title, '');
    RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION chunks_fill_title_on_insert()
RETURNS trigger AS $fn$
BEGIN
    IF NEW.title IS NULL OR NEW.title = '' THEN
        SELECT coalesce(d.title, '') INTO NEW.title
        FROM documents d
        WHERE d.doc_id = NEW.doc_id
          AND d.customer_id = NEW.customer_id
          AND d.version BETWEEN NEW.first_seen_version AND NEW.last_seen_version
        ORDER BY d.version DESC
        LIMIT 1;
        NEW.title := coalesce(NEW.title, '');
    END IF;
    RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_sync_title ON documents;
CREATE TRIGGER trg_chunks_sync_title
    AFTER UPDATE OF title ON documents
    FOR EACH ROW
    WHEN (OLD.title IS DISTINCT FROM NEW.title)
    EXECUTE FUNCTION chunks_sync_title_from_document();

DROP TRIGGER IF EXISTS trg_chunks_fill_title ON chunks;
CREATE TRIGGER trg_chunks_fill_title
    BEFORE INSERT ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_fill_title_on_insert();
-- Partial index keeps retrieval's per-doc chunk fetch index-only once
-- visibility='draft' rows start appearing (post-approval generated
-- artifacts). See migration 0082.
CREATE INDEX IF NOT EXISTS chunks_visibility_approved_idx
    ON chunks (customer_id, doc_id) WHERE visibility = 'approved';

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
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
    customer_id            TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
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
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

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

ALTER TABLE failed_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE failed_chunks FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON failed_chunks
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

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
    -- Migration 0082: per-entity embedding for AutoMergeAnalyzer vector
    -- candidate gen. Same dim as chunks.embedding_v2 so the existing
    -- GeminiEmbedder writes in-place. Nullable initially; backfill via
    -- scripts/backfill_graph_node_embeddings.py.
    embedding     halfvec(3072) NULL,
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
-- Migration 0082: HNSW + trigram indexes for AutoMergeAnalyzer candidate gen.
CREATE INDEX idx_graph_nodes_embedding_hnsw
    ON graph_nodes USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX idx_graph_nodes_canonical_id_trgm
    ON graph_nodes USING gin (LOWER(canonical_id) gin_trgm_ops);
CREATE INDEX idx_graph_nodes_name_trgm
    ON graph_nodes USING gin (LOWER(properties->>'name') gin_trgm_ops);
-- Lane A: partial index on community_id for cross-community surprise-score lookups.
CREATE INDEX idx_graph_nodes_customer_community
    ON graph_nodes (customer_id, community_id) WHERE community_id IS NOT NULL;
-- Migration 0091: partial functional indexes for resolve_to_person_canonical_ids.
-- The retrieval path looks up Person nodes whose Lane E-enriched properties
-- carry alternate identifiers (e.g. claude_code better-auth uuid as employee_id
-- on a Slack-rooted Person row). Without these indexes, the per-tenant Person
-- table is seq-scanned on every retrieval call once tenants pass a few hundred
-- Persons. Partial WHERE keeps the indexes small for nodes that don't carry
-- the enrichment property.
CREATE INDEX idx_graph_nodes_person_employee_id
    ON graph_nodes ((properties->>'employee_id'))
    WHERE label = 'Person' AND properties->>'employee_id' IS NOT NULL;
CREATE INDEX idx_graph_nodes_person_login
    ON graph_nodes ((properties->>'login'))
    WHERE label = 'Person' AND properties->>'login' IS NOT NULL;
CREATE INDEX idx_graph_nodes_person_email
    ON graph_nodes ((properties->>'email'))
    WHERE label = 'Person' AND properties->>'email' IS NOT NULL;

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
    -- Provenance for alias-resolved or merge-rewritten edges. NULL when
    -- the edge has never been touched by alias resolution. Populated by
    -- graph_writer at ingest (when the inbound canonical_id was an alias)
    -- and by the merge transaction (when an alias node's edge was rewritten
    -- to point at the primary).
    aliased_from_canonical_id TEXT,
    aliased_to_canonical_id   TEXT
);

CREATE INDEX idx_graph_edges_customer_type ON graph_edges (customer_id, edge_type);
CREATE INDEX idx_graph_edges_from ON graph_edges (customer_id, from_node_id, edge_type);
CREATE INDEX idx_graph_edges_to ON graph_edges (customer_id, to_node_id, edge_type);
CREATE INDEX idx_graph_edges_confidence
    ON graph_edges (customer_id, edge_type, confidence);
-- Lane B: partial index for prompt-version invalidation queries.
CREATE INDEX idx_graph_edges_customer_extractor
    ON graph_edges (customer_id, extractor_id) WHERE extractor_id IS NOT NULL;
-- Composite UNIQUE keyed by (edge_type, from, to, alias_from, alias_to).
-- Different alias lanes coexist as distinct rows; common-case "both
-- aliased_from cols NULL" inserts still dedup (COALESCE-to-empty-string
-- collides). graph_writer.upsert_edges' ON CONFLICT references this
-- index by name.
CREATE UNIQUE INDEX graph_edges_unique_lane ON graph_edges (
    customer_id, edge_type, from_node_id, to_node_id,
    COALESCE(aliased_from_canonical_id, ''),
    COALESCE(aliased_to_canonical_id, '')
);

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
ALTER TABLE graph_node_provenance ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_node_provenance FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON graph_nodes
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

CREATE POLICY tenant_isolation ON graph_edges
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

CREATE POLICY tenant_isolation ON graph_node_provenance
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- Entity clusters: manual identity merging via dashboard (migration 0071).
-- Physical merge (B-promote) -- alias edges rewritten, alias nodes
-- hard-deleted. See docs/superpowers/specs/2026-05-13-entity-clusters-design.md.
-- ---------------------------------------------------------------------------
CREATE TABLE entity_merge_audit (
    merge_id                    UUID PRIMARY KEY,
    customer_id                 TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                       TEXT NOT NULL,
    primary_canonical_id        TEXT NOT NULL,
    merged_alias_canonical_ids  TEXT[] NOT NULL,
    performed_by_user_id        UUID NOT NULL,
    performed_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason                      TEXT NULL,
    status                      TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'reversed'))
);
CREATE INDEX idx_entity_merge_audit_primary
    ON entity_merge_audit (customer_id, label, primary_canonical_id);

CREATE TABLE entity_merge_node_snapshot (
    merge_id      UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label         TEXT NOT NULL,
    canonical_id  TEXT NOT NULL,
    properties    JSONB NOT NULL,
    degree        INT  NOT NULL,
    community_id  INT  NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    provenance    JSONB NOT NULL,
    PRIMARY KEY (merge_id, label, canonical_id)
);

CREATE TABLE entity_merge_edge_snapshot (
    merge_id                       UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    snapshot_seq                   INT  NOT NULL,
    customer_id                    TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    -- 'deleted_self_loop'      — both endpoints collapsed onto the primary;
    --                             unmerge restores it as ($node, $node).
    -- 'deleted_duplicate_lane'  — the rewritten edge collided with an edge the
    --                             primary already held in this alias's lane
    --                             (graph_edges_unique_lane). Keeps its two
    --                             distinct endpoints, so unmerge restores it by
    --                             resolving pre_from/pre_to canonical ids.
    --                             Added by migration 0117 — the writer at
    --                             entity_clusters_routes.py:363 predates it and
    --                             had never executed, so nothing ever violated
    --                             the narrower form.
    operation                      TEXT NOT NULL
                                   CHECK (operation IN ('deleted_self_loop',
                                                        'deleted_duplicate_lane')),
    pre_edge_type                  TEXT NOT NULL,
    pre_from_canonical_id          TEXT NOT NULL,
    pre_from_label                 TEXT NOT NULL,
    pre_to_canonical_id            TEXT NOT NULL,
    pre_to_label                   TEXT NOT NULL,
    pre_properties                 JSONB NOT NULL,
    pre_confidence                 TEXT NOT NULL,
    pre_valid_from                 TIMESTAMPTZ NOT NULL,
    pre_valid_to                   TIMESTAMPTZ NULL,
    pre_source_system              TEXT NULL,
    pre_extractor_id               TEXT NULL,
    pre_extracted_at               TIMESTAMPTZ NULL,
    pre_aliased_from_canonical_id  TEXT NULL,
    pre_aliased_to_canonical_id    TEXT NULL,
    PRIMARY KEY (merge_id, snapshot_seq)
);
CREATE INDEX idx_entity_merge_edge_snapshot_merge
    ON entity_merge_edge_snapshot (merge_id);

CREATE TABLE entity_aliases (
    customer_id           TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                 TEXT NOT NULL,
    alias_canonical_id    TEXT NOT NULL,
    primary_canonical_id  TEXT NOT NULL,
    merge_id              UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    added_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (customer_id, label, alias_canonical_id),
    CONSTRAINT entity_aliases_not_self CHECK (alias_canonical_id <> primary_canonical_id)
);
CREATE INDEX idx_entity_aliases_primary
    ON entity_aliases (customer_id, label, primary_canonical_id);
CREATE INDEX idx_entity_aliases_merge
    ON entity_aliases (merge_id);

CREATE TABLE entity_cluster_metadata (
    customer_id                  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                        TEXT NOT NULL,
    primary_canonical_id         TEXT NOT NULL,
    display_name                 TEXT NOT NULL,
    display_name_last_edited_by  UUID NULL,
    display_name_last_edited_at  TIMESTAMPTZ NULL,
    PRIMARY KEY (customer_id, label, primary_canonical_id)
);

ALTER TABLE entity_merge_audit         ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_audit         FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_merge_node_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_node_snapshot FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_merge_edge_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_edge_snapshot FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_aliases             ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_aliases             FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_cluster_metadata    ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_cluster_metadata    FORCE  ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON entity_merge_audit
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_merge_node_snapshot
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_merge_edge_snapshot
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_aliases
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_cluster_metadata
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));

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
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

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
    response_truncated  BOOLEAN NOT NULL DEFAULT FALSE,
    grounding_bundle    JSONB,
    router_raw          JSONB,
    intents_count       INT,
    intent_dispatch     JSONB,
    cache_tokens        JSONB,
    router_model        TEXT,
    failure_recovered   BOOLEAN NOT NULL DEFAULT FALSE,
    gatherer_status     TEXT,
    tool_calls_count    INT,
    need_deeper_extensions INT,
    confidence          TEXT,
    dropped_count       INT,
    cache_hit_rate      NUMERIC(4, 3),
    trace_blob_key      TEXT
);

CREATE INDEX idx_query_traces_customer_time
    ON query_traces (customer_id, occurred_at DESC);
CREATE INDEX idx_query_traces_request_id
    ON query_traces (request_id);

ALTER TABLE query_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_traces FORCE ROW LEVEL SECURITY;

CREATE POLICY query_traces_tenant_isolation ON query_traces
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

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

-- Dedup guard: at most one OUTSTANDING (not-yet-done) row per
-- (customer, anchor_doc, extractor). Makes the enqueue-time
-- `ON CONFLICT DO NOTHING` actually bite, so re-delivers and bursty
-- re-persists (agent-session batch appends, repo re-scans) cannot flood the
-- queue with duplicate pending work. Partial `WHERE done_at IS NULL` so a
-- genuine re-extraction after completion (prompt-version bump, real content
-- change) still enqueues a fresh row. See migration 0096.
CREATE UNIQUE INDEX idx_inferred_edges_queue_outstanding
    ON inferred_edges_queue (customer_id, anchor_doc_id, extractor_id)
    WHERE done_at IS NULL;

-- inferred_edges_queue is an internal queue table drained CROSS-tenant by
-- the inferred-edges side-worker (services/ingestion/inferred_edges/
-- worker.py:_claim_one). Under FORCE RLS that drain SELECT silently
-- zero-matches when running as a non-superuser role (e.g. probe_app),
-- because there's no GUC to set before the row is claimed. Follows the
-- same no-RLS pattern as ingestion_queue / backfill_state. See
-- migration 0068.
--
-- Tenant scoping is enforced by the side-worker wrapping the per-row
-- processing in `with_tenant(customer_id)` AND the SQL filtering on
-- customer_id explicitly — same belt-and-suspenders as the other queue
-- tables in this schema.

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

-- ---------------------------------------------------------------------------
-- ingestion_cursors: per-customer-per-source-per-resource polling state
-- (migration 0072). For self-host customers (INGESTION_MODE=poll), the
-- worker pod runs PollScheduler which walks this table every tick. See
-- services/ingestion/polling/ for the framework. RLS is NON-FORCE (matches
-- the inferred_edges_queue pattern) — the scheduler reads cross-tenant for
-- work distribution, per-tenant writes set the GUC via with_tenant().
-- ---------------------------------------------------------------------------
CREATE TABLE ingestion_cursors (
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source          TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    cursor_value    TEXT,
    polled_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_error      TEXT,
    last_error_at   TIMESTAMPTZ,
    CONSTRAINT ingestion_cursors_pkey PRIMARY KEY (customer_id, source, resource_id)
);

CREATE INDEX ix_ingestion_cursors_source_polled_at
    ON ingestion_cursors (source, polled_at DESC);

ALTER TABLE ingestion_cursors ENABLE ROW LEVEL SECURITY;
-- NO FORCE — the scheduler walks cross-tenant.
CREATE POLICY ingestion_cursors_tenant_isolation ON ingestion_cursors
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- node_post_write_queue: drain queue for the post-write pipeline
-- (migrations 0082 + 0083). Enqueued by graph_writer.upsert_nodes in the
-- same transaction as the node upsert. Drained by PostWriteWorker which
-- runs AutoMergeAnalyzer + (future) other NodeAnalyzers per row.
--
-- RLS is DISABLED (per 0083): the worker drains cross-tenant and probe_app
-- lacks BYPASSRLS. Tenant scoping is preserved at the INSERT site
-- (graph_writer runs inside with_tenant) and at the downstream node
-- read (worker calls with_tenant(customer_id) before loading the node).
-- ---------------------------------------------------------------------------
CREATE TABLE node_post_write_queue (
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    node_id          BIGINT NOT NULL,
    enqueued_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analyzer_status  JSONB NOT NULL DEFAULT '{}'::JSONB,
    locked_until     TIMESTAMPTZ NULL,
    PRIMARY KEY (customer_id, node_id)
);
CREATE INDEX idx_node_post_write_queue_pending
    ON node_post_write_queue (enqueued_at)
    WHERE locked_until IS NULL;

-- ---------------------------------------------------------------------------
-- pending_edges: deferred-edge queue (migration 0095).
--
-- upsert_edges resolves endpoints only against the node set in the same batch.
-- An edge whose endpoint is not yet ingested (a run before its experiment; the
-- research-os outbox delivers out of order) would otherwise be silently
-- dropped. Instead it parks here keyed by the MISSING endpoint's
-- (label, canonical_id); the post-write worker drains matching rows when that
-- node lands, and a TTL reaper sweeps rows whose counterpart never arrives.
-- Queue depth is the completeness signal the silent drop never gave.
-- ---------------------------------------------------------------------------
CREATE TABLE pending_edges (
    id                   BIGSERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    missing_label        TEXT NOT NULL,
    missing_canonical_id TEXT NOT NULL,
    edge_type            TEXT NOT NULL,
    from_label           TEXT NOT NULL,
    from_canonical_id    TEXT NOT NULL,
    to_label             TEXT NOT NULL,
    to_canonical_id      TEXT NOT NULL,
    source_system        TEXT NOT NULL,
    properties           JSONB NOT NULL DEFAULT '{}'::JSONB,
    valid_from           TIMESTAMPTZ NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_until         TIMESTAMPTZ NULL
);
CREATE INDEX idx_pending_edges_missing
    ON pending_edges (customer_id, missing_label, missing_canonical_id)
    WHERE locked_until IS NULL;
CREATE INDEX idx_pending_edges_created
    ON pending_edges (created_at)
    WHERE locked_until IS NULL;
ALTER TABLE pending_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pending_edges FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON pending_edges
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- entity_merge_suggestions: medium/low-confidence verdicts from
-- AutoMergeAnalyzer surfaced in the dashboard /graph cluster admin UI
-- (migration 0082). High-confidence verdicts go straight to the
-- entity-clusters merge endpoint; everything else lands here for human
-- review.
-- ---------------------------------------------------------------------------
CREATE TABLE entity_merge_suggestions (
    suggestion_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id            TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                  TEXT NOT NULL,
    primary_canonical_id   TEXT NOT NULL,
    candidate_canonical_id TEXT NOT NULL,
    confidence             TEXT NOT NULL CHECK (confidence IN ('high','medium','low')),
    rationale              TEXT NULL,
    llm_model              TEXT NOT NULL,
    run_id                 UUID NULL,
    status                 TEXT NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending','approved','dismissed','applied')),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at             TIMESTAMPTZ NULL,
    decided_by_user_id     UUID NULL
);
CREATE INDEX idx_entity_merge_suggestions_lookup
    ON entity_merge_suggestions (customer_id, status, created_at DESC);
CREATE UNIQUE INDEX uq_entity_merge_suggestions_pair
    ON entity_merge_suggestions (customer_id, label, primary_canonical_id, candidate_canonical_id)
    WHERE status = 'pending';

ALTER TABLE entity_merge_suggestions ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_suggestions FORCE ROW LEVEL SECURITY;
CREATE POLICY entity_merge_suggestions_tenant_isolation
    ON entity_merge_suggestions
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- purge_runs: durable outcome of a per-source disconnect purge.
--
-- The caller (research-os / prbe-backend) removes its own record of the
-- integration ONLY after a purge reports verified=true, so it needs an
-- outcome that survives a lost HTTP response, a pod restart mid-purge, or a
-- client timeout. Without this table a caller that dropped its connection has
-- no way to learn whether the delete finished, and re-running blind is the
-- only recovery.
--
-- Rows cascade with the customer: a purge is per-source, and a whole-tenant
-- delete removes everything the history could still describe.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS purge_runs (
    purge_id       UUID PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system  TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running','done','failed')),
    result         JSONB,
    error          TEXT,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_purge_runs_customer_source
    ON purge_runs (customer_id, source_system, started_at DESC);

ALTER TABLE purge_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE purge_runs FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON purge_runs
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- ---------------------------------------------------------------------------
-- Workflow memory Phase 0: the procedure store (migration 0114).
--
-- situations / clauses / clause_situation_edges / clause_evidence /
-- serve_ledger. All five carry data in v0. `procedures` /
-- `procedure_clauses` are deliberately NOT created — nothing writes them in
-- v0 and their shape is unknown.
--
-- DELIBERATELY ABSENT: clause_evidence stores evidence BY REFERENCE, enforced
-- on TWO axes — (a) no quote/text column, and (b) a CHECK refusing quote-shaped
-- keys inside source_ref, which is the unconstrained JSONB where a quote would
-- actually land. Evidence resolves at view time through the viewer's ACL and a
-- baked-in quote escapes that check; both axes have tests in
-- tests/test_workflow_memory_isolation.py. situation_occurrences and
-- conformance are not created either — they need Phase 1's detectors.
--
-- This block MUST stay identical to
-- db/migrations/versions/20260820_0114_workflow_memory_store.py: CI applies
-- this file and stamps alembic head, it never runs the migration chain.
-- ---------------------------------------------------------------------------
CREATE TABLE situations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    slug            TEXT NOT NULL,
    label           TEXT NOT NULL,
    description     TEXT NOT NULL,
    example_signals JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 0118. Whether this row is one of the CLASSIFIER'S LABELS. The `misc`
    -- fallback is false: rules land there when nothing fit, and putting a
    -- "none of the above" description into an embedding label space either
    -- matches nothing or weakly matches everything -- the second of which
    -- steals traffic from real situations. Declared LAST because
    -- `ALTER TABLE ADD COLUMN` appends and this file must reproduce the
    -- migration chain's column order exactly.
    classifiable    BOOLEAN NOT NULL DEFAULT true,
    UNIQUE (customer_id, slug),
    -- Redundant for uniqueness (id is already the PK), but REQUIRED as
    -- the target of the composite FKs below: Postgres will only
    -- reference columns that carry a unique constraint.
    UNIQUE (customer_id, id)
);

CREATE TABLE clauses (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    kind                TEXT NOT NULL
                        CHECK (kind IN ('step','check','exception','anti_pattern','asset')),
    semantic_action     TEXT,
    body                TEXT NOT NULL,
    binding             JSONB NOT NULL DEFAULT '{}'::jsonb,
    scope               JSONB NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT NOT NULL
                        CHECK (status IN ('declared','documented','observed_convention',
                                          'success_associated','expert_confirmed',
                                          'intervention_validated','exception','anti_pattern',
                                          'contested','stale','agent_proposed')),
    owner_ref           TEXT,
    author_ref          TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    lineage             JSONB NOT NULL DEFAULT '{}'::jsonb,
    salience            REAL NOT NULL DEFAULT 0.5,
    binding_health      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Publication, which is a SEPARATE AXIS from corroboration (migration
    -- 0112). NULL means the clause is visible under the ordinary two-human
    -- rule; non-NULL means a named person published it on their own
    -- authority, without waiting for a second human to agree. Two columns
    -- rather than a boolean because unilateral publication is an act of
    -- authority over what a team is told to do, and an unattributable one is
    -- worse than none. NOT a `status` value: status describes how strong the
    -- evidence is, this describes whether the clause has been published, and
    -- a force-published rule is still `declared` with one human behind it.
    --
    -- LAST, AFTER created_at/updated_at, and that is not cosmetic. 0112 adds
    -- them with ALTER TABLE ADD COLUMN, which appends -- so declaring them
    -- mid-table here gives schema.sql different ordinal positions than the
    -- migration chain produces. The drift guard compares ordinals and caught
    -- exactly that. Any future column added by an ALTER goes at the bottom.
    shared_by           TEXT,
    shared_at           TIMESTAMPTZ,
    CONSTRAINT ck_clauses_publication_is_attributed
        CHECK ((shared_by IS NULL) = (shared_at IS NULL)),
    -- The body's embedding, stored so neighbour search embeds ONE text per
    -- declaration instead of the whole corpus (migration 0117). The model id
    -- is not optional: a cosine between vectors from two different embedders
    -- is a plausible-looking number rather than an error, so a row embedded by
    -- an older model must be EXCLUDED from a search, not silently compared.
    -- Same reasoning that puts model_id in the classifier's cache key.
    body_embedding       halfvec(3072),
    body_embedding_model TEXT,
    CONSTRAINT ck_clauses_embedding_names_its_model
        CHECK ((body_embedding IS NULL) = (body_embedding_model IS NULL)),
    UNIQUE (customer_id, id)          -- composite-FK target; see situations
);

CREATE INDEX clauses_customer_status_idx ON clauses (customer_id, status);
-- Partial: published clauses are the minority and the only rows this serves.
-- The visibility predicate tests `shared_by IS NOT NULL`, which a full index
-- over a mostly-NULL column would answer no faster.
CREATE INDEX clauses_published_idx ON clauses (customer_id, shared_at)
    WHERE shared_by IS NOT NULL;
-- Neighbour search at declaration time. House pattern: chunks,
-- directed_vectors and graph_nodes all index halfvec with HNSW cosine.
CREATE INDEX clauses_body_embedding_hnsw
    ON clauses USING hnsw (body_embedding halfvec_cosine_ops);

-- COMPOSITE FKs, not simple ones. Postgres referential-integrity checks
-- bypass row security by design, and the tenant policy only inspects THIS
-- row's customer_id -- so a simple `REFERENCES clauses(id)` lets tenant A
-- attach an edge to tenant B's clause id. That succeeds iff the uuid
-- exists in any tenant, which is a cross-tenant existence oracle, and it
-- corrupts any later join that runs outside a tenant GUC. Keying the FK
-- on (customer_id, id) makes the mismatch unrepresentable.
-- Caveat: "unrepresentable" is slightly too strong. The FK route is closed,
-- but an explicit-id insert of another tenant's uuid still errors DIFFERENTLY
-- from a random uuid via the single-column PK, so a narrow existence oracle
-- survives at the primary key.
CREATE TABLE clause_situation_edges (
    clause_id       UUID NOT NULL,
    situation_id    UUID NOT NULL,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id, clause_id)
        REFERENCES clauses (customer_id, id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id, situation_id)
        REFERENCES situations (customer_id, id) ON DELETE CASCADE,
    when_conditions JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- {method,confidence,model,prompt_version,classified_at}. The only
    -- reclassification input not derivable from the clause itself: once
    -- an edge is written, "situation X" is all you know.
    classification  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (clause_id, situation_id)
);

CREATE INDEX cse_situation_idx ON clause_situation_edges (customer_id, situation_id);

-- NO quote/text column here, AND no quote-shaped key inside source_ref.
-- See the section header: the column-absence claim alone was wrong.
CREATE TABLE clause_evidence (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    clause_id        UUID NOT NULL,
    FOREIGN KEY (customer_id, clause_id)
        REFERENCES clauses (customer_id, id) ON DELETE CASCADE,
    source_class     TEXT NOT NULL
                     CHECK (source_class IN ('declared','human_doc','human_message',
                                             'pr_review','human_wiki_edit',
                                             'agent_transcript','run_outcome')),
    -- A POINTER, not a payload: {"session": "...", "span": [0, 1]}.
    -- Unconstrained JSONB is exactly where a quote would land, so the
    -- quote-shaped keys are refused here rather than left to review.
    source_ref       JSONB NOT NULL
                     CHECK (NOT (source_ref ?| ARRAY['quote','text','body','content',
                                                     'excerpt','snippet','raw','full_text',
                                                     'verbatim','passage'])),
    author_ref       TEXT,
    -- NO DEFAULT, deliberately. A default of FALSE fails OPEN: a writer that
    -- forgets the taint computation records the evidence as clean, and
    -- taint-excluded support is a stated non-negotiable. With NOT NULL and no
    -- default, an omission is a constraint violation and every write has to
    -- state its taint.
    exposure_tainted BOOLEAN NOT NULL,
    ts               TIMESTAMPTZ NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Leads with the tenant column, matching the composite FK.
CREATE INDEX clause_evidence_clause_idx ON clause_evidence (customer_id, clause_id);

CREATE TABLE serve_ledger (
    id           BIGSERIAL PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    -- Bare ids, deliberately no FK: this is an append-only audit log and
    -- an ON DELETE rule would silently rewrite exposure history. The only
    -- FK is customer_id, whose CASCADE is intended.
    clause_ids   UUID[] NOT NULL,
    situation_id UUID,
    session_id   TEXT,
    -- WHO the delivery went to. Without this the ledger cannot answer
    -- "was this person's agent shown this clause before they produced
    -- that evidence", which IS the taint join -- and taint-excluded
    -- support is the §7 non-negotiable the whole ledger exists for.
    -- Note for Stage 4: no user identity crosses into prbe-knowledge
    -- today (engine_headers sends only the internal key + customer), so
    -- the /v1/procedures contract must start carrying the actor.
    actor_ref    TEXT,
    channel      TEXT NOT NULL
                 CHECK (channel IN ('retrieved','strip','compiled','injected')),
    route        TEXT NOT NULL DEFAULT 'n/a' CHECK (route IN ('dumb','smart','n/a')),
    mode         TEXT NOT NULL DEFAULT 'live' CHECK (mode IN ('live','shadow')),
    trigger      TEXT,
    served_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX serve_ledger_session_idx ON serve_ledger (customer_id, session_id);
CREATE INDEX serve_ledger_actor_idx ON serve_ledger (customer_id, actor_ref, served_at);
-- Phase 1's taint join asks "which clauses were served into this session",
-- which is a containment query over the array. Without GIN that is a seq
-- scan over a table that grows with every request.
CREATE INDEX serve_ledger_clause_ids_idx ON serve_ledger USING GIN (clause_ids);

-- Postgres has no ON UPDATE for column defaults, so an `updated_at DEFAULT
-- now()` never advances past insert time. Shape copied from
-- customers_fill_r2_bucket above, the house convention.
CREATE OR REPLACE FUNCTION wfmem_touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER situations_touch_updated_at_trg
    BEFORE UPDATE ON situations
    FOR EACH ROW
    EXECUTE FUNCTION wfmem_touch_updated_at();

CREATE TRIGGER clauses_touch_updated_at_trg
    BEFORE UPDATE ON clauses
    FOR EACH ROW
    EXECUTE FUNCTION wfmem_touch_updated_at();

-- A stored embedding goes stale the moment the text it describes changes, and
-- staleness here is INVISIBLE: the search still returns results, just wrong
-- ones. Clearing both columns on a body edit means a stale vector can never be
-- compared against -- the clause drops out of neighbour search until something
-- re-embeds it, which is the safe direction to fail. Nothing in v0 updates a
-- body; this exists so that when an edit path is added, it cannot introduce
-- the bug by omission.
CREATE OR REPLACE FUNCTION wfmem_clear_stale_clause_embedding() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.body IS DISTINCT FROM OLD.body THEN
        NEW.body_embedding := NULL;
        NEW.body_embedding_model := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER clauses_clear_stale_embedding_trg
    BEFORE UPDATE ON clauses
    FOR EACH ROW
    EXECUTE FUNCTION wfmem_clear_stale_clause_embedding();

-- RLS on all five. FORCE so the policy applies to the table owner too, and
-- BOTH halves: USING alone hides another tenant's rows on read but still
-- lets a buggy writer file a row under the wrong customer.
--
-- serve_ledger is the exception: FOR SELECT + FOR INSERT policies only. It is
-- an append-only exposure log, and under FORCE RLS the ABSENCE of an
-- UPDATE/DELETE policy is the deny — a tenant cannot backdate or erase its own
-- history. Backdating is worse than deletion: it silently inverts the taint
-- join the table exists for.
ALTER TABLE situations ENABLE ROW LEVEL SECURITY;
ALTER TABLE situations FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON situations
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

ALTER TABLE clauses ENABLE ROW LEVEL SECURITY;
ALTER TABLE clauses FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON clauses
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

ALTER TABLE clause_situation_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE clause_situation_edges FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON clause_situation_edges
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

ALTER TABLE clause_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE clause_evidence FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON clause_evidence
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

ALTER TABLE serve_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE serve_ledger FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_select ON serve_ledger
    FOR SELECT
    USING (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation_insert ON serve_ledger
    FOR INSERT
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));
-- Belt-and-braces for any role that would otherwise inherit the privilege
-- through PUBLIC; the deny above is the policy absence, not this REVOKE.
REVOKE UPDATE, DELETE ON serve_ledger FROM PUBLIC;

-- ---------------------------------------------------------------------------
-- pg_search_guardian_state: one row remembering the Postgres timeline ID the
-- guardian cron last saw (migration 0120).
--
-- The timeline increments on every promotion, and a promotion is the only
-- reliable signal that this instance's pg_search indexes are suspect. A
-- promoted standby carries either a 0-byte index (pg_search Community does not
-- replicate index storage -- the 2026-08-25 kb outage) or, worse, a NONZERO
-- index frozen at clone time that plans fine and silently returns incomplete
-- results. Size-based detection sees only the first. See
-- scripts/cron_pg_search_guardian.py.
--
-- NO RLS: there is no customer_id here -- one row about one Postgres instance,
-- not tenant data. Same shape of exemption as node_post_write_queue.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pg_search_guardian_state (
    id                SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_timeline_id  BIGINT NOT NULL,
    observed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Which required-index absences have already been alerted (0121), so the
    -- guardian alerts on transitions rather than once a minute for the whole
    -- attended-rebuild window.
    known_absent      TEXT[] NOT NULL DEFAULT '{}'
);

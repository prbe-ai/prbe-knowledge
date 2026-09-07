-- Installation-bound GitHub control protocol v2. Legacy singleton rows remain
-- readable/claimable by old binaries; v2 work never uses their status values.
CREATE TABLE IF NOT EXISTS github_installations (
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    installation_id TEXT NOT NULL,
    managed BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    sync_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    scope JSONB NOT NULL DEFAULT '[]',
    revision BIGINT NOT NULL DEFAULT 1,
    generation BIGINT NOT NULL DEFAULT 1,
    last_attempt_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, installation_id)
);
CREATE TABLE IF NOT EXISTS github_backfill_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id TEXT NOT NULL,
    installation_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'backfill' CHECK (kind IN ('backfill','catchup')),
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN
      ('queued','running','cancel_requested','canceled','completed','failed')),
    scope JSONB NOT NULL,
    generation BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    cursor TEXT,
    enumeration_complete BOOLEAN NOT NULL DEFAULT FALSE,
    attempts INT NOT NULL DEFAULT 0,
    lease_id UUID,
    heartbeat_at TIMESTAMPTZ,
    processed_count BIGINT NOT NULL DEFAULT 0,
    counts JSONB NOT NULL DEFAULT '{}',
    warnings JSONB NOT NULL DEFAULT '[]',
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    FOREIGN KEY (customer_id,installation_id)
      REFERENCES github_installations(customer_id,installation_id) ON DELETE CASCADE,
    UNIQUE (customer_id,installation_id,idempotency_key)
);
CREATE INDEX IF NOT EXISTS github_jobs_claim ON github_backfill_jobs(state,created_at);
CREATE INDEX IF NOT EXISTS github_jobs_history ON github_backfill_jobs
    (customer_id,installation_id,created_at DESC,id DESC);
CREATE TABLE IF NOT EXISTS github_worker_capabilities (
    worker_id TEXT PRIMARY KEY,
    protocol_version INT NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS github_document_bindings (
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    installation_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    PRIMARY KEY(customer_id,installation_id,doc_id),
    FOREIGN KEY(customer_id,installation_id)
        REFERENCES github_installations(customer_id,installation_id) ON DELETE CASCADE
);
ALTER TABLE github_document_bindings ENABLE ROW LEVEL SECURITY;
ALTER TABLE github_document_bindings FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON github_document_bindings USING
    (customer_id = current_setting('app.current_customer_id',true));
ALTER TABLE ingestion_queue ADD COLUMN IF NOT EXISTS github_installation_id TEXT;
ALTER TABLE ingestion_queue ADD COLUMN IF NOT EXISTS github_generation BIGINT;
ALTER TABLE ingestion_queue ADD COLUMN IF NOT EXISTS github_job_id UUID;
ALTER TABLE ingestion_queue ADD COLUMN IF NOT EXISTS github_payload JSONB;
ALTER TABLE ingestion_queue ADD COLUMN IF NOT EXISTS github_lease_id UUID;
CREATE INDEX IF NOT EXISTS github_queue_claim ON ingestion_queue(priority DESC,enqueued_at)
    WHERE status = 'v2_pending';
CREATE INDEX IF NOT EXISTS github_queue_job ON ingestion_queue(github_job_id,status)
    WHERE github_job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS github_queue_installation ON ingestion_queue
    (customer_id,github_installation_id,status) WHERE github_installation_id IS NOT NULL;
ALTER TABLE github_installations ENABLE ROW LEVEL SECURITY;
ALTER TABLE github_installations FORCE ROW LEVEL SECURITY;
ALTER TABLE github_backfill_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE github_backfill_jobs FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON github_installations USING
    (customer_id = current_setting('app.current_customer_id',true));
CREATE POLICY tenant_isolation ON github_backfill_jobs USING
    (customer_id = current_setting('app.current_customer_id',true));
-- Existing installations stay on their accepted legacy lane until an explicit
-- settings save adopts protocol 2. Do not reset or relabel an existing job.
-- Migrations run as schema owner; FORCE RLS requires a tenant per insertion.
DO $$ DECLARE item RECORD; BEGIN
    FOR item IN SELECT customer_id,external_id FROM customer_source_mapping WHERE source_system='github' LOOP
        PERFORM set_config('app.current_customer_id',item.customer_id,true);
        INSERT INTO github_installations(customer_id,installation_id,sync_enabled)
            VALUES(item.customer_id,item.external_id,TRUE) ON CONFLICT DO NOTHING;
    END LOOP;
END $$;

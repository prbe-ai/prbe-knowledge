-- Deleted coding-agent sessions: the record that keeps them deleted (migration 0140).
-- One row per (tenant, source, session). Every writer of a session checks it
-- under the session's advisory lock and refuses (engine/shared/session_suppression.py),
-- so a client that retries or re-uploads cannot bring the session back.
-- Holds no transcript content and no author identity: the selector records only
-- HOW the session was chosen ({"by": "id"} / {"by": "author"}).
CREATE TABLE IF NOT EXISTS session_deletions (
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    -- The request that last ran this deletion (a re-run restamps it).
    deletion_id   UUID NOT NULL,
    reason        TEXT NOT NULL,
    ticket        TEXT,
    selector      JSONB NOT NULL DEFAULT '{}'::jsonb,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'done', 'failed', 'held')),
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- When a run last started on this session. A `pending` row whose attempt
    -- is old was interrupted (pod restart); re-POSTing the request resumes it.
    attempted_at  TIMESTAMPTZ,
    -- Set when a run found no row and no raw object of the session left.
    deleted_at    TIMESTAMPTZ,
    -- Raw object keys found through database references (queue, receipts,
    -- ingestion events) BEFORE those rows were removed. Protocol-1 keys are
    -- date-addressed, so once the rows are gone this is the only list of them.
    -- Emptied when their deletion is confirmed.
    pending_keys  TEXT[] NOT NULL DEFAULT '{}',
    result        JSONB,
    error         TEXT,
    PRIMARY KEY (customer_id, source_system, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_deletions_request
    ON session_deletions (customer_id, deletion_id);
ALTER TABLE session_deletions ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_deletions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON session_deletions;
CREATE POLICY tenant_isolation ON session_deletions
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

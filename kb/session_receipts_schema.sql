-- Protocol-2 acceptance is immutable and tenant-scoped. Bodies remain in R2.
CREATE TABLE IF NOT EXISTS session_streams (
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system TEXT NOT NULL,
    session_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    protocol_version INTEGER NOT NULL CHECK (protocol_version = 2),
    last_seq BIGINT NOT NULL DEFAULT -1,
    source_byte_end BIGINT NOT NULL DEFAULT 0,
    source_line_end BIGINT NOT NULL DEFAULT 0,
    event_end BIGINT NOT NULL DEFAULT 0,
    prefix_sha256 TEXT NOT NULL,
    finalized BOOLEAN NOT NULL DEFAULT FALSE,
    uploader_device_id TEXT,
    snapshot_byte_end BIGINT,
    snapshot_sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, source_system, session_id)
);
CREATE TABLE IF NOT EXISTS session_batch_receipts (
    customer_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    session_id TEXT NOT NULL,
    batch_seq BIGINT NOT NULL,
    body_sha256 TEXT NOT NULL,
    payload_key TEXT NOT NULL,
    source_byte_end BIGINT NOT NULL,
    source_line_end BIGINT NOT NULL,
    event_end BIGINT NOT NULL,
    prefix_sha256 TEXT NOT NULL,
    finalized BOOLEAN NOT NULL,
    accepted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, source_system, session_id, batch_seq),
    FOREIGN KEY (customer_id, source_system, session_id)
      REFERENCES session_streams(customer_id, source_system, session_id) ON DELETE CASCADE
);
ALTER TABLE session_streams ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_streams FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON session_streams USING
    (customer_id = current_setting('app.current_customer_id', true));
ALTER TABLE session_batch_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_batch_receipts FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON session_batch_receipts USING
    (customer_id = current_setting('app.current_customer_id', true));

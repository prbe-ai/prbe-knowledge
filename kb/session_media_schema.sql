-- Pasted images of captured coding-agent sessions (migration 0149, kb/session_media.py).
-- One row per (tenant, source, session, image): written when the upload route
-- hands out a presigned PUT for the image's staging key, stamped `named_at`
-- when an accepted batch of the session names the image, and `promoted_at`
-- when the worker copies the staged bytes into the session's folder
-- (`raw/<src>/<customer>/<session>/media/<sha256>.<ext>`). The rows of the
-- last 24 hours are the customer's daily upload budget. The hourly sweep
-- retries named, unpromoted images for 7 days and removes unpromoted rows past
-- their window; a session deletion removes all of them.
-- Holds no image bytes and nothing from the transcript: a digest, a size, a type.
CREATE TABLE IF NOT EXISTS session_media (
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    source_system TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    sha256        TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes    BIGINT NOT NULL CHECK (size_bytes > 0),
    media_type    TEXT NOT NULL
                  CHECK (media_type IN ('image/jpeg', 'image/png', 'image/gif', 'image/webp')),
    -- When an upload URL was last handed out for it (a re-request restamps it).
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- When an accepted batch first named it: from then on its promotion is
    -- owed, and the sweep retries it until promoted or given up.
    named_at      TIMESTAMPTZ,
    promoted_at   TIMESTAMPTZ,
    PRIMARY KEY (customer_id, source_system, session_id, sha256)
);
-- The daily budget (a customer's rows of the last 24 hours) and the sweep.
CREATE INDEX IF NOT EXISTS idx_session_media_created
    ON session_media (customer_id, created_at);
-- The sweep's retry: named, not promoted.
CREATE INDEX IF NOT EXISTS idx_session_media_owed
    ON session_media (customer_id, named_at)
    WHERE promoted_at IS NULL AND named_at IS NOT NULL;
ALTER TABLE session_media ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_media FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON session_media;
CREATE POLICY tenant_isolation ON session_media
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));
-- The app role is deployment-specific (`app` on research, `probe_app` on
-- managed, neither on some self-hosts), so the grant is discovered rather than
-- hardcoded -- same as migration 0140.
DO $$
DECLARE role_name text;
BEGIN
    FOR role_name IN
        SELECT rolname FROM pg_roles WHERE rolname IN ('app', 'probe_app', 'probe_admin')
    LOOP
        EXECUTE format(
            'GRANT SELECT, INSERT, UPDATE, DELETE ON session_media TO %I',
            role_name
        );
    END LOOP;
END
$$;

-- ---------------------------------------------------------------------------
-- retrieve_pages (migration 0129)
--
-- The rest of a retrieval, kept so the caller can ask for it.
--
-- /retrieve returns `top_k` documents out of a fused pool that is routinely
-- wider, and until now the surplus was simply dropped: the response had no
-- cursor, so "there were more" was unsayable and an agent's only move was to
-- re-run the same search with a bigger number and pay the whole pipeline again
-- (~7-14s, an LLM turn, four retrieval channels).
--
-- A page row is that surplus, WITH its content. Storing ids and re-querying on
-- page 2 would be smaller, and it would also make paging a moving corpus: a
-- document re-indexed between pages moves in the ranking, so the reader sees
-- one result twice and never sees another. A page is deliberately a SNAPSHOT of
-- what the search found, which is the only way "page 2" can mean anything.
--
-- Written synchronously, before the response returns, so a cursor never names a
-- row that does not exist yet -- the failure mode a post-flush background write
-- would have handed to every fast caller.
--
-- Short-lived by construction: every write first deletes rows older than a day,
-- so the store cleans itself and there is no cron to deploy, monitor or forget.
-- A cursor older than that is refused rather than silently re-searched, because
-- a silent re-search is exactly what produces the duplicates paging prevents.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retrieve_pages (
    page_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query       TEXT NOT NULL,
    items       JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieve_pages_created
    ON retrieve_pages (created_at);

ALTER TABLE retrieve_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE retrieve_pages FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS retrieve_pages_tenant_isolation ON retrieve_pages;
CREATE POLICY retrieve_pages_tenant_isolation ON retrieve_pages
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));

-- The app role is deployment-specific (`app` on research, `probe_app` on
-- managed, neither on some self-hosts), so the grant is discovered rather than
-- hardcoded -- same reason as migration 0112. Without it the engine's page
-- writes fail quietly and pagination looks shipped-but-unused.
DO $$
DECLARE role_name text;
BEGIN
    FOR role_name IN
        SELECT rolname FROM pg_roles WHERE rolname IN ('app', 'probe_app', 'probe_admin')
    LOOP
        EXECUTE format(
            'GRANT SELECT, INSERT, UPDATE, DELETE ON retrieve_pages TO %I',
            role_name
        );
    END LOOP;
END
$$;

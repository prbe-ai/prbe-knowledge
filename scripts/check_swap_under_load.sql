CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE documents (
    doc_id      TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    version     INT  NOT NULL,
    title       TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (customer_id, doc_id, version)
);

CREATE TABLE chunks (
    chunk_id            TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    chunk_index         INT  NOT NULL,
    content             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    token_count         INT  NOT NULL DEFAULT 1,
    embedding_v2        vector(3) NULL,
    first_seen_version  INT  NOT NULL DEFAULT 1,
    last_seen_version   INT  NOT NULL DEFAULT 1,
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to            TIMESTAMPTZ,
    project_id          TEXT,
    title               TEXT,
    kind                TEXT NOT NULL DEFAULT 'body',
    visibility          TEXT NOT NULL DEFAULT 'private',
    content_tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    PRIMARY KEY (customer_id, chunk_id),
    CONSTRAINT chunks_customer_doc_hash_key UNIQUE (customer_id, doc_id, content_hash)
);

CREATE INDEX idx_chunks_customer ON chunks (customer_id);
CREATE INDEX idx_chunks_doc      ON chunks (doc_id);
CREATE INDEX idx_chunks_doc_live ON chunks (doc_id) WHERE valid_to IS NULL;
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);
CREATE INDEX idx_chunks_embedding_v2_hnsw_live
    ON chunks USING hnsw (embedding_v2 vector_cosine_ops) WHERE valid_to IS NULL;

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
    USING (customer_id = current_setting('app.current_customer_id', true));

CREATE OR REPLACE FUNCTION chunks_fill_project_id_on_insert() RETURNS trigger AS $fn$
BEGIN
    IF NEW.project_id IS NULL THEN
        SELECT d.metadata->>'project_id' INTO NEW.project_id
        FROM documents d
        WHERE d.doc_id = NEW.doc_id AND d.customer_id = NEW.customer_id
          AND d.version BETWEEN NEW.first_seen_version AND NEW.last_seen_version
        ORDER BY d.version DESC LIMIT 1;
    END IF;
    RETURN NEW;
END; $fn$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION chunks_fill_title_on_insert() RETURNS trigger AS $fn$
BEGIN
    IF NEW.title IS NULL OR NEW.title = '' THEN
        SELECT coalesce(d.title, '') INTO NEW.title
        FROM documents d
        WHERE d.doc_id = NEW.doc_id AND d.customer_id = NEW.customer_id
        ORDER BY d.version DESC LIMIT 1;
    END IF;
    RETURN NEW;
END; $fn$ LANGUAGE plpgsql;

CREATE TRIGGER trg_chunks_fill_project_id BEFORE INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_fill_project_id_on_insert();
CREATE TRIGGER trg_chunks_fill_title BEFORE INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_fill_title_on_insert();

INSERT INTO customers (customer_id) VALUES
    ('alpha-co'), ('beta_co'), ('gamma-3');

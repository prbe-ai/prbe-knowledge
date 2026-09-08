"""Companion injection infrastructure: mailbox, deliveries, claims.

The transport half of the mid-session companion (implementation spec v3
§2.1). Three tables:

  companion_mailbox     append-only: one row per card a writer wants a session
                        (or a person, for the MCP lane) to receive.
  companion_deliveries  append-only: one row per EMISSION-ATTEMPT OUTCOME,
                        keyed by a client-minted attempt_id so ack retries are
                        no-ops and repeated emissions stay visible.
  companion_claims      the one MUTABLE table: short server-side leases for the
                        actor-keyed lane, so two concurrent MCP calls cannot
                        both take the same card.

Dedupe is two PARTIAL unique indexes, not one UNIQUE. A plain
UNIQUE (customer_id, session_id, dedupe_key) lets every actor-keyed row through,
because their session_id is NULL and NULLs never collide -- the audit reproduced
duplicate actor rows against exactly that constraint.

RLS mirrors serve_ledger: FORCE + SELECT/INSERT policies only on the append-only
tables (the missing UPDATE/DELETE policy IS the deny, and it is a quiet one:
zero rows affected, no error), all four on claims. Grants stay with the
deployment's role setup, as for every other table here; schema.sql carries no
application-role GRANTs, so the operator step that grants the app role must add
SELECT, INSERT on the two append-only tables and all four on claims.

Deliberately OUTSIDE the wfmem family: clause_ids and serve_ledger_id are
reserved for the intelligent layer and stay NULL until it exists.
"""

from alembic import op

revision = "0129_companion_mailbox"
down_revision = "0127_github_installation_control"
branch_labels = None
depends_on = None

UP = """
CREATE TABLE companion_mailbox (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    -- Always 'user:<uuid>', resolved by research-os from either credential
    -- kind (a user principal, or an ingest token's owning user).
    recipient     TEXT NOT NULL CHECK (recipient LIKE 'user:%'),
    -- NULL only for actor-keyed rows (the MCP-rider lane).
    session_id    TEXT CHECK (session_id IS NULL OR length(session_id) BETWEEN 1 AND 200),
    class         TEXT NOT NULL CHECK (class IN ('seam', 'decision-push')),
    -- Routing metadata from the driver's --seam; NULL means any seam may take it.
    intended_seam TEXT,
    -- The '[probe companion] ' prefix is applied at write time and counts
    -- toward the cap.
    body          TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 4000),
    clause_ids    UUID[],
    dedupe_key    TEXT NOT NULL CHECK (length(dedupe_key) BETWEEN 1 AND 200),
    mode          TEXT NOT NULL DEFAULT 'live' CHECK (mode IN ('live', 'shadow')),
    source        TEXT NOT NULL CHECK (source IN ('driver', 'brain')),
    trial_id      UUID NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL,
    CHECK (expires_at > created_at AND expires_at <= created_at + interval '24 hours'),
    -- A decision card belongs to a session's decision generation; an
    -- actor-keyed one has no session whose decision it could belong to.
    CHECK (class = 'seam' OR session_id IS NOT NULL),
    UNIQUE (customer_id, id)
);
CREATE UNIQUE INDEX companion_mailbox_session_dedupe_idx
    ON companion_mailbox (customer_id, session_id, dedupe_key) WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX companion_mailbox_actor_dedupe_idx
    ON companion_mailbox (customer_id, recipient, dedupe_key) WHERE session_id IS NULL;
CREATE INDEX companion_mailbox_pending_idx
    ON companion_mailbox (customer_id, session_id, expires_at);
CREATE INDEX companion_mailbox_recipient_idx
    ON companion_mailbox (customer_id, recipient, expires_at);

CREATE TABLE companion_deliveries (
    id                     BIGSERIAL PRIMARY KEY,
    customer_id            TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    mailbox_id             UUID NOT NULL,
    -- Minted once per emission attempt by the client; ack retries reuse it.
    attempt_id             UUID NOT NULL,
    seam                   TEXT NOT NULL CHECK (seam IN
                             ('stop', 'user-prompt', 'post-tool', 'post-tool-batch', 'push',
                              'mcp-rider', 'pi-input', 'pi-extension',
                              'codex-session-start', 'codex-user-prompt', 'codex-async',
                              'codex-stop')),
    outcome                TEXT NOT NULL CHECK (outcome IN
                             ('emitted', 'expired', 'canceled', 'unknown', 'unsupported')),
    -- 'ingest:<token>' or 'user:<uuid>' -- WHO delivered, never the recipient.
    delivering_credential  TEXT,
    receiving_instance     TEXT NOT NULL,
    harness_version        TEXT,
    session_state          TEXT CHECK (session_state IN ('active', 'idle')),
    -- Client wall clocks, informational; the monotonic duration below is the
    -- only exact local latency.
    client_received_at     TIMESTAMPTZ,
    client_emitted_at      TIMESTAMPTZ,
    receipt_to_emission_ms INTEGER CHECK (receipt_to_emission_ms IS NULL OR receipt_to_emission_ms >= 0),
    ack_received_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    evidence               JSONB NOT NULL DEFAULT '{}'::jsonb,
    serve_ledger_id        BIGINT,
    UNIQUE (customer_id, attempt_id),
    FOREIGN KEY (customer_id, mailbox_id) REFERENCES companion_mailbox (customer_id, id)
);
CREATE INDEX companion_deliveries_mailbox_idx
    ON companion_deliveries (customer_id, mailbox_id);

CREATE TABLE companion_claims (
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    mailbox_id   UUID NOT NULL,
    claimed_by   TEXT NOT NULL,
    lease_until  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, mailbox_id),
    FOREIGN KEY (customer_id, mailbox_id) REFERENCES companion_mailbox (customer_id, id)
);

ALTER TABLE companion_mailbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_mailbox FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_select ON companion_mailbox
    FOR SELECT
    USING (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation_insert ON companion_mailbox
    FOR INSERT
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));
REVOKE UPDATE, DELETE ON companion_mailbox FROM PUBLIC;

ALTER TABLE companion_deliveries ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_deliveries FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_select ON companion_deliveries
    FOR SELECT
    USING (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation_insert ON companion_deliveries
    FOR INSERT
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));
REVOKE UPDATE, DELETE ON companion_deliveries FROM PUBLIC;

ALTER TABLE companion_claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_claims FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON companion_claims
    USING (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));
"""

DOWN = """
DROP TABLE IF EXISTS companion_claims;
DROP TABLE IF EXISTS companion_deliveries;
DROP TABLE IF EXISTS companion_mailbox;
"""


def upgrade() -> None:
    op.execute(UP)


def downgrade() -> None:
    op.execute(DOWN)

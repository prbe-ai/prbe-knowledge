"""Companion observations: did the model-readable context actually carry the card?

Spec §8 keeps four facts apart: emitted (actuator wrote to the harness),
harness-accepted (the harness said it applied the output), observed in
context (the trial nonce was seen where the model reads), and behaviour.
The first two ride on the ack. The third arrives LATER than the ack -- the
tap finds the card body in the transcript it already tails, seconds after
the hook returned -- and the deliveries table is append-only with one row
per attempt, so it cannot be updated in place.

`companion_observations` is the append-only sidecar: at most one row per
attempt (first write wins), keyed to the delivery row it qualifies.
`observed = true` means the body was found in model-readable context;
`observed = false` is a bounded search that finished without finding it --
an emitted-but-never-seen attempt is the most informative row in the fault
catalog and must not be confused with "nobody looked".

Same RLS posture as deliveries: FORCE + SELECT/INSERT only; UPDATE/DELETE
revoked from PUBLIC. The report counts an attempt as observed when EITHER
the ack's evidence said so (an actuator that knew at ack time) OR an
observation row says so.
"""

from alembic import op

revision = "0130_companion_observations"
down_revision = "0129_companion_mailbox"
branch_labels = None
depends_on = None

UP = """
CREATE TABLE companion_observations (
    id                 BIGSERIAL PRIMARY KEY,
    customer_id        TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    mailbox_id         UUID NOT NULL,
    attempt_id         UUID NOT NULL,
    -- Which fact this row asserts. 'context': the model-readable context
    -- carried the card. 'behaviour': what the model did with it afterwards,
    -- which only a local observer reading the transcript can say.
    kind               TEXT NOT NULL DEFAULT 'context' CHECK (kind IN ('context', 'behaviour')),
    -- context:   true = the card body / trial nonce was seen where the model
    --            reads; false = a bounded search ended without finding it.
    -- behaviour: true = the model followed the card (outcome 'followed').
    observed           BOOLEAN NOT NULL,
    -- behaviour rows only: what the model did with the card.
    outcome            TEXT CHECK (outcome IS NULL OR outcome IN
                         ('followed', 'ignored', 'contradicted', 'overridden')),
    -- Who looked: 'tap:<device>:<harness>:<version>', 'driver:<user>', ...
    observer           TEXT NOT NULL CHECK (length(observer) BETWEEN 1 AND 200),
    observed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- The observer's own wall clock, kept apart from the server's.
    client_observed_at TIMESTAMPTZ,
    evidence           JSONB NOT NULL DEFAULT '{}',
    CHECK ((kind = 'behaviour') = (outcome IS NOT NULL)),
    CHECK (kind <> 'behaviour' OR observed = (outcome = 'followed')),
    -- One verdict per attempt PER FACT: first write wins within a kind.
    UNIQUE (customer_id, attempt_id, kind),
    FOREIGN KEY (customer_id, attempt_id) REFERENCES companion_deliveries (customer_id, attempt_id),
    FOREIGN KEY (customer_id, mailbox_id) REFERENCES companion_mailbox (customer_id, id)
);
CREATE INDEX companion_observations_mailbox_idx
    ON companion_observations (customer_id, mailbox_id);

ALTER TABLE companion_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_observations FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_select ON companion_observations
    FOR SELECT
    USING (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation_insert ON companion_observations
    FOR INSERT
    WITH CHECK (customer_id = current_setting('app.current_customer_id', true));
REVOKE UPDATE, DELETE ON companion_observations FROM PUBLIC;
"""

DOWN = """
DROP TABLE IF EXISTS companion_observations;
"""


def upgrade() -> None:
    op.execute(UP)


def downgrade() -> None:
    op.execute(DOWN)

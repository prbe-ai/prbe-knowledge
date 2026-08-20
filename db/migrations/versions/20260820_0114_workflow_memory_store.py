"""Workflow memory Phase 0: the procedure store.

Creates the five tables behind the declared-rule wedge: situations, clauses,
clause_situation_edges, clause_evidence, serve_ledger. All five carry data in
v0.

`procedures` / `procedure_clauses` are deliberately NOT created. Nothing writes
them in v0, and their eventual shape is unknown -- today's sketch has no
conditional branching, no situation narrowing, and no pin to a clause version
despite `clauses.version` existing -- so creating them now buys an empty table
and a near-certain second migration anyway.

DELIBERATELY ABSENT
-------------------
* ``clause_evidence`` stores evidence BY REFERENCE, and that is enforced on
  TWO axes, not one. (a) No quote/text column exists. (b) ``source_ref`` --
  unconstrained JSONB, which is where a quote would actually land -- carries
  a CHECK refusing quote-shaped keys. An earlier revision of this migration
  claimed the absent column was sufficient; it was not. A full quote, or an
  entire private DM, fit inside ``source_ref`` with no DDL change at all.
  Both axes have tests; evidence resolves at view time through the viewer's
  ACL, and a baked-in quote escapes that check.
* ``situation_occurrences`` and ``conformance`` are NOT created. They need
  Phase 1's detectors and nightly join; empty tables invite premature writes.

Four of the five tables get ENABLE + FORCE ROW LEVEL SECURITY and a
tenant_isolation policy with BOTH USING and WITH CHECK, per the audit intent
of migration 0067 -- USING alone hides another tenant's rows on read but
still lets a buggy writer file a row under the wrong customer.

``serve_ledger`` is the exception and gets FOR SELECT + FOR INSERT policies
only. It is an append-only exposure log; under FORCE RLS the ABSENCE of an
UPDATE/DELETE policy is a deny, so a tenant cannot rewrite its own history.
Backdating (``SET served_at = '2001-01-01'``) is worse than deletion -- it
silently inverts the taint join the table exists for.
"""

from alembic import op

revision = "0114_workflow_memory_store"
down_revision = "0113_duplicate_lane_snapshot"
branch_labels = None
depends_on = None

#: Creation order matters (FKs); RLS is applied to all of them afterwards.
_TABLES = (
    "situations",
    "clauses",
    "clause_situation_edges",
    "clause_evidence",
    "serve_ledger",
)

#: The tables a tenant may UPDATE and DELETE within, i.e. everything except the
#: append-only serve_ledger, which gets FOR SELECT + FOR INSERT policies only.
_MUTABLE_TABLES = tuple(t for t in _TABLES if t != "serve_ledger")


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE situations (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            slug            TEXT NOT NULL,
            label           TEXT NOT NULL,
            description     TEXT NOT NULL,
            example_signals JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (customer_id, slug),
            -- Redundant for uniqueness (id is already the PK), but REQUIRED as
            -- the target of the composite FKs below: Postgres will only
            -- reference columns that carry a unique constraint.
            UNIQUE (customer_id, id)
        )
        """
    )

    op.execute(
        """
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
            UNIQUE (customer_id, id)          -- composite-FK target; see situations
        )
        """
    )
    op.execute("CREATE INDEX clauses_customer_status_idx ON clauses (customer_id, status)")

    op.execute(
        """
        -- COMPOSITE FKs, not simple ones. Postgres referential-integrity checks
        -- bypass row security by design, and the tenant policy only inspects THIS
        -- row's customer_id -- so a simple `REFERENCES clauses(id)` lets tenant A
        -- attach an edge to tenant B's clause id. That succeeds iff the uuid
        -- exists in any tenant, which is a cross-tenant existence oracle, and it
        -- corrupts any later join that runs outside a tenant GUC. Keying the FK
        -- on (customer_id, id) makes the mismatch unrepresentable.
        -- Caveat: "unrepresentable" is slightly too strong. The FK route is
        -- closed, but an explicit-id insert of another tenant's uuid still
        -- errors DIFFERENTLY from a random uuid via the single-column PK, so a
        -- narrow existence oracle survives at the primary key.
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
        )
        """
    )
    op.execute(
        "CREATE INDEX cse_situation_idx ON clause_situation_edges (customer_id, situation_id)"
    )

    # NO quote/text column here, AND no quote-shaped key inside source_ref.
    # See module docstring: the column-absence claim alone was wrong.
    op.execute(
        """
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
            -- NO DEFAULT, deliberately. A default of FALSE fails OPEN: a writer
            -- that forgets the taint computation records the evidence as clean,
            -- and taint-excluded support is a stated non-negotiable. With NOT
            -- NULL and no default, an omission is a constraint violation and
            -- every write has to state its taint.
            exposure_tainted BOOLEAN NOT NULL,
            ts               TIMESTAMPTZ NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # Leads with the tenant column, matching the composite FK.
    op.execute(
        "CREATE INDEX clause_evidence_clause_idx ON clause_evidence (customer_id, clause_id)"
    )

    op.execute(
        """
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
        )
        """
    )
    op.execute("CREATE INDEX serve_ledger_session_idx ON serve_ledger (customer_id, session_id)")
    op.execute(
        "CREATE INDEX serve_ledger_actor_idx ON serve_ledger (customer_id, actor_ref, served_at)"
    )
    # Phase 1's taint join asks "which clauses were served into this session",
    # which is a containment query over the array. Without GIN that is a seq
    # scan over a table that grows with every request.
    op.execute("CREATE INDEX serve_ledger_clause_ids_idx ON serve_ledger USING GIN (clause_ids)")

    # Postgres has no ON UPDATE for column defaults, so an `updated_at DEFAULT
    # now()` never advances past insert time. §3.3.1's reclassification story and
    # the version/lineage handling both read it as meaningful, so it needs a
    # trigger. Function/trigger shape copied from customers_fill_r2_bucket
    # (db/schema.sql:52-64), the house convention.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION wfmem_touch_updated_at() RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at := now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in ("situations", "clauses"):
        op.execute(
            f"""
            CREATE TRIGGER {table}_touch_updated_at_trg
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION wfmem_touch_updated_at();
            """
        )

    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

    # Everything but serve_ledger: one FOR ALL policy, both halves.
    for table in _MUTABLE_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                USING      (customer_id = current_setting('app.current_customer_id', true))
                WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
            """
        )

    # serve_ledger is append-only: SELECT and INSERT policies only. Under FORCE
    # RLS a command with no applicable policy matches no rows, so the ABSENCE of
    # an UPDATE/DELETE policy is the deny -- a tenant cannot backdate or erase
    # its own exposure history. The REVOKE is belt-and-braces for any role that
    # would otherwise inherit the privilege through PUBLIC.
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON serve_ledger")
    op.execute(
        """
        CREATE POLICY tenant_isolation_select ON serve_ledger
            FOR SELECT
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )
    op.execute(
        """
        CREATE POLICY tenant_isolation_insert ON serve_ledger
            FOR INSERT
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )
    op.execute("REVOKE UPDATE, DELETE ON serve_ledger FROM PUBLIC")


def downgrade() -> None:
    # RESTRICT in dependency order, not CASCADE: CASCADE makes the ordering
    # pointless and would silently drop anything a later migration attached
    # (views, Phase 1 FKs). If a drop fails here, something depends on these
    # tables and that is worth knowing, not worth steamrolling.
    for table in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table} RESTRICT")
    op.execute("DROP FUNCTION IF EXISTS wfmem_touch_updated_at()")

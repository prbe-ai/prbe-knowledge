"""kb_provision_tenant(): one implementation of "give this team its partitions".

WHY. research-os creates kb tenants (kb_mirror.ensure_customer from the index
relay, claim_customer from team creation) by inserting into `customers` and
nothing else, so a new team's rows land in the DEFAULT partition of `chunks`:
`probe-bench-temp` has had 34,299 chunks there since 2026-09-26. Only the
engine's own create_customer made a partition, and research-os cannot import
it. A SQL function is the one thing both repositories can call.

WHY ITS OWN TRANSACTION, NEVER INSIDE THE CUSTOMERS INSERT. ATTACH clones the
parent's `customer_id` foreign key onto the new partition, which takes SHARE ROW
EXCLUSIVE on `customers`. A transaction that has just inserted a customer holds
ROW EXCLUSIVE on it; two such transactions each wait for the other's lock
(deadlock). So callers commit the customer first and call this in a fresh
transaction; the advisory lock below serializes every partition DDL against the
next, so provisioning calls queue instead of deadlocking each other. The
3 s lock_timeout bounds any wait behind ordinary traffic; the caller retries
and the pg_search guardian reconciles every minute as the backstop.

WHAT IT DOES, per partitioned parent (discovered from the catalog: every public
table LIST-partitioned on customer_id, FK-referenced parents first):
    CREATE TABLE <name> (LIKE parent INCLUDING ALL)   -- no parent lock
    ALTER TABLE parent ATTACH PARTITION <name> ...    -- SHARE UPDATE EXCLUSIVE
    ENABLE + FORCE RLS and a copy of EVERY parent policy on the leaf
The leaf policies close the door a query naming the partition directly would
otherwise open (LIKE copies no policies). Idempotent: a team that already has
an attached partition is skipped, found by its partition BOUND, not its name.

NAMES. kb_partition_name() is the creation name, mirrored in Python by
engine.shared.partitions.partition_name_for (a test pins them equal): a short
per-parent prefix + slug + 8 hex of sha256(customer_id), at most 58 bytes, so
the hash is never truncated by the 63-byte identifier limit. Partitions created
before this migration carry sha1 names; nothing looks a partition up by name
any more, so both coexist.
"""

from alembic import op

revision = "0144_kb_provision_tenant"
down_revision = "0143_canonical_and_purge_idx"
branch_labels = None
depends_on = None

# db/schema.sql carries these two bodies byte for byte (tests pin it).
PARTITION_NAME_SQL = r"""
CREATE OR REPLACE FUNCTION kb_partition_name(parent text, tenant text)
RETURNS text
LANGUAGE sql IMMUTABLE STRICT
AS $$
    SELECT (CASE regexp_replace(parent, '__conv$', '')
                WHEN 'chunks' THEN 'chunks_p_'
                WHEN 'documents' THEN 'doc_p_'
                WHEN 'usage_events' THEN 'ue_p_'
                WHEN 'graph_nodes' THEN 'gn_p_'
                WHEN 'graph_edges' THEN 'ge_p_'
                WHEN 'graph_node_provenance' THEN 'gnp_p_'
            END)
        || left(trim(both '_' from regexp_replace(lower(tenant), '[^a-z0-9]+', '_', 'g')), 40)
        || '_'
        || left(encode(sha256(convert_to(tenant, 'UTF8')), 'hex'), 8)
$$;
"""

PROVISION_SQL = r"""
CREATE OR REPLACE FUNCTION kb_provision_tenant(tenant text)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    parent record;
    pol record;
    part text;
    roles text;
    created integer := 0;
BEGIN
    IF tenant IS NULL OR tenant !~ '^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$' THEN
        RAISE EXCEPTION 'kb_provision_tenant: refusing unsafe customer_id %', tenant
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    -- A typo'd or deleted id would get partitions no row can ever reach.
    IF NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = tenant) THEN
        RAISE EXCEPTION 'kb_provision_tenant: no customer %', tenant
            USING ERRCODE = 'foreign_key_violation';
    END IF;
    -- Fast path, no lock: the relay calls this once per tenant per process,
    -- and nearly always there is nothing to do.
    IF NOT EXISTS (
        SELECT 1
        FROM pg_partitioned_table pt
        JOIN pg_class c ON c.oid = pt.partrelid
        JOIN pg_namespace ns ON ns.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = pt.partattrs[0]
        WHERE ns.nspname = 'public' AND pt.partstrat = 'l' AND pt.partnatts = 1
          AND a.attname = 'customer_id'
          AND NOT EXISTS (
              SELECT 1
              FROM pg_inherits h
              JOIN pg_class leaf ON leaf.oid = h.inhrelid
              WHERE h.inhparent = c.oid
                AND pg_get_expr(leaf.relpartbound, leaf.oid) = format('FOR VALUES IN (%L)', tenant)
          )
    ) THEN
        RETURN 0;
    END IF;
    -- One lock for every partition DDL (provision, conversion swap, detach,
    -- drop): they queue behind each other instead of deadlocking.
    PERFORM pg_advisory_xact_lock(hashtextextended('kb_partition_ddl', 0));
    PERFORM set_config('lock_timeout', '3s', true);

    FOR parent IN
        SELECT c.oid, c.relname
        FROM pg_partitioned_table pt
        JOIN pg_class c ON c.oid = pt.partrelid
        JOIN pg_namespace ns ON ns.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = pt.partattrs[0]
        WHERE ns.nspname = 'public'
          AND pt.partstrat = 'l'
          AND pt.partnatts = 1
          AND a.attname = 'customer_id'
        -- Parents other partitioned parents reference come first.
        ORDER BY (
            SELECT count(*) FROM pg_constraint k
            WHERE k.conrelid = c.oid AND k.contype = 'f'
              AND k.confrelid IN (SELECT partrelid FROM pg_partitioned_table)
        ), c.relname
    LOOP
        IF EXISTS (
            SELECT 1
            FROM pg_inherits h
            JOIN pg_class leaf ON leaf.oid = h.inhrelid
            WHERE h.inhparent = parent.oid
              AND pg_get_expr(leaf.relpartbound, leaf.oid) = format('FOR VALUES IN (%L)', tenant)
        ) THEN
            CONTINUE;
        END IF;

        part := kb_partition_name(parent.relname, tenant);
        IF part IS NULL THEN
            RAISE EXCEPTION 'kb_provision_tenant: no partition prefix for parent %', parent.relname;
        END IF;
        IF to_regclass(format('public.%I', part)) IS NOT NULL THEN
            -- A detached-but-not-dropped leftover, or a half-built table from a
            -- failed run. Never adopt it: its rows could be another team's.
            RAISE EXCEPTION 'kb_provision_tenant: % exists but is not attached to %; resolve it by hand',
                part, parent.relname;
        END IF;

        EXECUTE format('CREATE TABLE public.%I (LIKE public.%I INCLUDING ALL)', part, parent.relname);
        EXECUTE format('ALTER TABLE public.%I ATTACH PARTITION public.%I FOR VALUES IN (%L)',
                       parent.relname, part, tenant);
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', part);
        EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', part);

        FOR pol IN
            SELECT p.polname, p.polcmd, p.polpermissive, p.polroles,
                   pg_get_expr(p.polqual, p.polrelid) AS qual,
                   pg_get_expr(p.polwithcheck, p.polrelid) AS withcheck
            FROM pg_policy p
            WHERE p.polrelid = parent.oid
        LOOP
            IF pol.polroles = '{0}'::oid[] THEN
                roles := 'PUBLIC';
            ELSE
                SELECT string_agg(quote_ident(r.rolname), ', ') INTO roles
                FROM pg_roles r WHERE r.oid = ANY (pol.polroles);
            END IF;
            EXECUTE format(
                'CREATE POLICY %I ON public.%I AS %s FOR %s TO %s%s%s',
                pol.polname, part,
                CASE WHEN pol.polpermissive THEN 'PERMISSIVE' ELSE 'RESTRICTIVE' END,
                CASE pol.polcmd WHEN 'r' THEN 'SELECT' WHEN 'a' THEN 'INSERT'
                                WHEN 'w' THEN 'UPDATE' WHEN 'd' THEN 'DELETE' ELSE 'ALL' END,
                roles,
                CASE WHEN pol.qual IS NULL THEN '' ELSE format(' USING (%s)', pol.qual) END,
                CASE WHEN pol.withcheck IS NULL THEN '' ELSE format(' WITH CHECK (%s)', pol.withcheck) END
            );
        END LOOP;

        created := created + 1;
    END LOOP;
    RETURN created;
END
$$;
"""


def upgrade() -> None:
    op.execute(PARTITION_NAME_SQL)
    op.execute(PROVISION_SQL)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS kb_provision_tenant(text)")
    op.execute("DROP FUNCTION IF EXISTS kb_partition_name(text, text)")

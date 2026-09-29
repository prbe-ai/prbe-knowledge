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
(deadlock, reproduced on the T9 rig). So callers commit the customer first and
call this in a fresh transaction, and the function REFUSES when its transaction
holds ROW EXCLUSIVE on `customers` (research-os calls it straight from SQL, so a
Python-side check alone would leave that path unguarded).

LOCKS, IN THIS ORDER, EVERY TIME. `lock_timeout = 3s` is a SET clause of the
function, so it is in force before the first wait -- the advisory lock
included -- and ends with the call instead of leaking into the caller's
transaction. Then the `kb_partition_ddl` advisory lock, then the tables.
drop_tenant_partition and split_default take the advisory lock FIRST too: a
split that held DEFAULT before asking for the advisory lock deadlocked against
a provision holding the advisory lock and asking for DEFAULT (ATTACH scans it).
A caller that times out retries; the pg_search guardian reconciles every minute.

WHAT IT DOES, per partitioned parent (discovered from the catalog: every public
table LIST-partitioned on customer_id that kb_partition_name() knows, FK-
referenced parents first; an unknown one, e.g. a conversion's scratch table, is
skipped, not fatal):
    refuse if the tenant already has rows in the parent's DEFAULT partition
        (ATTACH would fail after locking DEFAULT for up to 3 s; split_default()
        is the attended fix, and the guardian's DEFAULT alarm names the tenant)
    CREATE TABLE <name> (LIKE parent INCLUDING ALL)   -- no parent lock
    owner and grants set to exactly the parent's      -- LIKE copies neither
    ALTER TABLE parent ATTACH PARTITION <name> ...    -- SHARE UPDATE EXCLUSIVE
    ENABLE + FORCE RLS and a copy of EVERY parent policy on the leaf
The leaf policies close the door a query naming the partition directly would
otherwise open (LIKE copies no policies). The owner matters because every later
migration that ALTERs the parent recurses into each leaf as the parent's owner:
a leaf created by a superuser by hand would crash-loop the migrate Job.
Idempotent: a team that already has an attached partition is skipped, found by
its partition BOUND, not its name.

NAMES. kb_partition_name() is the creation name, mirrored in Python by
engine.shared.partitions.partition_name_for (a test pins them equal): a short
per-parent prefix + slug + 8 hex of sha256(customer_id), at most 58 bytes, so
the hash is never truncated by the 63-byte identifier limit. Partitions created
before this migration carry sha1 names; lookups try both names and fall back to
the bound, so both coexist.
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
SET search_path = pg_catalog
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
SET search_path = pg_catalog, public, pg_temp
SET lock_timeout = '3s'
SET app.current_customer_id = ''
AS $$
DECLARE
    todo oid[];
    parent record;
    pol record;
    grant_row record;
    part text;
    dflt text;
    resident boolean;
    roles text;
    created integer := 0;
BEGIN
    IF tenant IS NULL OR tenant !~ '^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$' THEN
        RAISE EXCEPTION 'kb_provision_tenant: refusing unsafe customer_id %', tenant
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    -- A typo'd or deleted id would get partitions no row can ever reach.
    IF NOT EXISTS (SELECT 1 FROM public.customers c WHERE c.customer_id = tenant) THEN
        RAISE EXCEPTION 'kb_provision_tenant: no customer %', tenant
            USING ERRCODE = 'foreign_key_violation';
    END IF;

    -- Parents this tenant still lacks a partition on, referenced parents
    -- first. Read without a lock: the relay calls this once per tenant per
    -- process and nearly always there is nothing to do.
    SELECT array_agg(c.oid ORDER BY (
               SELECT count(*) FROM pg_constraint k
               WHERE k.conrelid = c.oid AND k.contype = 'f'
                 AND k.confrelid IN (SELECT partrelid FROM pg_partitioned_table)
           ), c.relname)
    INTO todo
    FROM pg_partitioned_table pt
    JOIN pg_class c ON c.oid = pt.partrelid
    JOIN pg_namespace ns ON ns.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = pt.partattrs[0]
    WHERE ns.nspname = 'public'
      AND pt.partstrat = 'l'
      AND pt.partnatts = 1
      AND a.attname = 'customer_id'
      AND public.kb_partition_name(c.relname, tenant) IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM pg_inherits h
          JOIN pg_class leaf ON leaf.oid = h.inhrelid
          WHERE h.inhparent = c.oid
            AND pg_get_expr(leaf.relpartbound, leaf.oid) = format('FOR VALUES IN (%L)', tenant)
      );
    IF todo IS NULL THEN
        RETURN 0;
    END IF;

    -- Inside the transaction that wrote `customers`, the ATTACH below would
    -- deadlock against a second such transaction. Refuse instead.
    IF EXISTS (
        SELECT 1 FROM pg_locks l
        WHERE l.pid = pg_backend_pid()
          AND l.locktype = 'relation'
          AND l.relation = 'public.customers'::regclass
          AND l.mode = 'RowExclusiveLock'
    ) THEN
        RAISE EXCEPTION 'kb_provision_tenant: called inside a transaction that wrote customers; commit it first'
            USING ERRCODE = 'active_sql_transaction';
    END IF;

    -- One lock for every partition DDL (provision, drop, split): they queue
    -- behind each other instead of deadlocking. Bounded by lock_timeout.
    PERFORM pg_advisory_xact_lock(hashtextextended('kb_partition_ddl', 0));

    -- Rows already in a DEFAULT partition: ATTACH would lock DEFAULT for up to
    -- the lock_timeout and then fail its scan. split_default() is the fix.
    -- UNDER the advisory lock: reading DEFAULT takes a table lock, and one
    -- taken before the advisory lock deadlocks against a provision holding
    -- it and asking for DEFAULT (ATTACH). The tenant GUC is function-scoped
    -- (SET clause above): DEFAULT is under FORCE RLS on a converted plane,
    -- and without it no row is visible.
    PERFORM set_config('app.current_customer_id', tenant, true);
    FOR parent IN SELECT c.oid, c.relname FROM pg_class c WHERE c.oid = ANY (todo) LOOP
        SELECT format('public.%I', d.relname) INTO dflt
        FROM pg_inherits h
        JOIN pg_class d ON d.oid = h.inhrelid
        WHERE h.inhparent = parent.oid
          AND pg_get_expr(d.relpartbound, d.oid) = 'DEFAULT';
        IF dflt IS NOT NULL THEN
            EXECUTE format('SELECT EXISTS (SELECT 1 FROM ONLY %s WHERE customer_id = $1)', dflt)
                INTO resident USING tenant;
            IF resident THEN
                RAISE EXCEPTION 'kb_provision_tenant: % already has rows in %; run split_default() for it',
                    tenant, dflt
                    USING ERRCODE = 'object_not_in_prerequisite_state';
            END IF;
        END IF;
    END LOOP;

    FOR parent IN
        SELECT c.oid, c.relname, c.relowner, c.relacl
        FROM unnest(todo) WITH ORDINALITY AS u(oid, ord)
        JOIN pg_class c ON c.oid = u.oid
        ORDER BY u.ord
    LOOP
        -- Provisioned by another session while this one waited for the lock.
        IF EXISTS (
            SELECT 1
            FROM pg_inherits h
            JOIN pg_class leaf ON leaf.oid = h.inhrelid
            WHERE h.inhparent = parent.oid
              AND pg_get_expr(leaf.relpartbound, leaf.oid) = format('FOR VALUES IN (%L)', tenant)
        ) THEN
            CONTINUE;
        END IF;

        part := public.kb_partition_name(parent.relname, tenant);
        IF to_regclass(format('public.%I', part)) IS NOT NULL THEN
            -- A detached-but-not-dropped leftover, or a half-built table from a
            -- failed run. Never adopt it: its rows could be another team's.
            RAISE EXCEPTION 'kb_provision_tenant: % exists but is not attached to %; resolve it by hand',
                part, parent.relname;
        END IF;

        EXECUTE format('CREATE TABLE public.%I (LIKE public.%I INCLUDING ALL)', part, parent.relname);
        IF (SELECT relowner FROM pg_class WHERE oid = format('public.%I', part)::regclass)
                <> parent.relowner THEN
            EXECUTE format('ALTER TABLE public.%I OWNER TO %s', part, parent.relowner::regrole);
        END IF;
        -- Exactly the parent's grants: first drop what the creating role's
        -- default privileges put on the new table.
        FOR grant_row IN
            SELECT DISTINCT g.grantee
            FROM pg_class leaf, aclexplode(leaf.relacl) g
            WHERE leaf.oid = format('public.%I', part)::regclass
              AND g.grantee <> leaf.relowner
        LOOP
            EXECUTE format('REVOKE ALL ON public.%I FROM %s', part,
                CASE WHEN grant_row.grantee = 0 THEN 'PUBLIC' ELSE grant_row.grantee::regrole::text END);
        END LOOP;
        FOR grant_row IN
            SELECT g.privilege_type, g.grantee, g.is_grantable
            FROM aclexplode(parent.relacl) g
            WHERE g.grantee <> parent.relowner
        LOOP
            EXECUTE format('GRANT %s ON public.%I TO %s%s',
                grant_row.privilege_type, part,
                CASE WHEN grant_row.grantee = 0 THEN 'PUBLIC' ELSE grant_row.grantee::regrole::text END,
                CASE WHEN grant_row.is_grantable THEN ' WITH GRANT OPTION' ELSE '' END);
        END LOOP;

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

REVOKE ALL ON FUNCTION kb_provision_tenant(text) FROM PUBLIC;
"""


def upgrade() -> None:
    op.execute(PARTITION_NAME_SQL)
    op.execute(PROVISION_SQL)


def downgrade() -> None:
    # Kept on purpose. research-os calls kb_provision_tenant() whenever it
    # exists (kb_mirror checks with to_regprocedure), and the engine code that
    # calls it may still be running while the schema steps back; both
    # functions are inert until called.
    pass

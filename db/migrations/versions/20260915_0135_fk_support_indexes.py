"""Index the three foreign keys whose absence made a tenant purge impossible.

An FK is enforced in BOTH directions. Postgres indexes the REFERENCED side for
you (it must be unique) and indexes the REFERENCING side for nobody -- so every
delete of a referenced row runs a check against the child table, and with no
index that check is a SEQUENTIAL SCAN, once per deleted row.

Measured on the research plane, 2026-09-15, while draining a deleted tenant:

    documents.ingestion_event_id -> ingestion_events   ON DELETE SET NULL
        documents is 4 GB / 928,402 rows, so deleting ONE ingestion_events row
        scanned all of it. `pg_stat_user_tables` had recorded 277,693 such scans
        and 241,743,618,971 tuples read. 3,869 rows were unpurgeable at ANY
        batch size: research-os's purge reaper spent 45 minutes timing out on
        them at 300s a time, having already drained 161,031 rows from every
        other table. With the index the same purge finished in 9 seconds.

        Worth noting for whoever revisits this: ingestion_event_id is NULL in
        100% of those rows. Every one of those scans proved a negative about a
        column nothing populates, and the FK may not be earning its keep at all
        -- but dropping a constraint is a decision, and an index is not, so this
        migration only does the part that is safe either way.

    graph_edges.from_node_id -> graph_nodes            ON DELETE CASCADE
    graph_edges.to_node_id   -> graph_nodes            ON DELETE CASCADE
        Same shape, not yet hit. `idx_graph_edges_from` and `idx_graph_edges_to`
        look like they cover these and do NOT: both lead with `customer_id`, so
        neither can serve the RI check's bare `from_node_id = $1`. Deleting a
        tenant's 7,739 graph_nodes rows would have fired 15,478 scans of a
        193,165-row table immediately after the above was fixed.

A composite FK needs an index whose LEADING columns match its column list in
order; several kb tables already satisfy that through their primary key
(`session_batch_receipts`' 4-column PK covers its 3-column FK, for one), which
is why this migration adds three indexes and not every FK in the schema.

PLAIN CREATE INDEX, not CONCURRENTLY, and IF NOT EXISTS: the same convention as
0105/0124. All three were built attended via CIC on the research plane on
2026-09-15, so this is a NO-OP there; the plain form is for fresh installs and
self-hosts, whose row counts make the write lock a non-event. The INVALID-leftover
guard is why `IF NOT EXISTS` alone is not enough -- it matches on NAME, so an
interrupted CONCURRENTLY build leaves a corpse the planner ignores and this
migration would record itself applied against it.
"""

from alembic import op

revision = "0135_fk_support_indexes"
down_revision = "0134_chunks_tenant_unique"
branch_labels = None
depends_on = None

#: (index name, table, column)
_INDEXES = (
    ("idx_documents_ingestion_event_id", "documents", "ingestion_event_id"),
    ("idx_graph_edges_from_node_id_fk", "graph_edges", "from_node_id"),
    ("idx_graph_edges_to_node_id_fk", "graph_edges", "to_node_id"),
)


def upgrade() -> None:
    for name, table, column in _INDEXES:
        # Drop an INVALID leftover FIRST -- see the module docstring.
        op.execute(
            f"""
            DO $$
            DECLARE
                invalid_name text;
            BEGIN
                SELECT c.relname INTO invalid_name
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indexrelid
                WHERE c.relname = '{name}'
                  AND NOT i.indisvalid;
                IF invalid_name IS NOT NULL THEN
                    RAISE NOTICE 'dropping INVALID %', invalid_name;
                    EXECUTE 'DROP INDEX ' || quote_ident(invalid_name);
                END IF;
            END $$
            """
        )
        op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})")


def downgrade() -> None:
    for name, _table, _column in _INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")

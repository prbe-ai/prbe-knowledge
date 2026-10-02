"""A live chunk is exactly one whose last_seen_version is the sentinel.

CHECK ((valid_to IS NULL) = (last_seen_version = 2147483647)) on `chunks`.

WHY. The BM25 pool (engine/retrieval/retrievers/bm25.py) filtered live rows
with `c.valid_to IS NULL`. `valid_to` is not a BM25 index field, so pg_search
applied it as a heap filter: TopK fetched every closed version of every
matching chunk and threw it away (30-day mean of that statement: 4.1 s, 888 k
buffers per call). `last_seen_version` IS an index field, and every writer
already opens a live row at LIVE_CHUNK_LAST_SEEN and closes it below it
(#604). With this constraint the two predicates select the same rows, so the
pool can ask the index for `last_seen_version = 2147483647` instead (T9 rig:
4.2-8.6x fewer buffers, 2.3-4.5x faster, same top-50). Plan T8, D15/D75.

PRECONDITION. Rows written before #604 kept an exact last_seen_version while
live (915 k of ~990 k live rows on the research plane on 2026-10-02). They
were rewritten by an attended, throttled backfill before this migration ran;
VALIDATE fails, and the deploy rolls back, if any remain.

LOCKS. ADD ... NOT VALID takes ACCESS EXCLUSIVE on the parent and every
partition for a moment (it scans nothing). It is tried with a 2 s
lock_timeout and retried, so it never queues long behind a search; each
statement runs in its own transaction (autocommit) so that brief lock is
released before VALIDATE, which scans every row under SHARE UPDATE EXCLUSIVE
only (reads and writes continue). Idempotent: an existing constraint is not
added again, and a validated one is not validated again.
"""

import time

import sqlalchemy as sa
from alembic import op

revision = "0145_chunks_live_sentinel"
down_revision = "0144_kb_provision_tenant"
branch_labels = None
depends_on = None

CONSTRAINT = "chunks_live_sentinel_chk"
CHECK_EXPR = "((valid_to IS NULL) = (last_seen_version = 2147483647))"

ADD_ATTEMPTS = 60
ADD_RETRY_SECONDS = 3.0
LOCK_NOT_AVAILABLE = "55P03"


def _state(bind):
    """(present, validated) for the constraint on the parent."""
    row = bind.execute(
        sa.text(
            "SELECT convalidated FROM pg_constraint "
            "WHERE conrelid = 'public.chunks'::regclass AND conname = :name"
        ),
        {"name": CONSTRAINT},
    ).first()
    return (row is not None, bool(row and row.convalidated))


def run(bind, *, attempts: int = ADD_ATTEMPTS, retry_seconds: float = ADD_RETRY_SECONDS) -> None:
    """The whole migration on an autocommit connection (tests call this)."""
    try:
        present, validated = _state(bind)
        if not present:
            bind.execute(sa.text("SET lock_timeout = '2s'"))
            for attempt in range(1, attempts + 1):
                try:
                    bind.execute(
                        sa.text(
                            f"ALTER TABLE public.chunks ADD CONSTRAINT {CONSTRAINT} "
                            f"CHECK {CHECK_EXPR} NOT VALID"
                        )
                    )
                    break
                except sa.exc.OperationalError as exc:
                    if getattr(exc.orig, "sqlstate", None) != LOCK_NOT_AVAILABLE or attempt == attempts:
                        raise
                    time.sleep(retry_seconds)
        if not validated:
            # SHARE UPDATE EXCLUSIVE: waits only for DDL/VACUUM, never blocks
            # reads or writes, so an unbounded lock wait is acceptable here; the
            # hook's own deadline bounds the whole migration.
            bind.execute(sa.text("SET lock_timeout = 0"))
            bind.execute(sa.text(f"ALTER TABLE public.chunks VALIDATE CONSTRAINT {CONSTRAINT}"))
    finally:
        bind.execute(sa.text("RESET lock_timeout"))
    present, validated = _state(bind)
    if not (present and validated):
        raise RuntimeError(f"{CONSTRAINT} not in place after 0145: present={present} validated={validated}")


def upgrade() -> None:
    with op.get_context().autocommit_block():
        run(op.get_bind())


def downgrade() -> None:
    op.execute(f"ALTER TABLE public.chunks DROP CONSTRAINT IF EXISTS {CONSTRAINT}")

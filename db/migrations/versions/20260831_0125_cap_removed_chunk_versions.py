"""Align last_seen_version with valid_to on chunks removed mid-version.

The in-place resync path (incomplete docs update at the SAME version) marked
removed chunks `valid_to = NOW()` without capping `last_seen_version`, so the
version join (`d.version BETWEEN first_seen AND last_seen`) kept serving
content the producer had deleted wherever a query lacked the valid_to
predicate (TemporalMode.ALL, ad-hoc SQL) -- and any index or retention policy
keyed on version ranges would disagree with one keyed on valid_to. Measured
2026-08-30: 10,611 such rows, all transcript sources (codex 5,621 /
claude_code 4,996), timestamps down to minutes old. The producer fix lands
with this migration (normalizer chunk removal now caps with LEAST); this
backfills the rows already written.

PER-TENANT GUC LOOP, not a bare UPDATE, and this is load-bearing: `chunks`
and `documents` carry FORCE RLS and migrations run as `app` -- the owner,
whom FORCE subjects to the policy. A bare UPDATE compares customer_id against
a NULL GUC, matches zero rows, and reports success (0118 shipped exactly that
failure; 0119 established this pattern and the rule).

Sets last_seen_version = live_doc_version - 1: the chunk was removed FROM the
live version, so it was last part of a completed state one version earlier. A
chunk born and removed inside one version ends with first_seen >
last_seen_version -- an empty range that joins nothing, which is truthful.
Idempotent: rows already capped no longer satisfy the BETWEEN predicate.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0125_cap_removed_chunk_versions"
down_revision = "0124_hnsw_live_partial_index"
branch_labels = None
depends_on = None

_ALL_TENANTS = sa.text("SELECT customer_id FROM customers ORDER BY 1")

#: set_config(..., is_local => true), not SET: scopes to the migration's
#: transaction so nothing leaks through a pooled connection (0119's pattern).
_BIND_TENANT = sa.text(
    "SELECT set_config('app.current_customer_id', :customer_id, true)"
)

_CAP = sa.text(
    """
    UPDATE chunks c
    SET last_seen_version = d.version - 1
    FROM documents d
    WHERE d.customer_id = c.customer_id
      AND d.doc_id = c.doc_id
      AND d.valid_to IS NULL
      AND c.customer_id = :customer_id
      AND c.valid_to IS NOT NULL
      AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
    """
)


def upgrade() -> None:
    conn = op.get_bind()
    for (customer_id,) in conn.execute(_ALL_TENANTS):
        conn.execute(_BIND_TENANT, {"customer_id": customer_id})
        conn.execute(_CAP, {"customer_id": customer_id})


def downgrade() -> None:
    # Irreversible by design: the pre-fix ranges asserted membership in a
    # version state the producer had removed the content from. There is
    # nothing correct to restore.
    pass

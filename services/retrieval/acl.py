"""ACL enforcement layer.

Phase 0: pass-through stub. Data is captured in `acl_snapshots` during ingestion
— enforcement happens here once the Phase 1 Gate flips `ENFORCE_ACL = True`.

Keeping the layer in place (rather than adding it in Phase 1) means:
    - no structural refactor at the Phase 0 → 1 boundary
    - tests exercise the filter shape from day one
    - the query path is stable for operators

To enable enforcement in Phase 1, set ENFORCE_ACL=True and implement the
SQL filter in `_filter_with_acl`. The call site and filter semantics do not
need to change.
"""

from __future__ import annotations

from typing import Any

ENFORCE_ACL = False  # Phase 1 flips this on.


async def filter_by_acl(
    customer_id: str,
    requesting_user_id: str | None,
    hits: list[Any],
) -> list[Any]:
    """Filter retriever hits by the requesting user's ACL grants.

    Phase 0: no-op. Phase 1 will intersect against `acl_snapshots` rows
    whose `principal_id` matches the requester's membership (direct +
    transitive via group/channel/workspace).
    """
    if not ENFORCE_ACL:
        return hits
    if not requesting_user_id:
        # Strict mode: no requester identity = no data.
        return []
    return await _filter_with_acl(customer_id, requesting_user_id, hits)


async def _filter_with_acl(
    customer_id: str,
    requesting_user_id: str,
    hits: list[Any],
) -> list[Any]:
    """Phase 1 stub. Left unimplemented so flipping ENFORCE_ACL without the
    filter landing fails loudly rather than silently."""
    raise NotImplementedError(
        "ACL enforcement is enabled but filter is not implemented yet"
    )

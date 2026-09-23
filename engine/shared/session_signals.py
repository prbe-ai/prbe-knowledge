"""When has a coding-agent session ENDED? One rule, shared by the worker and the sweep.

Every batch of a session is appended, in arrival order, to one queue row's
`payload_s3_keys`. A session has ended when the newest thing on that pile is an
end signal:

    keys (arrival order)                            ended?
    ------------------------------------------      ------
    b0  b1  b2                                      no
    b0  b1  b2  <client finalize>                   yes  (client said goodbye)
    b0  b1  b2  <finalize.marker>                   yes  (sweep: idle 24h, nobody did)
    b0  b1  <finalize>  b3                          no   (resumed: live again)
    b0  b1  <finalize>  b3  <finalize.marker>       yes

Nothing is ever removed from the pile. An end signal that is no longer on top
simply stops counting, which is what "the session resumed" means.

WHY THIS MODULE EXISTS. The rule used to be spelled three ways in two files,
with two lifecycles: the worker counted a signal ANYWHERE in the pile and then
deleted it after mining, while the sweep counted a session as finished only
while the signal was still THERE. Each half was reasonable alone; together they
re-mined every idle protocol-1 session once a day (docs/plans/
extraction-spend-plan.md). Both sides now import this one definition.

Protocol 2 is order-checked by the server (`kb/session_receipts.accept` admits
batches strictly in sequence), so for a v2 stream "newest" means the highest
accepted `batch_seq`, and `session_streams.finalized` records whether that
batch was the finalize. The sweep's own marker is not a client claim and never
touches `session_streams`; it ends a v2 session only while it is the newest key.
"""

from __future__ import annotations

from enum import StrEnum

#: Suffix of the sweep's placeholder object, written under
#: `raw/<source>/<customer>/<session_id>/finalize.marker`.
CRON_MARKER_SUFFIX = "/finalize.marker"

#: Path segment of every protocol-2 batch key (`kb/session_receipts.accept`).
V2_KEY_SEGMENT = "/sessions-v2/"


class CompletedBy(StrEnum):
    """Which signal ended a session. Recorded on every extraction pass."""

    V2_FINALIZE = "v2_finalize"
    V1_CLIENT_FINALIZE = "v1_client_finalize"
    CRON_MARKER = "cron_marker"


def cron_marker_key(source: str, customer_id: str, session_id: str) -> str:
    """The sweep's marker key for one session. One per session, reused."""
    return f"raw/{source}/{customer_id}/{session_id}{CRON_MARKER_SUFFIX}"


def is_cron_marker_key(key: str) -> bool:
    return key.endswith(CRON_MARKER_SUFFIX)


def is_v2_key(key: str) -> bool:
    return V2_KEY_SEGMENT in key


def is_v1_client_finalize_key(key: str, session_id: str) -> bool:
    """A protocol-1 client finalize: `raw/<src>/<cust>/<date>/<session_id>.json`.

    A protocol-1 BATCH carries a `:<batch_seq>` suffix before `.json`
    (`kb/ingestion_app._compose_storage_id`); the finalize has no batch_seq to
    suffix with, which is the only thing that tells the two apart by key.
    """
    return not is_v2_key(key) and key.endswith(f"/{session_id}.json")


def last_key_ends_v1_session(keys: list[str], session_id: str) -> bool:
    """Key-only form of the rule for a protocol-1 row (what the sweep can see).

    The worker applies the same rule with the payloads in hand
    (`kb/handlers/claude_code.fetch_supplementary`), which additionally tells a
    protocol-2 finalize batch from an ordinary one.
    """
    if not keys:
        return False
    last = keys[-1]
    return is_cron_marker_key(last) or is_v1_client_finalize_key(last, session_id)


def last_key_ends_v1_session_sql(keys: str = "payload_s3_keys", session_id: str = "source_event_id") -> str:
    """SQL twin of `last_key_ends_v1_session`, for the sweep and the backfill.

    `right(...) = ...` rather than LIKE: a session id is interpolated, and LIKE
    would read any `_` or `%` in it as a wildcard.
    """
    last = f"{keys}[cardinality({keys})]"
    return (
        f"(cardinality({keys}) > 0 AND ("
        f"right({last}, {len(CRON_MARKER_SUFFIX)}) = '{CRON_MARKER_SUFFIX}'"
        f" OR (strpos({last}, '{V2_KEY_SEGMENT}') = 0"
        f" AND right({last}, length({session_id}) + 6) = '/' || {session_id} || '.json')))"
    )


def has_v2_key_sql(keys: str = "payload_s3_keys") -> str:
    """True when any key on the row is a protocol-2 batch."""
    return f"EXISTS (SELECT 1 FROM unnest({keys}) AS _k WHERE strpos(_k, '{V2_KEY_SEGMENT}') > 0)"

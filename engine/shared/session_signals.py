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
extraction-spend-plan.md). The worker (kb/handlers/claude_code.py), the sweep
(kb/session_completer.py) and the backfill all classify keys with
`signal_for_key` or its SQL twin, pinned against the same fixtures.

Protocol 2 is the one part decided from the payloads rather than the keys: the
server admits v2 batches strictly in sequence (kb/session_receipts.accept), so
"newest" is the highest accepted `batch_seq`, and `session_streams.finalized`
records whether that batch was the finalize. A sweep marker on top of a v2 row
also ends it; nothing puts one there until the sweep covers protocol 2.
"""

from __future__ import annotations

from enum import StrEnum

import orjson

#: Suffix of the sweep's placeholder object, written under
#: `raw/<source>/<customer>/<session_id>/finalize.marker`.
CRON_MARKER_SUFFIX = "/finalize.marker"

#: Path segment of every protocol-2 batch key (`kb/session_receipts.accept`).
V2_KEY_SEGMENT = "/sessions-v2/"

#: A protocol-1 key ends in `<storage id>.json`, where the storage id is the
#: session id plus `:<batch_seq>` for a batch and the bare session id for a
#: finalize (kb/ingestion_app._compose_storage_id).
V1_KEY_EXT = ".json"


class CompletedBy(StrEnum):
    """Which signal ended a session. Recorded on every extraction pass."""

    V2_FINALIZE = "v2_finalize"
    V1_CLIENT_FINALIZE = "v1_client_finalize"
    CRON_MARKER = "cron_marker"


def cron_marker_key(source: str, customer_id: str, session_id: str) -> str:
    """The sweep's marker key for one session. One per session, reused."""
    return f"raw/{source}/{customer_id}/{session_id}{CRON_MARKER_SUFFIX}"


def cron_marker_body(session_id: str) -> bytes:
    """What the marker object holds: no events. The worker never needs to read
    it -- the key alone says what it is -- but a reader that does sees a finalize."""
    return orjson.dumps(
        {
            "device_id": "cron-finalize",
            "session_id": session_id,
            "batch_seq": -1,
            "cwd": None,
            "events": [],
            "finalize": True,
        }
    )


def is_cron_marker_key(key: str) -> bool:
    return key.endswith(CRON_MARKER_SUFFIX)


def is_v2_key(key: str) -> bool:
    return V2_KEY_SEGMENT in key


def is_v1_client_finalize_key(key: str, session_id: str) -> bool:
    """A protocol-1 client finalize: `raw/<src>/<cust>/<date>/<session_id>.json`.

    A protocol-1 BATCH carries `:<batch_seq>` before `.json`; the finalize has
    no batch_seq to suffix with, which is the only thing that tells the two
    apart by key. A session id that itself contains `:` is a pre-0026 legacy
    row identity (`<session>:<batch>`), whose batch keys would otherwise look
    exactly like a finalize: never an end signal.
    """
    return (
        ":" not in session_id
        and not is_v2_key(key)
        and key.endswith(f"/{storage_id(session_id)}{V1_KEY_EXT}")
    )


def storage_id(session_id: str) -> str:
    """How a session id appears in its protocol-1 key (`/` would split the path,
    so kb/ingestion_app._payload_key writes it as `_`)."""
    return session_id.replace("/", "_")


def signal_for_key(key: str, session_id: str) -> CompletedBy | None:
    """What this key says about the session, judged by its key alone.

    Protocol-2 batches return None here: whether one was the finalize is in its
    payload and in `session_streams`, not in its key.
    """
    if is_cron_marker_key(key):
        return CompletedBy.CRON_MARKER
    if is_v1_client_finalize_key(key, session_id):
        return CompletedBy.V1_CLIENT_FINALIZE
    return None


def last_key_ends_v1_session(keys: list[str], session_id: str) -> bool:
    """Key-only form of the rule: is the newest key an end signal?"""
    return bool(keys) and signal_for_key(keys[-1], session_id) is not None


def last_key_sql(keys: str = "payload_s3_keys") -> str:
    """The newest key of an array column (NULL for an empty array)."""
    return f"{keys}[cardinality({keys})]"


def is_cron_marker_key_sql(key: str) -> str:
    return f"(right({key}, {len(CRON_MARKER_SUFFIX)}) = '{CRON_MARKER_SUFFIX}')"


def ends_v1_session_sql(last_key: str, session_id: str = "source_event_id") -> str:
    """SQL twin of `signal_for_key(...) is not None`, given the newest key.

    Pass a precomputed last key where the query scans many rows (a LATERAL
    column): repeating `payload_s3_keys[cardinality(payload_s3_keys)]` makes
    Postgres de-TOAST the whole array once per reference. False, never NULL,
    for an empty array. `right(...) = ...` rather than LIKE: a session id is
    interpolated, and LIKE would read `_` or `%` in it as a wildcard.
    """
    stored = f"replace({session_id}, '/', '_')"
    tail = f"'/' || {stored} || '{V1_KEY_EXT}'"
    return (
        f"COALESCE(({is_cron_marker_key_sql(last_key)}"
        f" OR (strpos({session_id}, ':') = 0"
        f" AND strpos({last_key}, '{V2_KEY_SEGMENT}') = 0"
        f" AND right({last_key}, length({stored}) + {len(V1_KEY_EXT) + 1}) = {tail})), false)"
    )


def has_v2_key_sql(keys: str = "payload_s3_keys") -> str:
    """True when any key on the row is a protocol-2 batch."""
    return f"EXISTS (SELECT 1 FROM unnest({keys}) AS _k WHERE strpos(_k, '{V2_KEY_SEGMENT}') > 0)"

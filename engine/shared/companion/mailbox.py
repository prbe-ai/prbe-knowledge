"""The mailbox: cards in, emission outcomes out, one lease table in between.

THREE RULES WITH TEETH (implementation spec v3 §2.3, §3):

* ENQUEUE IS IDEMPOTENT ON THE CALLER'S KEY. Same key + byte-identical payload
  hands back the original card; same key + a different payload is a CONFLICT,
  never a silent replace (rows are immutable); same key after the original
  expired is `expired` -- the caller mints a new key. The key is permanent for
  its target, which is what makes a retried `send` safe.
* NOTHING HERE PROMISES EXACTLY-ONCE MODEL EXPOSURE. `ack` records what an
  actuator DID (an emission attempt and its outcome), keyed by a client-minted
  `attempt_id` so retries collapse and repeated emissions stay visible. A card
  with ANY delivery row is no longer pending for anyone.
* THE ACTOR LANE CLAIMS BEFORE IT EMITS. Session-keyed cards are consumed by
  one process reading one journal; actor-keyed cards (the MCP rider) can be
  selected by concurrent tool calls, so `claim_for_actor` takes a short lease
  atomically and exactly one caller wins. A lease that lapses without an ack
  can be reclaimed; that is the caller's `unknown` outcome to record, not ours.

The `[probe companion] ` prefix is applied HERE, at write time, and counts
toward the 4,000-character cap. Actuators do zero formatting; a body that
arrives already prefixed is normalised rather than double-prefixed.

Everything runs under `with_tenant`, so FORCE RLS applies on every statement.
`source` and `mode` are pinned here (`driver` / `live`): no caller can claim
to be the intelligent layer before it exists.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

import asyncpg

from engine.shared.db import with_tenant

PREFIX = "[probe companion] "
BODY_MAX = 4_000
TTL_MIN = 1
TTL_MAX = 86_400
DEDUPE_KEY_MAX = 200
SESSION_ID_MAX = 200
RECIPIENT_MAX = 200
LEASE_SECONDS_DEFAULT = 60
#: How many pending actor cards one claim call will try before giving up.
CLAIM_CANDIDATES = 5

CLASSES: frozenset[str] = frozenset({"seam", "decision-push"})
SEAMS: frozenset[str] = frozenset(
    {
        "stop",
        "user-prompt",
        "post-tool",
        "post-tool-batch",
        "push",
        "mcp-rider",
        "pi-input",
        "pi-extension",
        "codex-session-start",
        "codex-user-prompt",
        "codex-async",
        "codex-stop",
    }
)
OUTCOMES: frozenset[str] = frozenset({"emitted", "expired", "canceled", "unknown", "unsupported"})
SESSION_STATES: frozenset[str] = frozenset({"active", "idle"})


@dataclass(frozen=True)
class Card:
    mailbox_id: UUID
    recipient: str
    session_id: str | None
    class_: str
    intended_seam: str | None
    body: str
    trial_id: UUID
    created_at: datetime
    expires_at: datetime


class EnqueueRefused(ValueError):
    """The request is structurally invalid; nothing was written."""


class EnqueueConflict(Exception):
    """The dedupe key is already taken and cannot be reused as asked."""

    def __init__(self, reason: Literal["conflict", "expired"], existing: Card) -> None:
        super().__init__(f"dedupe key already used: {reason}")
        self.reason = reason
        self.existing = existing


class AckRefused(ValueError):
    """The ack names a seam, outcome or card this tenant does not have."""


class UnknownCard(AckRefused):
    """The ack names a card this tenant does not have (a 404, not a 422)."""


_CARD_COLUMNS = (
    "id, recipient, session_id, class, intended_seam, body, trial_id, created_at, expires_at"
)


def _card(row: asyncpg.Record) -> Card:
    return Card(
        mailbox_id=row["id"],
        recipient=row["recipient"],
        session_id=row["session_id"],
        class_=row["class"],
        intended_seam=row["intended_seam"],
        body=row["body"],
        trial_id=row["trial_id"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


def _normalise_body(body: object) -> str:
    text = body.strip() if isinstance(body, str) else ""
    if text.startswith(PREFIX.strip()):
        text = text[len(PREFIX.strip()) :].strip()
    if not text:
        raise EnqueueRefused("body is empty")
    full = PREFIX + text
    if len(full) > BODY_MAX:
        raise EnqueueRefused(
            f"body is {len(full)} chars including the prefix; the cap is {BODY_MAX}"
        )
    return full


def _validate_enqueue(
    *,
    recipient: object,
    session_id: object,
    class_: object,
    dedupe_key: object,
    ttl_seconds: object,
    intended_seam: object,
) -> None:
    if (
        not isinstance(recipient, str)
        or not recipient.startswith("user:")
        or len(recipient) > RECIPIENT_MAX
    ):
        raise EnqueueRefused("recipient must be 'user:<id>'")
    if session_id is not None and (
        not isinstance(session_id, str) or not 1 <= len(session_id) <= SESSION_ID_MAX
    ):
        raise EnqueueRefused(
            "session_id must be null or 1..200 chars -- an empty string is not a third targeting mode"
        )
    if class_ not in CLASSES:
        raise EnqueueRefused(f"class must be one of {sorted(CLASSES)}")
    if class_ == "decision-push" and session_id is None:
        raise EnqueueRefused(
            "a decision-push card needs a session: an actor-keyed card has no decision to belong to"
        )
    if intended_seam is not None and intended_seam not in SEAMS:
        raise EnqueueRefused(f"intended_seam must be null or one of {sorted(SEAMS)}")
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or not TTL_MIN <= ttl_seconds <= TTL_MAX
    ):
        raise EnqueueRefused(f"ttl_seconds must be an integer in [{TTL_MIN}, {TTL_MAX}]")
    if not isinstance(dedupe_key, str) or not 1 <= len(dedupe_key) <= DEDUPE_KEY_MAX:
        raise EnqueueRefused("dedupe_key must be 1..200 chars")


async def enqueue(
    customer_id: str,
    *,
    recipient: str,
    session_id: str | None,
    class_: str,
    body: str,
    dedupe_key: str,
    ttl_seconds: int,
    intended_seam: str | None = None,
) -> tuple[Card, bool]:
    """Write one card, or hand back the one this key already names.

    Returns `(card, created)`. Raises `EnqueueRefused` on a structurally bad
    request (nothing written) and `EnqueueConflict` when the key exists with a
    different payload (`conflict`) or has expired (`expired`).
    """
    _validate_enqueue(
        recipient=recipient,
        session_id=session_id,
        class_=class_,
        dedupe_key=dedupe_key,
        ttl_seconds=ttl_seconds,
        intended_seam=intended_seam,
    )
    full_body = _normalise_body(body)
    trial_id = uuid4()

    if session_id is not None:
        conflict_target = "(customer_id, session_id, dedupe_key) WHERE session_id IS NOT NULL"
        existing_predicate = "session_id = $2 AND dedupe_key = $3"
        existing_args: tuple[Any, ...] = (customer_id, session_id, dedupe_key)
    else:
        conflict_target = "(customer_id, recipient, dedupe_key) WHERE session_id IS NULL"
        existing_predicate = "recipient = $2 AND session_id IS NULL AND dedupe_key = $3"
        existing_args = (customer_id, recipient, dedupe_key)

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            f"""
            INSERT INTO companion_mailbox
                (customer_id, recipient, session_id, class, intended_seam, body,
                 dedupe_key, mode, source, trial_id, expires_at)
            VALUES ($1, $2, $3::text, $4, $5::text, $6, $7, 'live', 'driver', $8,
                    now() + make_interval(secs => $9))
            ON CONFLICT {conflict_target} DO NOTHING
            RETURNING {_CARD_COLUMNS}
            """,
            customer_id,
            recipient,
            session_id,
            class_,
            intended_seam,
            full_body,
            dedupe_key,
            trial_id,
            ttl_seconds,
        )
        if row is not None:
            return _card(row), True

        existing = await conn.fetchrow(
            f"""
            SELECT {_CARD_COLUMNS}, expires_at <= now() AS expired
            FROM companion_mailbox
            WHERE customer_id = $1 AND {existing_predicate}
            """,
            *existing_args,
        )
    if existing is None:  # pragma: no cover -- append-only, so the row cannot have vanished
        raise EnqueueRefused("dedupe key collided with a row that could not be read back")
    card = _card(existing)
    if existing["expired"]:
        raise EnqueueConflict("expired", card)
    # created_at and expires_at come from the same transaction timestamp, so
    # the stored TTL is exact -- a one-second difference is a different request.
    same_ttl = round((card.expires_at - card.created_at).total_seconds()) == ttl_seconds
    if (
        card.body != full_body
        or card.class_ != class_
        or card.intended_seam != intended_seam
        or card.recipient != recipient
        or not same_ttl
    ):
        raise EnqueueConflict("conflict", card)
    return card, False


_PENDING_PREDICATE = """
    m.customer_id = $1
    AND m.expires_at > now()
    AND NOT EXISTS (
        SELECT 1 FROM companion_deliveries d
        WHERE d.customer_id = m.customer_id AND d.mailbox_id = m.id
    )
    AND NOT EXISTS (
        SELECT 1 FROM companion_claims c
        WHERE c.customer_id = m.customer_id AND c.mailbox_id = m.id AND c.lease_until > now()
    )
"""


async def pending(
    customer_id: str,
    *,
    session_id: str,
    recipient: str,
    exclude: Collection[UUID] = (),
) -> list[Card]:
    """Cards a session's actuators may still emit, oldest first.

    "Pending" is derived, never stored: unexpired, no delivery row of ANY
    outcome, no live lease, and not one of the ids the client already holds.
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            f"""
            SELECT {", ".join("m." + c.strip() for c in _CARD_COLUMNS.split(","))}
            FROM companion_mailbox m
            WHERE {_PENDING_PREDICATE}
              AND m.session_id = $2
              AND m.recipient = $3
              AND NOT (m.id = ANY($4::uuid[]))
            ORDER BY m.created_at, m.id
            """,
            customer_id,
            session_id,
            recipient,
            list(exclude),
        )
    return [_card(r) for r in rows]


async def claim_for_actor(
    customer_id: str,
    *,
    recipient: str,
    claimed_by: str,
    lease_seconds: int = LEASE_SECONDS_DEFAULT,
) -> Card | None:
    """Take a short lease on the oldest pending actor-keyed card, or None.

    Atomic across concurrent callers: the lease row's primary key is the card,
    and the upsert only overwrites a lease that has already lapsed. A caller
    that loses the race gets no row back and tries the next candidate.
    """
    if not 1 <= lease_seconds <= 3_600:
        raise ValueError("lease_seconds must be in [1, 3600]")
    async with with_tenant(customer_id) as conn:
        candidates = await conn.fetch(
            f"""
            SELECT {", ".join("m." + c.strip() for c in _CARD_COLUMNS.split(","))}
            FROM companion_mailbox m
            WHERE {_PENDING_PREDICATE}
              AND m.session_id IS NULL
              AND m.recipient = $2
            ORDER BY m.created_at, m.id
            LIMIT $3
            """,
            customer_id,
            recipient,
            CLAIM_CANDIDATES,
        )
        for row in candidates:
            won = await conn.fetchval(
                """
                INSERT INTO companion_claims (customer_id, mailbox_id, claimed_by, lease_until)
                VALUES ($1, $2, $3, now() + make_interval(secs => $4))
                ON CONFLICT (customer_id, mailbox_id) DO UPDATE
                    SET claimed_by = EXCLUDED.claimed_by,
                        lease_until = EXCLUDED.lease_until
                    WHERE companion_claims.lease_until <= now()
                RETURNING mailbox_id
                """,
                customer_id,
                row["id"],
                claimed_by,
                lease_seconds,
            )
            if won is not None:
                return _card(row)
    return None


async def ack(
    customer_id: str,
    *,
    mailbox_id: UUID,
    attempt_id: UUID,
    seam: str,
    outcome: str,
    receiving_instance: str,
    delivering_credential: str | None = None,
    harness_version: str | None = None,
    session_state: str | None = None,
    client_received_at: datetime | None = None,
    client_emitted_at: datetime | None = None,
    receipt_to_emission_ms: int | None = None,
    evidence: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """Record one emission attempt's outcome. Idempotent on `attempt_id`.

    Returns `(delivery_id, created)`. Releases any lease on the card. Raises
    `AckRefused` (a ValueError) for an unknown seam/outcome/state or a card this
    tenant does not have -- a mis-addressed ack must never file a row under
    somebody else's card, and the composite FK guarantees it cannot.
    """
    if seam not in SEAMS:
        raise AckRefused(f"seam must be one of {sorted(SEAMS)}")
    if outcome not in OUTCOMES:
        raise AckRefused(f"outcome must be one of {sorted(OUTCOMES)}")
    if session_state is not None and session_state not in SESSION_STATES:
        raise AckRefused("session_state must be null, 'active' or 'idle'")
    if not isinstance(receiving_instance, str) or not receiving_instance.strip():
        raise AckRefused("receiving_instance is required")
    if receipt_to_emission_ms is not None and receipt_to_emission_ms < 0:
        raise AckRefused("receipt_to_emission_ms cannot be negative")

    async with with_tenant(customer_id) as conn:
        try:
            new_id = await conn.fetchval(
                """
                INSERT INTO companion_deliveries
                    (customer_id, mailbox_id, attempt_id, seam, outcome,
                     delivering_credential, receiving_instance, harness_version,
                     session_state, client_received_at, client_emitted_at,
                     receipt_to_emission_ms, evidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
                ON CONFLICT (customer_id, attempt_id) DO NOTHING
                RETURNING id
                """,
                customer_id,
                mailbox_id,
                attempt_id,
                seam,
                outcome,
                delivering_credential,
                receiving_instance,
                harness_version,
                session_state,
                client_received_at,
                client_emitted_at,
                receipt_to_emission_ms,
                json.dumps(evidence or {}),
            )
        except asyncpg.ForeignKeyViolationError as exc:
            raise UnknownCard("mailbox_id is not a card of this tenant") from exc
        if new_id is not None:
            await conn.execute(
                "DELETE FROM companion_claims WHERE customer_id = $1 AND mailbox_id = $2",
                customer_id,
                mailbox_id,
            )
            return int(new_id), True
        existing_id = await conn.fetchval(
            "SELECT id FROM companion_deliveries WHERE customer_id = $1 AND attempt_id = $2",
            customer_id,
            attempt_id,
        )
    return int(existing_id), False


async def deliveries(
    customer_id: str,
    *,
    session_id: str | None = None,
    recipient: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Readback for the fault catalog: one dict per emission attempt."""
    if session_id is None and recipient is None:
        raise ValueError("filter by session_id or recipient")
    if not 1 <= limit <= 1_000:
        raise ValueError("limit must be in [1, 1000]")
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.mailbox_id, d.attempt_id, d.seam, d.outcome,
                   d.delivering_credential, d.receiving_instance, d.harness_version,
                   d.session_state, d.client_received_at, d.client_emitted_at,
                   d.receipt_to_emission_ms, d.ack_received_at, d.evidence,
                   m.session_id, m.recipient, m.trial_id, m.intended_seam, m.class,
                   m.created_at AS enqueued_at
            FROM companion_deliveries d
            JOIN companion_mailbox m
              ON m.customer_id = d.customer_id AND m.id = d.mailbox_id
            WHERE d.customer_id = $1
              AND ($2::text IS NULL OR m.session_id = $2)
              AND ($3::text IS NULL OR m.recipient = $3)
            ORDER BY d.ack_received_at, d.id
            LIMIT $4
            """,
            customer_id,
            session_id,
            recipient,
            limit,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("evidence"), str):
            d["evidence"] = json.loads(d["evidence"])
        out.append(d)
    return out

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
`source` and `mode` are pinned per entry point, never taken from a request
body: `enqueue` writes `driver`/`live` (a person's card, served through poll);
`register_local` writes `local-brain` with `live` or `shadow` (a card the
device's own harness already emitted or held, registered so its receipts and
observations have a home). Poll and the actor claim serve driver/live rows
only -- a registered card was emitted where it was minted and must never be
served a second time.
"""

from __future__ import annotations

import hashlib
import json
import re
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
SOURCES: frozenset[str] = frozenset({"driver", "local-brain"})
MODES: frozenset[str] = frozenset({"live", "shadow"})
OBSERVATION_KINDS: frozenset[str] = frozenset({"context", "behaviour"})
BEHAVIOUR_OUTCOMES: frozenset[str] = frozenset(
    {"followed", "ignored", "contradicted", "overridden"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Card:
    mailbox_id: UUID
    recipient: str
    session_id: str | None
    class_: str
    intended_seam: str | None
    #: None only for a shadow registration, which is recorded by hash alone.
    body: str | None
    trial_id: UUID
    created_at: datetime
    expires_at: datetime
    body_sha256: str = ""
    source: str = "driver"
    mode: str = "live"
    local_card_id: UUID | None = None
    dedupe_key: str = ""


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
    "id, recipient, session_id, class, intended_seam, body, body_sha256, source, mode, "
    "local_card_id, dedupe_key, trial_id, created_at, expires_at"
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
        body_sha256=row["body_sha256"],
        source=row["source"],
        mode=row["mode"],
        local_card_id=row["local_card_id"],
        dedupe_key=row["dedupe_key"],
    )


def _sha256(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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
                 body_sha256, dedupe_key, mode, source, trial_id, expires_at)
            VALUES ($1, $2, $3::text, $4, $5::text, $6, $7, $8, 'live', 'driver', $9,
                    now() + make_interval(secs => $10))
            ON CONFLICT {conflict_target} DO NOTHING
            RETURNING {_CARD_COLUMNS}
            """,
            customer_id,
            recipient,
            session_id,
            class_,
            intended_seam,
            full_body,
            _sha256(full_body),
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


async def register_local(
    customer_id: str,
    *,
    recipient: str,
    session_id: str,
    local_card_id: UUID,
    class_: str,
    mode: str,
    body: str | None,
    body_sha256: str | None,
    dedupe_key: str,
    ttl_seconds: int,
    intended_seam: str | None = None,
) -> tuple[Card, bool]:
    """Register a card the device's own harness minted, so its receipts and
    observations land in the same ledger as driver cards. Idempotent on
    `local_card_id`.

    `mode='live'` carries the body (the card was emitted locally, and the
    transcript that carries it is uploaded by capture anyway); `mode='shadow'`
    carries only `body_sha256` (the card was held back, and its text stays on
    the device). Neither is ever served by poll or the actor claim.

    Returns `(card, created)`. Raises `EnqueueRefused` for a structurally bad
    request and `EnqueueConflict` when the local id or the session dedupe key
    already names a different card (`conflict`) or an expired one (`expired`).
    """
    _validate_enqueue(
        recipient=recipient,
        session_id=session_id,
        class_=class_,
        dedupe_key=dedupe_key,
        ttl_seconds=ttl_seconds,
        intended_seam=intended_seam,
    )
    if session_id is None:
        raise EnqueueRefused("a local card is always session-targeted")
    if mode not in MODES:
        raise EnqueueRefused(f"mode must be one of {sorted(MODES)}")
    full_body: str | None
    if mode == "live":
        if body is None:
            raise EnqueueRefused("a live registration carries the emitted body")
        full_body = _normalise_body(body)
        sha = _sha256(full_body)
        if body_sha256 is not None and body_sha256 != sha:
            raise EnqueueRefused("body_sha256 does not match the body")
    else:
        if body is not None:
            raise EnqueueRefused("a shadow registration carries only body_sha256")
        if not isinstance(body_sha256, str) or not _SHA256.match(body_sha256):
            raise EnqueueRefused("body_sha256 must be 64 lowercase hex characters")
        full_body = None
        sha = body_sha256
    trial_id = uuid4()

    async with with_tenant(customer_id) as conn:
        try:
            # A savepoint: `with_tenant` already holds the transaction, and a
            # dedupe-key violation must not abort the readback that explains it.
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO companion_mailbox
                        (customer_id, recipient, session_id, class, intended_seam, body,
                         body_sha256, dedupe_key, mode, source, local_card_id, trial_id,
                         expires_at)
                    VALUES ($1, $2, $3, $4, $5::text, $6::text, $7, $8, $9, 'local-brain',
                            $10, $11, now() + make_interval(secs => $12))
                    ON CONFLICT (customer_id, local_card_id) WHERE local_card_id IS NOT NULL
                        DO NOTHING
                    RETURNING {_CARD_COLUMNS}
                    """,
                    customer_id,
                    recipient,
                    session_id,
                    class_,
                    intended_seam,
                    full_body,
                    sha,
                    dedupe_key,
                    mode,
                    local_card_id,
                    trial_id,
                    ttl_seconds,
                )
        except asyncpg.UniqueViolationError:
            # The session dedupe key is taken by a card with another local id.
            other = await conn.fetchrow(
                f"""
                SELECT {_CARD_COLUMNS}
                FROM companion_mailbox
                WHERE customer_id = $1 AND session_id = $2 AND dedupe_key = $3
                """,
                customer_id,
                session_id,
                dedupe_key,
            )
            if other is None:  # pragma: no cover -- the violation named this key
                raise EnqueueRefused(
                    "dedupe key collided with a row that could not be read back"
                ) from None
            raise EnqueueConflict("conflict", _card(other)) from None
        if row is not None:
            return _card(row), True
        existing = await conn.fetchrow(
            f"""
            SELECT {_CARD_COLUMNS}, expires_at <= now() AS expired
            FROM companion_mailbox
            WHERE customer_id = $1 AND local_card_id = $2
            """,
            customer_id,
            local_card_id,
        )
    if existing is None:  # pragma: no cover -- append-only, so the row cannot have vanished
        raise EnqueueRefused("local card id collided with a row that could not be read back")
    card = _card(existing)
    if existing["expired"]:
        raise EnqueueConflict("expired", card)
    same_ttl = round((card.expires_at - card.created_at).total_seconds()) == ttl_seconds
    if (
        card.body_sha256 != sha
        or card.mode != mode
        or card.class_ != class_
        or card.intended_seam != intended_seam
        or card.recipient != recipient
        or card.session_id != session_id
        or card.dedupe_key != dedupe_key
        or not same_ttl
    ):
        raise EnqueueConflict("conflict", card)
    return card, False


#: Poll and the actor claim serve a person's live cards only. A registered
#: local card was emitted (or held) where it was minted; serving it again
#: would be a second exposure the local ledger never asked for.
_PENDING_PREDICATE = """
    m.customer_id = $1
    AND m.source = 'driver'
    AND m.mode = 'live'
    AND m.expires_at > now()
    AND NOT EXISTS (
        SELECT 1 FROM companion_deliveries d
        WHERE d.customer_id = m.customer_id AND d.mailbox_id = m.id
    )
    AND NOT EXISTS (
        SELECT 1 FROM companion_claims c
        WHERE c.customer_id = m.customer_id AND c.mailbox_id = m.id
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
        await _retire_lapsed_leases(conn, customer_id, recipient)
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


#: Marks a delivery row the ENGINE wrote, not an actuator: the lease lapsed
#: without an ack, so the outcome is unknown and the card is retired.
LEASE_LAPSED_INSTANCE = "engine:lease-lapsed"


async def _retire_lapsed_leases(
    conn: asyncpg.Connection, customer_id: str, recipient: str | None = None
) -> int:
    """Turn every lapsed, un-acked lease into a terminal `unknown` delivery.

    `recipient=None` sweeps the whole tenant. Called on every claim AND on
    every readback (poll, deliveries, report) so a lapse is visible as soon
    as anyone looks, not only when the next actor claim happens to run.
    Returns how many were retired."""
    lapsed = await conn.fetch(
        """
        DELETE FROM companion_claims c
        USING companion_mailbox m
        WHERE c.customer_id = $1
          AND m.customer_id = c.customer_id AND m.id = c.mailbox_id
          AND ($2::text IS NULL OR m.recipient = $2)
          AND c.lease_until <= now()
          AND NOT EXISTS (
              SELECT 1 FROM companion_deliveries d
              WHERE d.customer_id = c.customer_id AND d.mailbox_id = c.mailbox_id
          )
        RETURNING c.mailbox_id, c.claimed_by, c.lease_until
        """,
        customer_id,
        recipient,
    )
    for row in lapsed:
        await conn.execute(
            """
            INSERT INTO companion_deliveries
                (customer_id, mailbox_id, attempt_id, seam, outcome,
                 delivering_credential, receiving_instance, evidence)
            VALUES ($1, $2, $3, 'mcp-rider', 'unknown', $4, $5, $6::jsonb)
            ON CONFLICT (customer_id, attempt_id) DO NOTHING
            """,
            customer_id,
            row["mailbox_id"],
            uuid4(),
            row["claimed_by"],
            LEASE_LAPSED_INSTANCE,
            json.dumps(
                {
                    "reason": "lease lapsed without an ack",
                    "lease_until": row["lease_until"].isoformat(),
                }
            ),
        )
    return len(lapsed)


async def claim_for_actor(
    customer_id: str,
    *,
    recipient: str,
    claimed_by: str,
    lease_seconds: int = LEASE_SECONDS_DEFAULT,
) -> Card | None:
    """Take a short lease on the oldest pending actor-keyed card, or None.

    Atomic across concurrent callers: the lease row's primary key is the card,
    so two concurrent inserts resolve to exactly one winner; the loser gets no
    row back and tries the next candidate.

    A LAPSED LEASE IS NOT RECLAIMED (spec v3 §3). A lease that ran out without
    an ack is an AMBIGUOUS emission -- the card may already be in front of a
    model -- so re-issuing it would risk a second exposure. Instead the card is
    retired here as a terminal `unknown` delivery attributed to the lapsed
    claimant, and the claim moves on. The retirement is a DELETE ... RETURNING
    on the lease row, so concurrent callers cannot both retire the same card.
    """
    if not 1 <= lease_seconds <= 3_600:
        raise ValueError("lease_seconds must be in [1, 3600]")
    async with with_tenant(customer_id) as conn:
        await _retire_lapsed_leases(conn, customer_id, recipient)
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
                ON CONFLICT (customer_id, mailbox_id) DO NOTHING
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


class UnknownAttempt(AckRefused):
    """No delivery row carries this attempt_id for this tenant."""


async def observe(
    customer_id: str,
    *,
    mailbox_id: UUID,
    attempt_id: UUID,
    observed: bool,
    observer: str,
    kind: str = "context",
    outcome: str | None = None,
    client_observed_at: datetime | None = None,
    evidence: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """Qualify one delivery attempt with one fact (spec §8).

    `kind='context'`: `observed` says whether the model-readable context
    carried the card. `kind='behaviour'`: `outcome` says what the model did
    with it (followed / ignored / contradicted / overridden) and `observed` is
    derived as `outcome == 'followed'`; only a local observer reading the
    transcript after delivery can assert this one.

    First write wins PER KIND: an observation is a bounded search's verdict,
    and a later contradicting row would make the catalog say two things.
    Returns `(observation_id, created)`. Raises `UnknownAttempt` when no
    delivery row of this tenant carries `attempt_id` (the composite FK is the
    guard), and `AckRefused` for a malformed observer, kind or outcome.
    """
    if not isinstance(observer, str) or not observer.strip() or len(observer) > 200:
        raise AckRefused("observer is required (1..200 chars)")
    if kind not in OBSERVATION_KINDS:
        raise AckRefused(f"kind must be one of {sorted(OBSERVATION_KINDS)}")
    if kind == "behaviour":
        if outcome not in BEHAVIOUR_OUTCOMES:
            raise AckRefused(f"a behaviour verdict needs outcome in {sorted(BEHAVIOUR_OUTCOMES)}")
        observed = outcome == "followed"
    elif outcome is not None:
        raise AckRefused("outcome belongs to behaviour verdicts only")
    async with with_tenant(customer_id) as conn:
        try:
            new_id = await conn.fetchval(
                """
                INSERT INTO companion_observations
                    (customer_id, mailbox_id, attempt_id, kind, observed, outcome, observer,
                     client_observed_at, evidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (customer_id, attempt_id, kind) DO NOTHING
                RETURNING id
                """,
                customer_id,
                mailbox_id,
                attempt_id,
                kind,
                observed,
                outcome,
                observer,
                client_observed_at,
                json.dumps(evidence or {}),
            )
        except asyncpg.ForeignKeyViolationError as exc:
            raise UnknownAttempt("attempt_id is not a delivery of this tenant's card") from exc
        if new_id is not None:
            return int(new_id), True
        existing = await conn.fetchval(
            """
            SELECT id FROM companion_observations
            WHERE customer_id = $1 AND attempt_id = $2 AND kind = $3
            """,
            customer_id,
            attempt_id,
            kind,
        )
    return int(existing), False


def _validate_source(source: str | None) -> None:
    if source is not None and source not in SOURCES:
        raise ValueError(f"source must be null or one of {sorted(SOURCES)}")


async def deliveries(
    customer_id: str,
    *,
    session_id: str | None = None,
    recipient: str | None = None,
    limit: int = 200,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Readback for the fault catalog: one dict per emission attempt.

    Carries the card's `source`, `mode` and `local_card_id` so driver cards
    and locally registered cards read from one ledger, and both verdict kinds:
    `observed_in_context` (plus its observer and evidence) and
    `behaviour_outcome` (plus its observer and evidence). `source` narrows to
    one writer.
    """
    if session_id is None and recipient is None:
        raise ValueError("filter by session_id or recipient")
    # 1001 so a caller may fetch one row past its own 1,000 bound to detect truncation.
    if not 1 <= limit <= 1_001:
        raise ValueError("limit must be in [1, 1001]")
    _validate_source(source)
    async with with_tenant(customer_id) as conn:
        await _retire_lapsed_leases(conn, customer_id, recipient)
        rows = await conn.fetch(
            """
            SELECT d.id, d.mailbox_id, d.attempt_id, d.seam, d.outcome,
                   d.delivering_credential, d.receiving_instance, d.harness_version,
                   d.session_state, d.client_received_at, d.client_emitted_at,
                   d.receipt_to_emission_ms, d.ack_received_at, d.evidence,
                   m.session_id, m.recipient, m.trial_id, m.intended_seam, m.class,
                   m.source, m.mode, m.local_card_id,
                   m.created_at AS enqueued_at,
                   o.observed AS observed_in_context, o.observer, o.observed_at,
                   o.client_observed_at, o.evidence AS observation_evidence,
                   b.outcome AS behaviour_outcome, b.observer AS behaviour_observer,
                   b.observed_at AS behaviour_observed_at,
                   b.evidence AS behaviour_evidence
            FROM companion_deliveries d
            JOIN companion_mailbox m
              ON m.customer_id = d.customer_id AND m.id = d.mailbox_id
            LEFT JOIN companion_observations o
              ON o.customer_id = d.customer_id AND o.attempt_id = d.attempt_id
             AND o.kind = 'context'
            LEFT JOIN companion_observations b
              ON b.customer_id = d.customer_id AND b.attempt_id = d.attempt_id
             AND b.kind = 'behaviour'
            WHERE d.customer_id = $1
              AND ($2::text IS NULL OR m.session_id = $2)
              AND ($3::text IS NULL OR m.recipient = $3)
              AND ($5::text IS NULL OR m.source = $5)
            ORDER BY d.ack_received_at, d.id
            LIMIT $4
            """,
            customer_id,
            session_id,
            recipient,
            limit,
            source,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        for key in ("evidence", "observation_evidence", "behaviour_evidence"):
            if isinstance(d.get(key), str):
                d[key] = json.loads(d[key])
        out.append(d)
    return out


#: Evidence keys the actuators may set; the report counts rows where each is
#: a JSON `true`. Keeping the two facts separate is the point (spec §8):
#: `harness_accepted` = the harness applied the output; `observed_in_context`
#: = the trial nonce was seen where the model reads. Neither is implied by an
#: emitted ack.
EVIDENCE_HARNESS_ACCEPTED = "harness_accepted"
EVIDENCE_OBSERVED_IN_CONTEXT = "observed_in_context"


async def report(
    customer_id: str,
    *,
    session_id: str | None = None,
    recipient: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Per (seam, outcome) aggregates for the fault catalog.

    Counts attempts, the two context facts (`harness_accepted`,
    `observed_in_context` / `not_observed`), the behaviour verdicts
    (`followed` / `not_followed`), and summarises the monotonic
    receipt-to-emission latency (the only exact local latency; cross-clock
    deltas are estimates and are deliberately not aggregated here). `source`
    narrows to one writer so driver trials and local-brain cards can be read
    apart without a second report.
    """
    if session_id is None and recipient is None:
        raise ValueError("filter by session_id or recipient")
    _validate_source(source)
    async with with_tenant(customer_id) as conn:
        await _retire_lapsed_leases(conn, customer_id, recipient)
        rows = await conn.fetch(
            """
            SELECT d.seam, d.outcome,
                   count(*)::int AS attempts,
                   count(*) FILTER (WHERE d.evidence -> $4 = 'true'::jsonb)::int AS harness_accepted,
                   count(*) FILTER (
                       WHERE d.evidence -> $5 = 'true'::jsonb OR o.observed IS TRUE
                   )::int AS observed_in_context,
                   count(*) FILTER (
                       WHERE o.observed IS FALSE AND (d.evidence -> $5) IS DISTINCT FROM 'true'::jsonb
                   )::int AS not_observed,
                   count(*) FILTER (WHERE b.outcome = 'followed')::int AS followed,
                   count(*) FILTER (
                       WHERE b.outcome IN ('ignored', 'contradicted', 'overridden')
                   )::int AS not_followed,
                   count(d.receipt_to_emission_ms)::int AS latency_n,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY d.receipt_to_emission_ms) AS latency_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY d.receipt_to_emission_ms) AS latency_p95,
                   min(d.receipt_to_emission_ms) AS latency_min,
                   max(d.receipt_to_emission_ms) AS latency_max,
                   min(d.ack_received_at) AS first_at,
                   max(d.ack_received_at) AS last_at
            FROM companion_deliveries d
            JOIN companion_mailbox m
              ON m.customer_id = d.customer_id AND m.id = d.mailbox_id
            LEFT JOIN companion_observations o
              ON o.customer_id = d.customer_id AND o.attempt_id = d.attempt_id
             AND o.kind = 'context'
            LEFT JOIN companion_observations b
              ON b.customer_id = d.customer_id AND b.attempt_id = d.attempt_id
             AND b.kind = 'behaviour'
            WHERE d.customer_id = $1
              AND ($2::text IS NULL OR m.session_id = $2)
              AND ($3::text IS NULL OR m.recipient = $3)
              AND ($6::text IS NULL OR m.source = $6)
            GROUP BY d.seam, d.outcome
            ORDER BY d.seam, d.outcome
            """,
            customer_id,
            session_id,
            recipient,
            EVIDENCE_HARNESS_ACCEPTED,
            EVIDENCE_OBSERVED_IN_CONTEXT,
            source,
        )
    return [dict(r) for r in rows]

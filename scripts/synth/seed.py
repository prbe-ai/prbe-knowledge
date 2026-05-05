"""Synthetic tenant seeding — admin-triggered population of a real-shape
customer workspace with canonical synthetic content.

See docs/superpowers/specs/2026-05-04-synth-plan-4-tenant-seeding-v1-design.md.
"""

from __future__ import annotations

import sys

_VALID_PREFIXES: tuple[str, ...] = ("cust-eval-", "cust-synth-")


def is_seed_eligible(customer_id: str, metadata: dict | None) -> bool:
    """Return True if customer_id is allowed to receive synth seed data.

    Rules:
    - cust-eval-* / cust-synth-* prefixes are always eligible (existing rule
      from profile.py:42 _VALID_PREFIXES and clean_tenant at bootstrap.py:119).
    - Other prefixes are eligible only when metadata['allow_synth_seed'] is
      Python True. Truthy strings/ints don't count — explicit boolean only.
    """
    if customer_id.startswith(_VALID_PREFIXES):
        return True
    if metadata is None:
        return False
    return metadata.get("allow_synth_seed") is True


def prompt_typed_confirm(expected_customer_id: str) -> bool:
    """Prompt operator to type the customer_id back literally to confirm.

    Returns True iff the typed input (whitespace stripped) exactly matches
    expected_customer_id. Empty input returns False.

    Reads from sys.stdin so monkeypatch can substitute it in tests.
    """
    print(
        f"To confirm seeding {expected_customer_id!r}, type the customer_id back: ",
        end="",
        flush=True,
    )
    typed = sys.stdin.readline().strip()
    if not typed:
        return False
    return typed == expected_customer_id


async def set_allow_synth_seed(customer_id: str, db) -> None:
    """Toggle customers.metadata.allow_synth_seed = true for the named customer.

    Idempotent: re-running on an already-set tenant is a no-op (the UPDATE
    re-writes the same value). Refuses with ValueError if the customer row
    doesn't exist — synth doesn't auto-create real-shape tenants.

    `db` is an asyncpg Pool (matches the signature of bootstrap.py::init_tenant).
    """
    result = await db.execute(
        """
        UPDATE customers
           SET metadata = jsonb_set(
                   COALESCE(metadata, '{}'::jsonb),
                   '{allow_synth_seed}',
                   'true'::jsonb,
                   true
               )
         WHERE customer_id = $1
        """,
        customer_id,
    )
    parts = result.split()
    affected = int(parts[-1]) if parts and parts[-1].isdigit() else 0
    if affected == 0:
        raise ValueError(
            f"customer {customer_id!r} not found in customers table; "
            f"create the tenant via prbe-backend signup first"
        )


def _substitute_customer_id(
    *,
    payload: dict,
    old_key: str,
    old_id: str,
    new_id: str,
) -> tuple[dict, str]:
    """Rewrite a canonical envelope's customer_id field and R2 key for
    a target tenant.

    V1 scope: only the top-level `customer_id` field on the payload is
    rewritten. Nested references (e.g. thread_parent_id segments that
    happen to contain the canonical customer_id) are left untouched —
    no downstream consumer interprets them as customer_ids.

    Idempotent: if old_id is not in old_key but payload has new_id,
    the transformation was already applied; return unchanged.
    Raises ValueError if old_id is not in old_key and payload doesn't
    have new_id (malformed fixture).
    """
    new_payload = dict(payload)

    if old_id not in old_key:
        # Key was already substituted or is malformed
        if new_payload.get("customer_id") == new_id:
            # Already transformed, return as-is (idempotent)
            return new_payload, old_key
        # Malformed: old_id not in key and payload doesn't match new_id
        raise ValueError(
            f"old_id not found in R2 key: old_id={old_id!r}, old_key={old_key!r}"
        )

    new_payload["customer_id"] = new_id
    new_key = old_key.replace(old_id, new_id)
    return new_payload, new_key

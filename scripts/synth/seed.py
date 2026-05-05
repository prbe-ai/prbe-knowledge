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

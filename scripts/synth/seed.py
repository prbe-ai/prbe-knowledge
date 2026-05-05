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

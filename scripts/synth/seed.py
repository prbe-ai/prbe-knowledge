"""Synthetic tenant seeding — admin-triggered population of a real-shape
customer workspace with canonical synthetic content.

See docs/superpowers/specs/2026-05-04-synth-plan-4-tenant-seeding-v1-design.md.
"""

from __future__ import annotations

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

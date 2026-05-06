"""Unit tests for is_seed_eligible — the gate that lets non-eval-prefix
tenants be seeded when customers.metadata.allow_synth_seed is set.
"""

import pytest

from scripts.synth.seed import is_seed_eligible


@pytest.mark.parametrize(
    "customer_id,metadata,expected",
    [
        # Eval-prefix tenants are always eligible regardless of metadata.
        ("cust-eval-foo", {}, True),
        ("cust-eval-foo", None, True),
        ("cust-synth-bar", {}, True),
        ("cust-synth-bar", None, True),
        ("cust-eval-foo", {"allow_synth_seed": False}, True),
        # Real-shape tenants need the metadata flag set.
        ("cust-prbe-acme-co", {"allow_synth_seed": True}, True),
        # Real-shape without flag: ineligible.
        ("cust-prbe-acme-co", {}, False),
        ("cust-prbe-acme-co", None, False),
        ("cust-prbe-acme-co", {"allow_synth_seed": False}, False),
        ("cust-prbe-acme-co", {"other_key": True}, False),
        # The flag must be Python True (not truthy strings or ints).
        ("cust-prbe-acme-co", {"allow_synth_seed": "true"}, False),
        ("cust-prbe-acme-co", {"allow_synth_seed": 1}, False),
    ],
)
def test_is_seed_eligible(customer_id, metadata, expected):
    assert is_seed_eligible(customer_id, metadata) is expected

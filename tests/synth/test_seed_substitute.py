"""Unit tests for _substitute_customer_id — rewrites canonical envelopes for
a target customer at seed time."""

import pytest

from scripts.synth.seed import _substitute_customer_id


def test_rewrites_top_level_customer_id():
    payload = {
        "customer_id": "cust-eval-canonical-v1",
        "source": "slack",
        "event_id": "std-001",
        "body": "daily standup",
    }
    new_payload, new_key = _substitute_customer_id(
        payload=payload,
        old_key="raw/slack/cust-eval-canonical-v1/synth/std-001.json",
        old_id="cust-eval-canonical-v1",
        new_id="cust-prbe-acme-co",
    )
    assert new_payload["customer_id"] == "cust-prbe-acme-co"
    assert new_key == "raw/slack/cust-prbe-acme-co/synth/std-001.json"


def test_leaves_other_fields_untouched():
    payload = {
        "customer_id": "cust-eval-canonical-v1",
        "source": "slack",
        "event_id": "std-001",
        "body": "Yesterday: closed PRs. Today: review.",
        "thread_parent_id": "cust-eval-canonical-v1:thread-99",
    }
    new_payload, _ = _substitute_customer_id(
        payload=payload,
        old_key="raw/slack/cust-eval-canonical-v1/synth/std-001.json",
        old_id="cust-eval-canonical-v1",
        new_id="cust-prbe-acme-co",
    )
    # Top-level customer_id rewritten; other fields preserved verbatim.
    # (thread_parent_id is intentionally not rewritten in V1 — it would
    # require deep traversal and the customer_id segment inside it has no
    # downstream consumer that cares.)
    assert new_payload["body"] == "Yesterday: closed PRs. Today: review."
    assert new_payload["source"] == "slack"
    assert new_payload["event_id"] == "std-001"
    # V1 scope: thread_parent_id is intentionally NOT rewritten even though it
    # contains the canonical customer_id as a substring (no downstream consumer
    # parses this field as a customer_id — verified in slack/base.py).
    assert new_payload["thread_parent_id"] == "cust-eval-canonical-v1:thread-99"


def test_idempotent_on_repeat_application():
    payload = {"customer_id": "cust-eval-canonical-v1", "source": "slack"}
    new_payload, new_key = _substitute_customer_id(
        payload=payload,
        old_key="raw/slack/cust-eval-canonical-v1/synth/x.json",
        old_id="cust-eval-canonical-v1",
        new_id="cust-prbe-acme-co",
    )
    new_payload2, new_key2 = _substitute_customer_id(
        payload=dict(new_payload),  # avoid in-place mutation aliasing
        old_key=new_key,
        old_id="cust-eval-canonical-v1",
        new_id="cust-prbe-acme-co",
    )
    assert new_payload2 == new_payload
    assert new_key2 == new_key


def test_raises_if_old_id_not_in_key():
    with pytest.raises(ValueError, match="old_id not found"):
        _substitute_customer_id(
            payload={"customer_id": "wrong"},
            old_key="raw/slack/wrong/synth/x.json",
            old_id="cust-eval-canonical-v1",
            new_id="cust-prbe-acme-co",
        )

"""Typed status booleans survive ingestion without exempting credential values."""

import json

import pytest

from engine.ingest._credential_gate import CredentialBlocked, inspect_bytes
from engine.ingest._credential_redaction import default_scrub
from engine.ingest.payload_redaction import redact_payload


@pytest.mark.parametrize("value", [True, False])
def test_typed_boolean_status_survives_combined_ingestion_scrubber(value):
    payload = {"nested": {"synthetic_credentials_absent": value, "api_key": value}}
    assert redact_payload(payload) == payload


@pytest.mark.parametrize("value", [True, False])
def test_serialized_boolean_report_keeps_original_bytes(value):
    raw = json.dumps({"synthetic_credentials_absent": value}, indent=2)
    assert default_scrub(raw) == raw
    inspect_bytes(raw.encode())


@pytest.mark.parametrize("value", ["true", "false", "synthetic-credential", 123456, 12.34])
def test_sensitive_strings_and_numbers_still_redact_and_block(value):
    payload = {"synthetic_credentials_absent": value}
    assert redact_payload(payload) == {"synthetic_credentials_absent": "<redacted>"}
    with pytest.raises(CredentialBlocked):
        inspect_bytes(json.dumps(payload).encode())


def test_vendor_credential_key_with_boolean_value_is_still_removed():
    key = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    scrubbed = redact_payload({key: True})
    assert key not in json.dumps(scrubbed)
    assert list(scrubbed.values()) == [True]
    with pytest.raises(CredentialBlocked):
        inspect_bytes(json.dumps({key: True}).encode())


def test_untyped_boolean_assignment_is_not_exempt():
    assert default_scrub("password=true") != "password=true"
    with pytest.raises(CredentialBlocked):
        inspect_bytes(b"password=true")

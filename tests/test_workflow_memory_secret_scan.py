"""Ingest-time credential detection for declared rule text.

A rule is free-form prose a human typed, and people paste. Team precedent:
a .env already leaked into snapshots once.
"""

from __future__ import annotations

import pytest

from shared.wfmem.secret_scan import (
    SecretDetected,
    assert_clean,
    assert_clean_json,
    scan_for_secrets,
)


@pytest.mark.parametrize(
    "text,detector",
    [
        ("use AKIAIOSFODNN7EXAMPLE for the upload", "aws_access_key_id"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n", "private_key_block"),
        ("export PROBE_API_KEY=sk" "-live-9f2b7c1d4e6a8b0c2d4e6f8a", "assigned_secret"),
        ("ghp" "_16C7e42F292c6912E7710c838347Ae178B4a", "github_token"),
        # Fine-grained PAT: the mid-token underscore is why the detectors carry
        # no trailing \b. This is the case the first draft missed entirely.
        ("github" "_pat_11ABCDEFG0abcdefghijkl_MNOPqrstuvwxyz012345", "github_fine_grained_pat"),
        ("key is sk" "-ant-api03-abcdefghij0123456789-KLMNOP", "anthropic_api_key"),
        ("sk" "-proj-abcdefghij0123456789klmnopqr", "openai_api_key"),
        (
            "Authorization: Bearer eyJ" "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
            "jwt",
        ),
        ("DATABASE_URL=postgresql://prbe:hunter2pass@localhost:5432/db", "dsn_with_password"),
        (
            "https://hooks.slack.com" "/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
            "slack_webhook",
        ),
    ],
)
def test_known_credential_shapes_are_detected(text, detector):
    assert detector in scan_for_secrets(text)


@pytest.mark.parametrize(
    "text",
    [
        "always open a Probe run before the first GPU step",
        "never edit the canonical clone; use a worktree",
        "run scripts/smoke.sh against Docker Postgres before claiming done",
        "set PROBE_ASYNC=1 for the whole session",
        # A DSN with no password is a connection string, not a credential.
        "use postgresql://localhost:5432/prbe_knowledge for local dev",
        "the endpoint is https://api.research.prbe.ai/v1/search",
        "bump PROMPT_VERSION from 3 to 4",
    ],
)
def test_ordinary_rule_text_is_clean(text):
    assert scan_for_secrets(text) == []


def test_assert_clean_raises_on_secret():
    with pytest.raises(SecretDetected) as exc:
        assert_clean("token is ghp" "_16C7e42F292c6912E7710c838347Ae178B4a")
    assert "github_token" in str(exc.value)


def test_assert_clean_passes_on_ordinary_text():
    assert_clean("always log runs to both W&B and Probe")


def test_binding_jsonb_is_scanned_not_just_body():
    """`binding` is a dict, and the spec requires the scan on body AND binding."""
    with pytest.raises(SecretDetected):
        assert_clean_json(
            {"argv_template": "deploy --token ghp" "_16C7e42F292c6912E7710c838347Ae178B4a"}
        )


def test_binding_jsonb_scans_nested_values_and_keys():
    with pytest.raises(SecretDetected):
        assert_clean_json({"env": [{"AWS": "AKIAIOSFODNN7EXAMPLE"}]})


def test_ordinary_binding_passes():
    assert_clean_json(
        {"asset_refs": ["research-os:evals/validate_cases.py@HEAD"], "cwd_glob": "~/Desktop/prbe/*"}
    )

"""Ingest-time credential detection for declared rule text.

A rule is free-form prose a human typed, and people paste. Team precedent:
a .env already leaked into snapshots once. Also covers the hardening pass
after code review: ReDoS on the jwt detector, RecursionError on deep/cyclic
JSONB, missed credential shapes (Stripe, Google, AWS secret keys), and
zero-width/bidi unicode evasion.
"""

from __future__ import annotations

import time

import pytest

from shared.wfmem.secret_scan import (
    MAX_JSON_DEPTH,
    MAX_SCAN_CHARS,
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
        # Fine-grained PAT: its own character class already includes `_`, so a
        # trailing \b would NOT have broken it here. The detector that genuinely
        # needs no trailing \b is `github_token` (see the module-level note on
        # `_DETECTORS`); this fixture just exercises the fine-grained format.
        ("github" "_pat_11ABCDEFG0abcdefghijkl_MNOPqrstuvwxyz012345", "github_fine_grained_pat"),
        ("key is sk" "-ant-api03-abcdefghij0123456789-KLMNOP", "anthropic_api_key"),
        ("sk" "-proj-abcdefghij0123456789klmnopqr", "stripe_or_openai_key"),
        ("sk" "_live_51H8xyzABCDEFGHIJKLMNOPQRSTUVWX", "stripe_or_openai_key"),
        ("AIza" "SyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY", "google_api_key"),
        (
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI" "/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "assigned_secret",
        ),
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


# --- CRITICAL 1: jwt ReDoS + input cap -------------------------------------


def test_jwt_detector_does_not_backtrack_quadratically():
    """A pasted blob with a bare `eyJ` and no real JWT must not hang ingest.

    Before the atomic-group fix this was O(n^2): ~1.84s at 32,000 chars.
    `MAX_SCAN_CHARS` alone would already bound this, but the atomic groups
    are the actual ReDoS fix and are asserted directly here by timing a blob
    far longer than the cap.
    """
    text = "eyJ" + "a." * 100_000
    start = time.perf_counter()
    result = scan_for_secrets(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"scan_for_secrets took {elapsed:.3f}s, expected < 1.0s"
    assert "oversized_input" in result


def test_oversized_input_is_flagged_and_prefix_still_scanned():
    clean_blob = "x" * 25_000
    assert len(clean_blob) > MAX_SCAN_CHARS
    assert scan_for_secrets(clean_blob) == ["oversized_input"]


def test_oversized_input_with_credential_in_prefix_reports_both():
    blob = "AKIAIOSFODNN7EXAMPLE" + ("x" * 25_000)
    result = scan_for_secrets(blob)
    assert "aws_access_key_id" in result
    assert "oversized_input" in result


def test_assert_clean_refuses_oversized_input():
    with pytest.raises(SecretDetected) as exc:
        assert_clean("x" * 25_000)
    assert "oversized_input" in str(exc.value)


# --- CRITICAL 2: no RecursionError on deep/cyclic JSONB ---------------------


def _make_deep_dict(depth: int) -> dict:
    root: dict = {}
    cur = root
    for _ in range(depth):
        nxt: dict = {}
        cur["next"] = nxt
        cur = nxt
    return root


def test_deeply_nested_binding_raises_secret_detected_not_recursion_error():
    deep = _make_deep_dict(2000)
    assert MAX_JSON_DEPTH < 2000
    with pytest.raises(SecretDetected) as exc:
        assert_clean_json(deep)
    assert "unscannable_structure" in str(exc.value)


def test_self_referential_dict_raises_secret_detected_not_hang():
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(SecretDetected) as exc:
        assert_clean_json(cyclic)
    assert "unscannable_structure" in str(exc.value)


def test_normal_shallow_binding_still_scans_correctly():
    nested = {
        "level1": {
            "level2": {
                "level3": ["clean value", "another clean value"],
            }
        }
    }
    assert_clean_json(nested)  # does not raise

    nested_dirty = {
        "level1": {"level2": {"level3": ["AKIAIOSFODNN7EXAMPLE"]}},
    }
    with pytest.raises(SecretDetected):
        assert_clean_json(nested_dirty)


# --- IMPORTANT 4: unicode evasion -------------------------------------------


def test_zero_width_space_inside_aws_key_is_still_detected():
    assert "aws_access_key_id" in scan_for_secrets("AKIA​IOSFODNN7EXAMPLE")


def test_zero_width_joiner_inside_github_token_is_still_detected():
    assert "github_token" in scan_for_secrets("ghp‍_16C7e42F292c6912E7710c838347Ae178B4a")


def test_rtl_override_inside_aws_key_is_still_detected():
    assert "aws_access_key_id" in scan_for_secrets("AKIA‮IOSFODNN7EXAMPLE")


# --- IMPORTANT 6: length-threshold boundaries -------------------------------


def test_aws_access_key_id_boundary():
    assert scan_for_secrets("AKIA" + "A" * 15) == []
    assert "aws_access_key_id" in scan_for_secrets("AKIA" + "A" * 16)


def test_assigned_secret_boundary():
    assert scan_for_secrets("TOKEN=" + "a" * 15) == []
    assert "assigned_secret" in scan_for_secrets("TOKEN=" + "a" * 16)

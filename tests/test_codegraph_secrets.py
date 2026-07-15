"""Unit tests for the code-graph secrets skip-list."""

from __future__ import annotations

import pytest

from kb.code_graph.secrets import (
    SKIPPED_LANGUAGE_SENTINEL,
    looks_like_secret_dump,
)


def test_skips_aws_access_key() -> None:
    content = b"# config\nAWS_KEY = 'AKIAIOSFODNN7EXAMPLE'\n"
    assert looks_like_secret_dump("config.py", content) is True


def test_skips_pem_private_key() -> None:
    content = (
        b"-----BEGIN RSA PRIVATE KEY-----\n"
        b"MIIEogIBAAKCAQEA...\n"
        b"-----END RSA PRIVATE KEY-----\n"
    )
    assert looks_like_secret_dump("server.key", content) is True


def test_skips_stripe_live_key() -> None:
    content = b"export const STRIPE = 'sk_live_abcdefghijklmnopqrstuvwx'\n"
    assert looks_like_secret_dump("billing.ts", content) is True


def test_skips_github_pat() -> None:
    content = b"token: ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    assert looks_like_secret_dump("setup.md", content) is True


def test_skips_anthropic_key() -> None:
    content = b"client = Anthropic(api_key='sk-ant-api03-AbCdEfGhIjKlMnOpQrSt')\n"
    assert looks_like_secret_dump("agent.py", content) is True


def test_skips_dotenv_files_by_name() -> None:
    assert looks_like_secret_dump(".env", b"PORT=3000\n") is True
    assert looks_like_secret_dump(".env.production", b"") is True
    assert looks_like_secret_dump(".env.local", b"PORT=3000\n") is True
    assert looks_like_secret_dump("path/to/.env", b"") is True


def test_skips_credentials_json() -> None:
    assert looks_like_secret_dump("credentials.json", b"{}") is True
    assert (
        looks_like_secret_dump("path/to/service-account.json", b"{}") is True
    )


def test_skips_path_with_secrets_fragment() -> None:
    assert (
        looks_like_secret_dump("config/secrets/db.py", b"x = 1\n") is True
    )
    assert (
        looks_like_secret_dump("path/.secrets/api.py", b"x = 1\n") is True
    )
    assert (
        looks_like_secret_dump("internal/credentials/auth.py", b"x = 1\n") is True
    )


def test_does_not_skip_normal_python() -> None:
    content = b"def add(a, b):\n    return a + b\n"
    assert looks_like_secret_dump("math.py", content) is False


def test_does_not_skip_normal_typescript() -> None:
    content = b"export function add(a: number, b: number) { return a + b }\n"
    assert looks_like_secret_dump("math.ts", content) is False


def test_scan_bounds_to_first_64kb() -> None:
    """Pathological large file with a key past the 64KB head shouldn't trigger.

    Bounded scan keeps CPU cost on multi-MB lockfiles in check; a real key
    almost always lands near the top of a file anyway.
    """
    head = b"safe content\n" * 8000  # ~104KB
    tail = b"AKIAIOSFODNN7EXAMPLE in tail\n"
    content = head + tail
    # Confirm we're past the 64KB head cap.
    assert len(head) > 65536
    assert looks_like_secret_dump("big.txt", content) is False


def test_skipped_sentinel_constant() -> None:
    assert SKIPPED_LANGUAGE_SENTINEL == "_skipped_secrets"


@pytest.mark.parametrize(
    "filename",
    [
        "test_secrets.py",     # `secret` in name (not the path fragment) — should NOT trigger
        "secrets_helper.go",   # tests for secrets behavior, not a dump
        "src/utils/auth.py",
    ],
)
def test_filename_with_secret_word_does_not_trigger(filename: str) -> None:
    """`secrets` substring in a filename != path fragment match.

    Only fragments like `/secrets/`, `/.secrets/`, `/credentials/` trigger.
    A normal source file with 'secret' in its name doesn't get skipped.
    """
    content = b"def add(a, b):\n    return a + b\n"
    assert looks_like_secret_dump(filename, content) is False

"""Detect credentials in text on its way into the procedure store.

Declared rules are free-form prose typed by a human, and humans paste. This
runs on every clause body and binding write. It is a coarse net by design:
false positives cost one confused author, a false negative puts a live
credential in a store that other people's agents read.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

#: Every detector runs and every match is reported -- detectors overlap on
#: purpose (an `sk-ant-...` key trips both the Anthropic and the generic `sk-`
#: rule) and a credential matching two rules is not less of a credential.
#:
#: NOTE ON TRAILING \b: there is none, deliberately. `_` is a word character but
#: is excluded from the token character classes, so a trailing \b would make
#: `github_pat_11ABC..._xyz` -- GitHub's fine-grained PAT format, which carries a
#: mid-token underscore -- fail to match at the underscore boundary. The most
#: common modern GitHub credential would have sailed straight through.
_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("github_fine_grained_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    (
        "slack_webhook",
        re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/+_-]{10,}"),
    ),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("jwt", re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}")),
    # A DSN is only a secret when it carries an inline password. The {3,} on the
    # password segment is what keeps `postgresql://localhost:5432/db` clean.
    ("dsn_with_password", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]{3,}@")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|APIKEY|API_KEY|ACCESS_KEY)[A-Z0-9_]*"
            r"\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})"
        ),
    ),
)


class SecretDetected(ValueError):
    """Text carrying a credential was refused at ingest."""

    def __init__(self, detectors: list[str]) -> None:
        self.detectors = detectors
        super().__init__(
            "refusing to store text containing what looks like a credential: "
            + ", ".join(detectors)
        )


def scan_for_secrets(text: str) -> list[str]:
    """Return the names of every detector that matched. Empty means clean."""
    if not text:
        return []
    return [name for name, pattern in _DETECTORS if pattern.search(text)]


def assert_clean(text: str) -> None:
    """Raise `SecretDetected` if `text` carries anything credential-shaped."""
    found = scan_for_secrets(text)
    if found:
        raise SecretDetected(found)


def _walk_strings(value: Any) -> Iterator[str]:
    """Every string anywhere inside a JSON-shaped value, keys included.

    Keys are scanned too: `{"AWS_SECRET_ACCESS_KEY": "..."}` hides the credential
    in the value but a pasted blob can just as easily invert that.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk_strings(item)


def assert_clean_json(value: Any) -> None:
    """`assert_clean` for a JSONB-bound structure such as `clauses.binding`.

    The spec requires the scan on every `body` AND `binding` write, and `binding`
    is a dict -- a str-only entry point would have left half the requirement
    unimplementable, which is how a guarantee quietly becomes paper.
    """
    found: list[str] = []
    for text in _walk_strings(value):
        for name in scan_for_secrets(text):
            if name not in found:
                found.append(name)
    if found:
        raise SecretDetected(found)

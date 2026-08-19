"""Detect credentials in text on its way into the procedure store.

Declared rules are free-form prose typed by a human, and humans paste. This
runs on every clause body and binding write. It is a coarse net by design:
false positives cost one confused author, a false negative puts a live
credential in a store that other people's agents read.

Two pseudo-detector names can ride alongside the credential-shape ones in a
scan result, and both mean "refused", not "credential found": `oversized_input`
fires when the text exceeds `MAX_SCAN_CHARS` (a paste that size gets scanned
only in its prefix, so it is refused outright rather than silently accepted
past the point we actually checked), and `unscannable_structure` fires when a
JSONB value handed to `assert_clean_json` is too deep or cyclic to fully walk.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

#: Every detector runs and every match is reported -- detectors overlap on
#: purpose (an `sk-ant-...` key trips both the Anthropic and the generic `sk-`
#: rule) and a credential matching two rules is not less of a credential.
#:
#: NOTE ON TRAILING \b: there is none, deliberately. The rule: a trailing \b
#: breaks any detector whose character class excludes `_` when the credential
#: is immediately followed by `_`. `github_token`'s class is `[A-Za-z0-9]`,
#: which excludes `_` -- so `ghp_<36 chars>_suffix` matches without a trailing
#: \b and fails to match with one (verified: True vs False). Fine-grained PATs
#: are unaffected by this: `github_fine_grained_pat`'s own class already
#: includes `_`, so a trailing \b would not have broken it there.
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
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("stripe_or_openai_key", re.compile(r"\bsk[-_](?:proj[-_])?[A-Za-z0-9_\-]{20,}")),
    # Atomic groups ((?>...)) so a near-miss (a long `eyJ...`-prefixed blob with
    # no two `.`-delimited runs) fails fast instead of backtracking char-by-char
    # across every possible split point -- that backtracking was O(n^2) on
    # pasted blobs containing a bare "eyJ" with no real JWT, up to ~2s at 64KB.
    (
        "jwt",
        re.compile(r"\beyJ(?>[A-Za-z0-9_-]{10,})\.(?>[A-Za-z0-9_-]{10,})\.(?>[A-Za-z0-9_-]{10,})"),
    ),
    # A DSN is only a secret when it carries an inline password. `{3,}` is NOT
    # what keeps `postgresql://localhost:5432/db` clean -- that string has no
    # @-delimited userinfo at all, so it stays clean even at `{1,}`. What `{3,}`
    # actually guards against is a placeholder DSN like
    # `postgresql://user:x@host:5432/db`, which is a template, not a leak.
    ("dsn_with_password", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]{3,}@")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|APIKEY|API_KEY|ACCESS_KEY)[A-Z0-9_]*"
            r"\s*[=:]\s*['\"]?([A-Za-z0-9_\-/+]{16,})"
        ),
    ),
)

#: Longer than any plausible rule and long enough that no detector's
#: backtracking cost matters. A blob this size is a paste, not a procedure.
MAX_SCAN_CHARS = 20_000

#: `assert_clean_json` walks a JSONB value iteratively (see `_walk_strings`);
#: this bounds how deep that walk goes before it refuses the structure rather
#: than trust it is finite.
MAX_JSON_DEPTH = 50

#: Zero-width and bidi-control code points stripped before scanning. Rich-text
#: paste from Notion/Slack/Docs injects these routinely -- a ZWJ dropped
#: mid-token (`ghp_...`) breaks the literal-adjacency match every detector
#: relies on, which would otherwise let the credential behind it through.
_ZERO_WIDTH_AND_BIDI = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0xFEFF, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E],
    None,
)


class SecretDetected(ValueError):
    """Text carrying a credential was refused at ingest."""

    def __init__(self, detectors: list[str]) -> None:
        self.detectors = detectors
        super().__init__(
            "refusing to store text containing what looks like a credential: "
            + ", ".join(detectors)
        )


def _normalize(text: str) -> str:
    """Undo zero-width/bidi noise before scanning, fold NBSP to a plain space.

    Deletes the code points in `_ZERO_WIDTH_AND_BIDI` outright (they carry no
    visible content, so removing them restores the adjacency a detector's
    character class expects) and translates U+00A0 (NBSP) to an ordinary
    space (it is visible whitespace, so it is folded rather than deleted).
    """
    text = text.translate(_ZERO_WIDTH_AND_BIDI)
    return text.replace("\xa0", " ")


def scan_for_secrets(text: str) -> list[str]:
    """Return the names of every detector that matched. Empty means clean.

    `text` is normalized first (see `_normalize`) and then capped at
    `MAX_SCAN_CHARS`: only the prefix is scanned, and if the normalized text
    was longer than the cap, `"oversized_input"` is added to the result so an
    over-length paste is refused rather than silently accepted past the point
    it was actually checked.
    """
    if not text:
        return []
    normalized = _normalize(text)
    oversized = len(normalized) > MAX_SCAN_CHARS
    scan_text = normalized[:MAX_SCAN_CHARS]
    found = [name for name, pattern in _DETECTORS if pattern.search(scan_text)]
    if oversized:
        found.append("oversized_input")
    return found


def assert_clean(text: str) -> None:
    """Raise `SecretDetected` if `text` carries anything credential-shaped."""
    found = scan_for_secrets(text)
    if found:
        raise SecretDetected(found)


def _walk_strings(value: Any) -> Iterator[str]:
    """Every string anywhere inside a JSON-shaped value, keys included.

    Keys are scanned too: `{"AWS_SECRET_ACCESS_KEY": "..."}` hides the credential
    in the value but a pasted blob can just as easily invert that.

    Iterative and stack-based, not recursive: a >`MAX_JSON_DEPTH`-deep or
    self-referential structure must be refused, not crash the process with a
    `RecursionError` a caller's `except SecretDetected` cannot catch. Object
    identity (`id()`) tracked in `seen` catches cycles AND any other repeat
    visit to the same object; either way we cannot promise we scanned the
    whole thing, so we refuse rather than silently return a partial (and
    possibly falsely "clean") result. `bytes` and any other non-str/Mapping/
    list/tuple/set leaf are skipped on purpose, not by oversight -- the JSONB
    contract this scans never carries them, so there is nothing to walk into.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    while stack:
        item, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise SecretDetected(["unscannable_structure"])
        if isinstance(item, str):
            yield item
        elif isinstance(item, Mapping):
            item_id = id(item)
            if item_id in seen:
                raise SecretDetected(["unscannable_structure"])
            seen.add(item_id)
            for key, child in item.items():
                if isinstance(key, str):
                    yield key
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple, set)):
            item_id = id(item)
            if item_id in seen:
                raise SecretDetected(["unscannable_structure"])
            seen.add(item_id)
            for child in item:
                stack.append((child, depth + 1))


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

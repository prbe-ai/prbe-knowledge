"""Pre-parse skip-list — drops files that look like secret dumps.

Detection runs BEFORE tree-sitter parse. Files that match the regex floor
or have a known secret-vault filename are skipped entirely: no symbols
emitted, no chunks embedded. `code_repo_state` records `language=
'_skipped_secrets'` so we don't re-attempt on every push.

This is a deliberate floor, not a complete solution. Full secret-scanners
(detect-secrets, gitleaks, TruffleHog) with entropy + file-context
heuristics are a v2 follow-up. The patterns here are the obvious cases
that are dangerous to embed into the vector store.

False positives are acceptable — we'd rather miss indexing one file with
a literal AWS key in a docstring than embed a real one and have it
surface in retrieval results.
"""

from __future__ import annotations

import re

# Regex floor. Tuned for common cases that survive `git push` despite linters.
_SECRET_PATTERNS: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),                         # AWS access key id
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    re.compile(rb"sk_live_[0-9a-zA-Z]{24,}"),                 # Stripe live secret
    re.compile(rb"rk_live_[0-9a-zA-Z]{24,}"),                 # Stripe restricted key
    re.compile(rb"ghp_[0-9a-zA-Z]{36}"),                      # GitHub PAT (classic)
    re.compile(rb"github_pat_[0-9a-zA-Z_]{82}"),              # GitHub PAT (fine-grained)
    re.compile(rb"xox[baprs]-[0-9]{10,}-[0-9a-zA-Z-]+"),      # Slack token
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),                    # Google API key
    re.compile(rb"ya29\.[0-9A-Za-z_-]{20,}"),                 # Google OAuth refresh
    re.compile(rb"sk-[a-zA-Z0-9]{20,}"),                      # OpenAI / Anthropic-style API key
    re.compile(rb"sk-ant-[a-zA-Z0-9_-]{20,}"),                # Anthropic API key
    re.compile(rb"npm_[a-zA-Z0-9]{36}"),                      # npm access token
)

# Filename guards: these almost never contain code worth indexing.
_SECRET_FILENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        ".env.development",
        ".env.test",
        "credentials.json",
        "service-account.json",
        "service-account-key.json",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        ".npmrc",
        ".pypirc",
    }
)

# Path fragments that signal a secrets directory.
_SECRET_PATH_FRAGMENTS: tuple[str, ...] = (
    "/secrets/",
    "/.secrets/",
    "/credentials/",
)

# Cap regex scan at the first N bytes — bounds CPU cost on large files
# (a generated 50MB lockfile shouldn't burn CPU on regex). Most legit
# secrets land near the top anyway.
_SCAN_HEAD_BYTES = 65536


def looks_like_secret_dump(file_path: str, content: bytes) -> bool:
    """Return True if the file should be skipped pre-parse.

    Cheap checks first: filename, path fragment. Regex scan is bounded to
    the first 64KB of the file. Designed to fail closed (skip) on
    ambiguous cases.
    """
    name = file_path.rsplit("/", 1)[-1]
    if name in _SECRET_FILENAMES:
        return True
    lower_path = file_path.lower()
    for fragment in _SECRET_PATH_FRAGMENTS:
        if fragment in lower_path:
            return True
    head = content[:_SCAN_HEAD_BYTES]
    return any(p.search(head) for p in _SECRET_PATTERNS)


# Sentinel language id we record in code_repo_state when a file is skipped.
SKIPPED_LANGUAGE_SENTINEL = "_skipped_secrets"


__all__ = ["SKIPPED_LANGUAGE_SENTINEL", "looks_like_secret_dump"]

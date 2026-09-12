"""Server-side credential redaction, one shared boundary, before persistence.

WHY A SERVER SIDE AT ALL
------------------------
The tap redacts on the researcher's machine, which is strictly better: the
value never leaves the host. But the tap is user-installed and version-skewed,
so an old one keeps shipping unredacted, and two ingest lanes never touch it at
all — `custom_ingest` (workspace files) and `manual_uploads`. Whatever the tap
misses arrives here.

WHY THE REAL GITLEAKS AND NOT A PORT
------------------------------------
Hand-porting gitleaks' 222-rule corpus into Python was measured at 2.9s for a
110KB document (Python's `re` backtracks where Go's RE2 is linear by
construction) and would re-implement its capture-group selection, allowlists,
entropy inputs and scoped flags without ever specifying compatibility with any
of them. Here we can ship the binary, so we run the real thing.

`probe_rules.toml` extends the default corpus with ONE rule. Stock gitleaks
does not catch `AWS Secret Access Key [None]: <40 chars>` — the literal shape
that leaked from a customer's session on 2026-08-30 — because an AWS secret has
no format of its own and gitleaks' generic rule wants an `=`-style assignment.
`probe-anchored-secret` anchors on the key NAME instead. Measured against this
repo's false-positive corpus (the 615 drops that killed the previous gate):
0 findings before the extension, 0 after.

THE RESPONSE IS ALWAYS "REPLACE", NEVER "REJECT"
------------------------------------------------
The gate removed in 0.104.9.0 quarantined a whole session on any finding and
refused every later batch, so one false positive erased a transcript
permanently and silently. Nothing here can drop a document, a chunk, or a
batch. The worst a false positive can do is replace one string, and every
replacement is recorded where the customer can see it.

FAIL-OPEN, LOUDLY
-----------------
A missing binary, a crash or a timeout means the text passes through unredacted
with a warning and a metric. Ingestion continuing is the right call — the tap
is the primary control and this is the backstop — but a silent backstop is not
a backstop, so this never fails quietly.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

#: Extension config: the stock corpus plus `probe-anchored-secret`.
RULES_PATH = Path(__file__).with_name("probe_rules.toml")

#: Override for tests and for images that install the binary elsewhere.
_BIN_ENV = "PROBE_GITLEAKS_BIN"

#: A scan is a subprocess. Bound it so a pathological document cannot wedge an
#: ingestion worker; on expiry the text passes through unredacted and loudly.
SCAN_TIMEOUT_SECONDS = 20.0

#: Hard ceiling on one scan. Beyond this the input IS truncated, and the tail
#: goes unscanned — a real blind spot, logged every time so it is never a
#: surprise. (An earlier comment here claimed large documents were "scanned in
#: full anyway", which the line below has never done.)
MAX_SCAN_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Redaction:
    """One replaced value. Deliberately has NO field for the value itself."""

    rule: str
    #: Where in the document it was, for a customer-visible record.
    line: int


def binary_path() -> str | None:
    """The gitleaks binary, or None when this deployment has none."""
    override = os.environ.get(_BIN_ENV)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("gitleaks")


def available() -> bool:
    return binary_path() is not None


def find_secrets(text: str) -> list[tuple[str, str, int]]:
    """`(rule_id, secret_value, line)` for every credential gitleaks finds.

    The secret values are returned so the caller can replace them literally,
    which avoids mapping gitleaks' line/column spans back through a chunker
    that may have re-wrapped the text. They are held in memory for the length
    of one call and MUST NOT be logged, persisted, or put in an exception.
    """
    if not text or not text.strip():
        return []
    binary = binary_path()
    if binary is None:
        log.warning("secret_redaction.binary_missing", env_var=_BIN_ENV)
        return []

    encoded = text.encode("utf-8", errors="replace")
    payload = encoded[:MAX_SCAN_BYTES]
    if len(encoded) > MAX_SCAN_BYTES:
        log.warning(
            "secret_redaction.input_truncated",
            total_bytes=len(encoded),
            scanned_bytes=MAX_SCAN_BYTES,
            unscanned_bytes=len(encoded) - MAX_SCAN_BYTES,
        )
    cmd = [
        binary, "stdin",
        "--report-format", "json",
        "--report-path", "/dev/stdout",
        "--no-banner",
        # Findings are the normal case here, not an error condition.
        "--exit-code", "0",
    ]
    if RULES_PATH.is_file():
        cmd[2:2] = ["-c", str(RULES_PATH)]
    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            timeout=SCAN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("secret_redaction.timeout", seconds=SCAN_TIMEOUT_SECONDS, bytes=len(payload))
        return []
    except OSError as exc:
        log.warning("secret_redaction.spawn_failed", error=str(exc))
        return []

    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        # The one failure mode that used to be completely silent. A rules file
        # the installed gitleaks rejects, a flag removed by a CLI upgrade, an
        # OOM kill — all produce no stdout and fall through the `if not raw`
        # below, while /health still reports the scanner present because the
        # binary exists. "Fail open, LOUDLY" has to include this one.
        log.warning(
            "secret_redaction.scanner_failed",
            returncode=proc.returncode,
            # stderr is gitleaks' own diagnostics; its stdout can carry secrets
            # on a partial write, so only stderr is ever echoed, and truncated.
            stderr=proc.stderr.decode("utf-8", errors="replace")[:500],
        )
        return []
    if not raw:
        return []
    try:
        findings = json.loads(raw)
    except ValueError:
        # Never echo stdout: on a partial write it can contain the secret.
        log.warning("secret_redaction.unparseable_report", bytes=len(raw))
        return []

    out: list[tuple[str, str, int]] = []
    for item in findings if isinstance(findings, list) else []:
        secret = item.get("Secret")
        rule = item.get("RuleID")
        if isinstance(secret, str) and secret and isinstance(rule, str):
            out.append((rule, secret, int(item.get("StartLine") or 0)))
    return out


def redact_documents(texts: list[str]) -> tuple[list[str], list[Redaction]]:
    """Redact a document's pieces together, in ONE subprocess.

    Scanning each piece separately would cost a process spawn per chunk. The
    pieces are joined for the scan and the found values are then replaced
    literally in every piece — so a value that appears twice goes twice, which
    is what you want from a redactor.

    JOINED WITH "\n", DELIBERATELY. A space would let an anchor on the tail of
    one piece reach a value on the head of the next, inventing a credential
    neither piece contains, and it would destroy the line numbers gitleaks
    reports. Boundaries are already covered upstream: `chunk_text` emits
    OVERLAPPING windows (`stride = chunk_tokens - overlap`), so a credential
    near a boundary appears whole in at least one piece. Pinned by
    tests/test_secret_redaction.py.
    """
    if not texts:
        return texts, []
    joined = "\n".join(texts)
    found = find_secrets(joined)
    if not found:
        return texts, []

    # LONGEST FIRST. Two rules legitimately match the same credential with
    # different extents — a scoped rule and `generic-api-key` on one URI, say.
    # Replacing the shorter one first destroys the longer literal, so the
    # longer replace() silently no-ops and the remainder of the longer match
    # survives in the text. Sorting by length makes that impossible.
    replacements = sorted(
        {secret: rule for rule, secret, _ in found}.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    out: list[str] = []
    for piece in texts:
        redacted = piece
        for secret, rule in replacements:
            if secret in redacted:
                redacted = redacted.replace(secret, f"<redacted:{rule}>")
        out.append(redacted)
    return out, [Redaction(rule=rule, line=line) for rule, _, line in found]


async def redact_documents_async(texts: list[str]) -> tuple[list[str], list[Redaction]]:
    """`redact_documents` off the event loop.

    The scan is a blocking `subprocess.run` with a 20s ceiling, and both real
    callers are coroutines on the ingestion worker's loop. Called directly,
    one large document stalls EVERYTHING that loop is doing for the length of
    the scan — including the health endpoint, whose DB ping shares the loop, so
    a slow scan reads as an unhealthy pod and gets it restarted mid-ingestion.
    """
    return await asyncio.to_thread(redact_documents, texts)

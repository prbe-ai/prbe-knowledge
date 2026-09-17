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

FAIL-CLOSED, AND WHY THAT IS NOT THE GATE WE REMOVED
----------------------------------------------------
A scan that cannot produce a verdict raises `ScanUnavailable` (transient), so
the queue row retries and the text is never persisted unscanned. Until
2026-09-17 every failure here returned `[]` -- indistinguishable from "clean" --
and the text was stored and served. That is not theoretical: 328 scans timed
out in 22 hours on 2026-09-16 alone, each one a document stored unscanned, and
a live AWS key pair sat in five searchable chunks of one customer's session for
ten days.

This is NOT the 0.104.9.0 gate. That one rejected on a FINDING, so one false
positive destroyed a transcript permanently and silently. Nothing here reacts
to a finding except by replacing a substring. This reacts to the SCANNER
FAILING, the payload stays in R2, the row retries up to `worker_max_attempts`
(50), and only then does it dead-letter -- with the reason attached.

`secret_redaction_fail_closed=false` restores the old behavior for a
deployment that ships without the binary on purpose. It is a deployment-shape
switch, not a per-scan one: with it off, unscanned text IS stored, and the
warning says so every time.

NO TRUNCATION
-------------
An oversize document is scanned in overlapping windows rather than cut at
`MAX_SCAN_BYTES`. The old code truncated and logged, so the tail of a large
transcript was never scanned at all -- the module's own comment admitted it.
The overlap is `SCAN_WINDOW_OVERLAP_BYTES` so a credential lying across a
window boundary still appears whole in one of them.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import structlog

from engine.ingest import redactd
from engine.shared.config import get_settings
from engine.shared.exceptions import ScanUnavailable

log = structlog.get_logger(__name__)

#: Extension config: the stock corpus plus `probe-anchored-secret`.
RULES_PATH = Path(__file__).with_name("probe_rules.toml")

#: Override for tests and for images that install the binary elsewhere.
_BIN_ENV = "PROBE_GITLEAKS_BIN"

#: A scan is a subprocess. Bound it so a pathological document cannot wedge an
#: ingestion worker; on expiry the text passes through unredacted and loudly.
SCAN_TIMEOUT_SECONDS = 20.0

#: Bytes per scan window. Input longer than this is scanned in SEVERAL windows,
#: never truncated.
MAX_SCAN_BYTES = 8 * 1024 * 1024

#: Overlap between windows, so a credential lying across a boundary is whole in
#: one of them. Comfortably longer than any credential gitleaks matches.
SCAN_WINDOW_OVERLAP_BYTES = 4096

#: One retry before giving up on a timeout: the 2026-09-16 timeouts were CPU
#: starvation, not pathological input, and a starved scan often clears on the
#: next attempt. Worst case is two windows' timeout, off the event loop.
SCAN_ATTEMPTS = 2


@dataclass(frozen=True)
class Redaction:
    """One replaced value. Deliberately has NO field for the value itself."""

    rule: str
    #: Where in the document it was, for a customer-visible record.
    line: int


_supervisor: redactd.RedactdSupervisor | None = None
_supervisor_lock = threading.Lock()


def _shared_supervisor() -> redactd.RedactdSupervisor:
    """One daemon per process, started on first use.

    Lazy rather than at import: importing this module must not fork anything,
    or every CLI script and test collection that touches it inherits a child.
    """
    global _supervisor
    with _supervisor_lock:
        if _supervisor is None:
            sup = redactd.RedactdSupervisor()
            sup.start()
            _supervisor = sup
        return _supervisor


def shutdown_supervisor() -> None:
    """Stop the process-wide daemon, if one was started."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is not None:
            _supervisor.stop()
            _supervisor = None


def binary_path() -> str | None:
    """The gitleaks binary, or None when this deployment has none."""
    override = os.environ.get(_BIN_ENV)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("gitleaks")


def available() -> bool:
    """Is a scanner present at all? Either transport counts: the sweep uses
    this to refuse to report "0 findings" on a deployment that cannot scan."""
    if not redactd.disabled() and redactd.binary_path() is not None:
        return True
    return binary_path() is not None


def find_secrets(text: str) -> list[tuple[str, str, int]]:
    """`(rule_id, secret_value, line)` for every credential gitleaks finds.

    Raises `ScanUnavailable` when the scanner could not produce a verdict —
    missing binary, spawn failure, non-zero exit, unparseable report, or a
    timeout that survived a retry. An empty list means "scanned, clean", and
    nothing else may mean that.

    The secret values are returned so the caller can replace them literally,
    which avoids mapping gitleaks' line/column spans back through a chunker
    that may have re-wrapped the text. They are held in memory for the length
    of one call and MUST NOT be logged, persisted, or put in an exception.
    """
    return scan_many([text])[0]


def scan_many(texts: list[str]) -> list[list[tuple[str, str, int]]]:
    """`find_secrets` for several texts at once, one result list per input.

    With `redactd` this is ONE round trip of a few hundred microseconds. On the
    CLI path it is one subprocess per window per text, which is why the daemon
    exists. Same contract either way: an empty list means "scanned, clean", and
    anything that is not a verdict raises `ScanUnavailable`.
    """
    if not texts:
        return []
    if redactd.disabled():
        return [_find_secrets_via_cli(t) for t in texts]
    supervisor = _shared_supervisor()
    # Windowing still applies: the daemon has the same 8 MiB ceiling, and a
    # document over it must be scanned in parts rather than truncated.
    windowed: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        start = len(windowed)
        if not text or not text.strip():
            spans.append((start, start))
            continue
        for window in _windows(text.encode("utf-8", errors="replace")):
            windowed.append(window.decode("utf-8", errors="replace"))
        spans.append((start, len(windowed)))
    if not windowed:
        return [[] for _ in texts]

    results: list[list[tuple[str, str, int]]] = []
    # Chunked so one very large batch cannot exceed the server's own ceilings.
    scanned: list[list[tuple[str, str, int]]] = []
    for i in range(0, len(windowed), redactd.MAX_TEXTS):
        scanned.extend(supervisor.scan(windowed[i : i + redactd.MAX_TEXTS]))
    for start, end in spans:
        seen: set[tuple[str, str]] = set()
        merged: list[tuple[str, str, int]] = []
        for per_window in scanned[start:end]:
            for rule, secret, line in per_window:
                # Windows overlap, so the same credential can be reported twice.
                if (rule, secret) in seen:
                    continue
                seen.add((rule, secret))
                merged.append((rule, secret, line))
        results.append(merged)
    return results


def _find_secrets_via_cli(text: str) -> list[tuple[str, str, int]]:
    """The pre-daemon path. Selected by PROBE_REDACTD_DISABLED, and used by the
    differential test as the oracle the daemon is checked against."""
    if not text or not text.strip():
        return []
    binary = binary_path()
    if binary is None:
        if get_settings().secret_redaction_fail_closed:
            raise ScanUnavailable("credential scanner binary not found", env_var=_BIN_ENV)
        log.warning("secret_redaction.binary_missing", env_var=_BIN_ENV)
        return []
    encoded = text.encode("utf-8", errors="replace")
    out: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for window in _windows(encoded):
        for rule, secret, line in _scan_once(binary, window):
            # Windows overlap, so the same credential can be reported twice.
            # Line numbers are window-relative and only ever reach a
            # customer-visible record, so the first one wins.
            if (rule, secret) in seen:
                continue
            seen.add((rule, secret))
            out.append((rule, secret, line))
    return out


def _windows(payload: bytes) -> list[bytes]:
    """`payload` split into overlapping scan windows. One window when it fits."""
    if len(payload) <= MAX_SCAN_BYTES:
        return [payload]
    stride = MAX_SCAN_BYTES - SCAN_WINDOW_OVERLAP_BYTES
    windows = [payload[i : i + MAX_SCAN_BYTES] for i in range(0, len(payload), stride)]
    log.info(
        "secret_redaction.windowed",
        total_bytes=len(payload),
        windows=len(windows),
    )
    return windows


def _scan_once(binary: str, payload: bytes) -> list[tuple[str, str, int]]:
    """Scan ONE window. Raises `ScanUnavailable` on any non-verdict."""
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

    proc = None
    for attempt in range(1, SCAN_ATTEMPTS + 1):
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                timeout=SCAN_TIMEOUT_SECONDS,
                check=False,
            )
            break
        except subprocess.TimeoutExpired:
            log.warning(
                "secret_redaction.timeout",
                seconds=SCAN_TIMEOUT_SECONDS,
                bytes=len(payload),
                attempt=attempt,
                of=SCAN_ATTEMPTS,
            )
        except OSError as exc:
            raise ScanUnavailable("credential scanner would not start", error=str(exc)) from exc
    if proc is None:
        raise ScanUnavailable(
            "credential scanner timed out",
            seconds=SCAN_TIMEOUT_SECONDS,
            attempts=SCAN_ATTEMPTS,
            bytes=len(payload),
        )

    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        # A rules file the installed gitleaks rejects, a flag removed by a CLI
        # upgrade, an OOM kill. Until 2026-09-17 this returned [] and the text
        # was stored as if it had been cleared.
        raise ScanUnavailable(
            "credential scanner exited non-zero",
            returncode=proc.returncode,
            # stderr is gitleaks' own diagnostics; its stdout can carry secrets
            # on a partial write, so only stderr is ever echoed, and truncated.
            stderr=proc.stderr.decode("utf-8", errors="replace")[:500],
        )
    if not raw:
        return []
    try:
        findings = json.loads(raw)
    except ValueError as exc:
        # Never echo stdout: on a partial write it can contain the secret.
        raise ScanUnavailable("credential scanner report was unparseable", bytes=len(raw)) from exc

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

    Propagates `ScanUnavailable` from `find_secrets`: a caller that cannot
    get a verdict must not persist these pieces. Callers do not catch it —
    the worker's transient-retry path is what handles it.

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

    The scan is a blocking `subprocess.run` with a 20s ceiling (twice, on a
    timeout), and both real callers are coroutines on the ingestion worker's
    loop. Called directly,
    one large document stalls EVERYTHING that loop is doing for the length of
    the scan — including the health endpoint, whose DB ping shares the loop, so
    a slow scan reads as an unhealthy pod and gets it restarted mid-ingestion.
    """
    return await asyncio.to_thread(redact_documents, texts)

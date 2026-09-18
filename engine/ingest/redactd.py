"""Client and supervisor for `redactd`, the boot-once credential scanner.

WHY
---
`secret_redaction` scans by running the gitleaks CLI. One spawn costs ~1.05
CPU-seconds, of which 0.67 is the Go binary starting before it reads a byte
(measured in the pod, 2026-09-16). At 15,617 documents a day that is 4.6
CPU-hours of pure process startup, and it is what held the ingestion worker at
81.5% CFS throttling with a 33-minute median queue wait.

`tools/redactd` is the same gitleaks engine and the same rules file, compiled
once and served over a Unix socket. Measured on this machine: 0.9ms for the
first scan, 0.2ms thereafter, against ~1050ms per CLI spawn.

WHY A SUPERVISED CHILD AND NOT A SIDECAR
----------------------------------------
A sidecar is a chart change, a second container to schedule, and a lifecycle
that outlives the process it serves. This binary exists for exactly one parent
and should die with it, so the parent owns it: started at worker boot, restarted
with backoff if it exits, and killed by the kernel via PDEATHSIG if the parent
goes away without cleaning up.

WHAT FAILURE MEANS
------------------
Every failure here raises `ScanUnavailable`, exactly as the CLI path does. The
daemon being down is "we could not look", never "we looked and it was clean" --
the whole property `secret_redaction` exists to preserve. There is no fallback
to the CLI in the failure path on purpose: a silent downgrade to a 1-CPU-second
spawn per document is how the original problem is reintroduced under load, and
it would hide the daemon being broken for as long as the queue could absorb it.
`PROBE_REDACTD_DISABLED=1` selects the CLI deliberately and loudly instead.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import structlog

from engine.shared.exceptions import ScanUnavailable

log = structlog.get_logger(__name__)

#: Override for tests and for images that install the binary elsewhere.
_BIN_ENV = "PROBE_REDACTD_BIN"
#: Set to 1 to use the gitleaks CLI instead. A deliberate, visible choice.
_DISABLED_ENV = "PROBE_REDACTD_DISABLED"

#: Mirrors the server's own ceilings (tools/redactd/main.go).
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_TEXTS = 512

#: A scan is sub-millisecond. This is not a scan budget, it is "the daemon has
#: stopped answering" -- generous enough that a loaded box is not mistaken for
#: a wedged one.
REQUEST_TIMEOUT_SECONDS = 30.0
#: Readiness handshake at boot / after a restart.
PING_TIMEOUT_SECONDS = 10.0
#: Restart backoff, capped. A daemon that cannot start (bad rules file) must
#: not become a fork bomb inside the worker.
RESTART_BACKOFF_START_SECONDS = 0.5
RESTART_BACKOFF_CAP_SECONDS = 30.0


def binary_path() -> str | None:
    override = os.environ.get(_BIN_ENV)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("redactd")


def disabled() -> bool:
    return os.environ.get(_DISABLED_ENV, "").strip() in {"1", "true", "TRUE", "yes"}


def _rules_path() -> Path:
    from engine.ingest.secret_redaction import RULES_PATH

    return RULES_PATH


class RedactdSupervisor:
    """Owns one `redactd` child for the life of its context.

    Not a singleton: the worker owns one, and a sweep job that runs outside the
    worker owns its own. A module-level daemon shared by whatever imported it
    is a lifetime nobody controls.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._dir: str | None = None
        self._socket: str | None = None
        self._lock = threading.Lock()
        self._backoff = RESTART_BACKOFF_START_SECONDS
        self._stopped = False

    # ---- lifecycle ------------------------------------------------------
    def __enter__(self) -> RedactdSupervisor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        with self._lock:
            self._stopped = False
            self._spawn_locked()

    def stop(self) -> None:
        with self._lock:
            # Set FIRST: a restart racing a shutdown would otherwise leave an
            # orphan whose parent is already gone.
            self._stopped = True
            self._kill_locked()

    def _kill_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
            if proc.poll() is None:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
            self._socket = None

    def _spawn_locked(self) -> None:
        binary = binary_path()
        if binary is None:
            raise ScanUnavailable("redactd binary not found", env_var=_BIN_ENV)
        # The runtime user is not root and /run is not writable, so the socket
        # lives in a private directory this process creates and owns.
        self._dir = tempfile.mkdtemp(prefix="redactd-")
        os.chmod(self._dir, 0o700)
        self._socket = os.path.join(self._dir, "redactd.sock")
        try:
            self._proc = subprocess.Popen(
                [binary, "--socket", self._socket, "--config", str(_rules_path())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=_die_with_parent,
            )
        except OSError as exc:
            # `binary_path()` only proves the path EXISTED when it was read.
            # A binary deleted, unreadable, or built for another architecture
            # raises here, and a bare OSError out of a credential gate is an
            # unhandled exception somewhere upstream rather than a retry.
            raise ScanUnavailable("redactd would not start", error=str(exc)) from exc
        self._await_ready_locked()
        self._backoff = RESTART_BACKOFF_START_SECONDS

    def _await_ready_locked(self) -> None:
        """Readiness is a PING THROUGH THE SOCKET, not the socket file
        existing. A stale socket from a crashed predecessor accepts nothing,
        and a booting one has not compiled its 223 rules yet."""
        deadline = time.monotonic() + PING_TIMEOUT_SECONDS
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise ScanUnavailable(
                    "redactd exited during startup", returncode=self._proc.returncode
                )
            try:
                if self._request_locked({"op": "ping"}).get("ok") is True:
                    return
            except Exception as exc:
                last = exc
            time.sleep(0.05)
        raise ScanUnavailable("redactd did not become ready", error=str(last) if last else "timeout")

    def _restart_locked(self) -> None:
        if self._stopped:
            raise ScanUnavailable("redactd supervisor is shut down")
        backoff, self._backoff = self._backoff, min(self._backoff * 2, RESTART_BACKOFF_CAP_SECONDS)
        log.warning("redactd.restarting", backoff_seconds=backoff)
        self._kill_locked()
        time.sleep(backoff)
        self._spawn_locked()

    # ---- transport ------------------------------------------------------
    def _request_locked(self, payload: dict) -> dict:
        assert self._socket is not None
        body = json.dumps(payload).encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(REQUEST_TIMEOUT_SECONDS)
            sock.connect(self._socket)
            sock.sendall(struct.pack(">I", len(body)) + body)
            header = _recv_exactly(sock, 4)
            (length,) = struct.unpack(">I", header)
            raw = _recv_exactly(sock, length)
        out = json.loads(raw)
        if not isinstance(out, dict):
            raise ScanUnavailable("redactd response was not an object")
        return out

    def scan(self, texts: list[str]) -> list[list[tuple[str, str, int]]]:
        """`(rule, secret, line)` per input text, in input order.

        Raises `ScanUnavailable` for every failure: down, wedged, malformed, or
        a response whose length does not match the request. A short response is
        NOT a partial success -- silently treating missing entries as "clean"
        is exactly the class of bug this whole change exists to remove.
        """
        if not texts:
            return []
        if len(texts) > MAX_TEXTS:
            raise ScanUnavailable("too many texts for one scan", texts=len(texts), ceiling=MAX_TEXTS)
        total = sum(len(t.encode("utf-8", errors="replace")) for t in texts)
        if total > MAX_REQUEST_BYTES:
            raise ScanUnavailable("scan request too large", bytes=total, ceiling=MAX_REQUEST_BYTES)

        with self._lock:
            try:
                out = self._request_locked({"op": "scan", "texts": texts})
            except ScanUnavailable:
                raise
            except Exception:
                # One restart, then one retry. A wedged or crashed daemon
                # recovers without the caller knowing; a broken one fails.
                log.warning("redactd.request_failed")
                self._restart_locked()
                try:
                    out = self._request_locked({"op": "scan", "texts": texts})
                except Exception:
                    raise ScanUnavailable("redactd unreachable") from None

        if out.get("ok") is not True:
            raise ScanUnavailable("redactd refused the scan")
        findings = out.get("findings")
        if not isinstance(findings, list) or len(findings) != len(texts):
            raise ScanUnavailable(
                "redactd returned the wrong number of results",
                expected=len(texts),
                got=len(findings) if isinstance(findings, list) else "not-a-list",
            )
        parsed: list[list[tuple[str, str, int]]] = []
        for per_text in findings:
            if not isinstance(per_text, list):
                raise ScanUnavailable("redactd returned a malformed findings entry")
            rows = []
            for finding in per_text:
                if not isinstance(finding, dict):
                    raise ScanUnavailable("redactd returned a malformed finding")
                rule, secret, line = finding.get("rule"), finding.get("secret"), finding.get("line", 0)
                if not (isinstance(rule, str) and rule and isinstance(secret, str) and secret
                        and isinstance(line, int) and not isinstance(line, bool) and line >= 0):
                    raise ScanUnavailable("redactd returned a malformed finding")
                rows.append((rule, secret, line))
            parsed.append(rows)
        return parsed


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ScanUnavailable("redactd closed the connection mid-response")
        buf.extend(chunk)
    return bytes(buf)


def _die_with_parent() -> None:
    """PR_SET_PDEATHSIG: if the worker dies without running its shutdown path,
    the kernel SIGTERMs this child rather than leaving it holding a socket in a
    directory nobody will clean up."""
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGTERM

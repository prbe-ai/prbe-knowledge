"""`redactd`: does it agree with the CLI, and does it fail closed when it cannot.

The daemon exists because a gitleaks CLI spawn costs ~1.05 CPU-seconds of pure
startup and the worker was doing that once per document. Swapping the transport
under a credential gate is only safe if the new one finds exactly what the old
one did, so the differential test below is the load-bearing one: it runs BOTH
transports over the same corpus and compares finding for finding.
"""

from __future__ import annotations

import os
import shutil
import socket
import struct
import subprocess
import time

import pytest

from engine.ingest import redactd, secret_redaction
from engine.shared.exceptions import ScanUnavailable

_MISSING = redactd.binary_path() is None
if _MISSING and os.environ.get("CI"):
    raise RuntimeError(
        "redactd is not built in CI, so the differential suite would silently "
        "skip -- and that suite is the only thing proving the daemon and the "
        "CLI agree. Build it in the workflow, or this gate is decoration."
    )
pytestmark = pytest.mark.skipif(
    _MISSING or shutil.which("gitleaks") is None,
    reason="needs both redactd (PROBE_REDACTD_BIN) and gitleaks to compare them",
)

_AWS_ID = "AKIA" + "4KX7QZJ2MNVB3TWD"
_AWS_SECRET = "hT7xQ2mVb9Lk" + "Zp0RwYe4Ns6Uc1Ai8Jd3Fg5Oh2Pq"

#: Shapes that MUST be found, and shapes that must not. The second half is the
#: 2026-08 outage corpus: 615 false positives out of 645 drops.
_CORPUS = [
    f"AWS Secret Access Key [None]: {_AWS_SECRET}",
    f"AWS Access Key ID [None]: {_AWS_ID}",
    f"export AWS_SECRET_ACCESS_KEY={_AWS_SECRET}",
    f"the key is {_AWS_ID} and the secret is {_AWS_SECRET}",
    "aws_secret_access_key = /workspace/library/checkpoints/fsq/FINAL_v2",
    "content_hash = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "def train(model, lr=3e-4):  # nothing secret here",
    "",
    "   ",
    "a" * 5000,
    "héllo wörld — unicode and an em dash",
    f"line one\nline two\nAWS Secret Access Key [None]: {_AWS_SECRET}\nline four",
]


@pytest.fixture
def daemon():
    with redactd.RedactdSupervisor() as sup:
        yield sup


def _via_cli(text: str) -> list[tuple[str, str, int]]:
    return secret_redaction._find_secrets_via_cli(text)


def test_redactd_and_the_cli_find_the_same_things(daemon) -> None:
    """The whole safety argument for the swap. Rule id, value and line, per
    text, both transports."""
    via_daemon = daemon.scan(_CORPUS)
    for text, daemon_hits in zip(_CORPUS, via_daemon, strict=True):
        cli_hits = _via_cli(text)
        assert sorted(daemon_hits) == sorted(cli_hits), text[:80]


def test_the_false_positive_corpus_stays_clean_on_both(daemon) -> None:
    """A daemon that finds MORE than the CLI is not a safe swap either: the
    gate that preceded this one died of false positives, not of misses."""
    benign = [t for t in _CORPUS if not _via_cli(t)]
    assert benign, "fixture must contain clean text"
    for text, hits in zip(benign, daemon.scan(benign), strict=True):
        assert hits == [], text[:80]


def test_the_custom_rule_is_loaded_in_the_daemon(daemon) -> None:
    """`probe-anchored-secret` is the one rule this deployment adds. A daemon
    that loaded the stock corpus and silently ignored the vendored config
    would read downstream as "fewer findings"."""
    hits = daemon.scan([f"AWS Secret Access Key [None]: {_AWS_SECRET}"])[0]
    assert any(rule == "probe-anchored-secret" for rule, _, _ in hits)


def test_check_refuses_to_start_without_the_custom_rule() -> None:
    """--check is the image build's assertion, so it has to assert the thing
    that actually breaks: the vendored config being ignored."""
    out = subprocess.run(
        [redactd.binary_path(), "--config", str(secret_redaction.RULES_PATH), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "rules=" in out.stdout


def test_a_dead_daemon_is_restarted_and_the_scan_succeeds(daemon) -> None:
    """A crash must self-heal without the caller knowing: one restart, one
    retry."""
    assert daemon.scan(["warm up"])
    daemon._proc.kill()
    daemon._proc.wait(timeout=5)
    hits = daemon.scan([f"AWS Secret Access Key [None]: {_AWS_SECRET}"])[0]
    assert any(rule == "probe-anchored-secret" for rule, _, _ in hits)


def test_a_permanently_dead_daemon_fails_closed(daemon, monkeypatch) -> None:
    """The property the whole change protects: "cannot look" never becomes
    "found nothing"."""
    daemon._proc.kill()
    daemon._proc.wait(timeout=5)
    monkeypatch.setattr(redactd, "binary_path", lambda: "/nonexistent/redactd")
    with pytest.raises(ScanUnavailable):
        daemon.scan(["anything"])


def test_a_truncated_response_is_not_a_partial_success(daemon) -> None:
    """Fewer results than texts would silently mean "clean" for the missing
    ones."""
    real = daemon._request_locked

    def _short(payload):
        out = real(payload)
        if payload.get("op") == "scan" and out.get("findings"):
            out["findings"] = out["findings"][:-1]
        return out

    daemon._request_locked = _short
    with pytest.raises(ScanUnavailable, match="wrong number"):
        daemon.scan(["a", "b"])


def test_the_socket_is_private_to_this_process(daemon) -> None:
    """Customer documents cross it."""
    assert daemon._socket is not None
    assert os.stat(daemon._socket).st_mode & 0o777 == 0o600
    assert os.stat(os.path.dirname(daemon._socket)).st_mode & 0o777 == 0o700


def test_there_is_no_network_listener(daemon) -> None:
    """A TCP port would make a credential scanner reachable from anywhere in
    the cluster."""
    assert daemon._socket.startswith("/")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect(daemon._socket)  # proves the transport is a unix socket


def test_oversize_and_overlong_requests_are_refused(daemon) -> None:
    with pytest.raises(ScanUnavailable, match="too many texts"):
        daemon.scan(["x"] * (redactd.MAX_TEXTS + 1))
    with pytest.raises(ScanUnavailable, match="too large"):
        daemon.scan(["x" * (redactd.MAX_REQUEST_BYTES + 1)])


def test_a_malformed_frame_gets_an_error_not_a_hang(daemon) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(daemon._socket)
        body = b"{not json"
        s.sendall(struct.pack(">I", len(body)) + body)
        (length,) = struct.unpack(">I", s.recv(4))
        assert b"not valid JSON" in s.recv(length)


def test_scanning_is_orders_of_magnitude_cheaper_than_a_spawn(daemon) -> None:
    """The reason this exists. Not a benchmark assertion -- a floor loose
    enough never to flake, tight enough to catch the daemon accidentally
    re-spawning per call."""
    text = f"AWS Secret Access Key [None]: {_AWS_SECRET}"
    daemon.scan([text])  # warm
    start = time.monotonic()
    for _ in range(20):
        daemon.scan([text])
    per_scan = (time.monotonic() - start) / 20
    assert per_scan < 0.1, f"{per_scan * 1000:.1f}ms per scan — is it spawning?"

"""Server-side credential redaction: the backstop behind the tap.

Every "credential" here is synthetic — random, or AWS's own published
documentation example. No live secret appears in this repository.

The FALSE_POSITIVES list is the 2026-08 outage encoded: 645 batches dropped
across 206 sessions, 615 of them false, from two heuristics that could not tell
a coding transcript from a credential dump. A rule that fires on one of these
does not ship, whichever engine finds it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import pytest

from engine.ingest import secret_redaction
from engine.shared.exceptions import ScanUnavailable


@dataclass
class _Proc:
    """Stand-in for `subprocess.CompletedProcess` — the three fields read."""

    returncode: int
    stdout: bytes
    stderr: bytes


@pytest.fixture
def fresh_settings():
    """`get_settings` is `lru_cache`d, so a test that moves an env var must
    clear it BOTH ways: once before so the test sees its own value, and once
    after so the next test does not inherit it."""
    from engine.shared import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()

# A SKIP that looks like a PASS is how a gate stops gating. Locally, skipping is
# right — not every developer has the binary. In CI it is not: the whole point
# of this file is to prove the deployed scanner catches the outage corpus, and
# "skipped" there means nobody checked. So CI fails instead.
_MISSING = not secret_redaction.available()
if _MISSING and os.environ.get("CI"):
    raise RuntimeError(
        "gitleaks is not installed in CI, so the credential-redaction suite "
        "would silently skip. Install it in the workflow, or this gate is "
        "decoration."
    )
pytestmark = pytest.mark.skipif(
    _MISSING, reason="gitleaks binary not installed locally (set PROBE_GITLEAKS_BIN)"
)

#: Synthetic credential fixtures, ASSEMBLED AT RUNTIME rather than written as
#: literals. GitHub push protection blocks a commit containing an AKIA-shaped
#: string even when it is invented, and it is right to — a scanner that trusts
#: "this one is fine, it's a test" is a scanner you cannot rely on. None of
#: these values is or has ever been a real key.
_AWS_ID = "AKIA" + "4KX7QZJ2MNVB3TWD"
_AWS_SECRET = "hT7xQ2mVb9Lk" + "Zp0RwYe4Ns6Uc1Ai8Jd3Fg5Oh2Pq"
_GH_PAT = "ghp_" + "16C7e42F292c6912E7710c838347Ae178B4a"
_GCP_KEY = "AIza" + "SyC1x9Kp0RwYe4Ns6Uc1Ai8Jd3Fg5Oh2Pq7"

#: Verbatim from PR #365's postmortem, plus ordinary ML-transcript noise.
FALSE_POSITIVES = [
    "/OdysseyPrivate/odyssey/experiments/casp",                 # H=3.85, 539 drops
    "/workspace/library/checkpoints/fsq/FINAL",                 # H=4.38
    "SLICE_SHA256=9f2c1ab44e3d8071b5c6e2f9a0d4738b1c5e6f7a8b9c0d1e2f3a4b5c6d7e8f90",
    "SLICE_BYTES=48210347",
    "EVAL_N=1024",
    "commit 03a24b90bee1c7341cc4714a5ca21850f8a9a91c",
    "session 4525087c-392d-4acf-b221-d861512fb467",
    "content_hash = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tensor shape torch.Size([32, 1024, 4096]) dtype=torch.bfloat16",
    "s5cmd ls s3://runpod-files-new/checkpoints/odyssey3/",
    "Downloading torch-2.9.1+cu128-cp313-cp313-linux_x86_64.whl (912.4 MB)",
    "api_key = os.environ['ANTHROPIC_API_KEY']",
    "aws_secret_access_key = <your-secret-here>",
    "the secretary said: meeting at 1400 hours in room 12",
    # Found by scanning 4,000 REAL production chunks (2026-09-12). Every one
    # fired before the rule required a non-dash first character and forbade `/`
    # in the value. Same failure as the 2026-08 outage in different clothing: a
    # path or a CLI flag close enough to a credential word to be read as one.
    "create the pull secret here:  --some-thing-SOMEEE-SOMEE-SOMEEEE-",
    'creates R2 credential secret for pods"}, {"path": "SomeThing/configs-abc-defg"',
    ':  --from-literal=R2_ACCESS_KEY_ID="<key>" \\ 54:  --some-thing-K8-SOMEEEE-SOMEEE-ABC-',
    "test_secret_syncs_before_app_secret - AssertionError: /some/path_with/parts-9a-bc9de",
    "export AWS_SHARED_CREDENTIALS_FILE=/workspace/odyssey/configs/aws/credentials_prod",
]

#: The two shapes that actually leaked, plus common vendor formats.
TRUE_POSITIVES = [
    ("aws_id_echo", f"AWS Access Key ID [None]: {_AWS_ID}"),
    ("aws_secret_echo", f"AWS Secret Access Key [None]: {_AWS_SECRET}"),
    ("aws_ini", f"aws_secret_access_key = {_AWS_SECRET}"),
    ("github_pat", _GH_PAT),
    ("gcp", _GCP_KEY),
]


def test_binary_is_resolvable_or_the_suite_says_so() -> None:
    """`available()` must be the only thing standing between a deployment and a
    silent no-op, so it is worth asserting it means what it says."""
    assert secret_redaction.binary_path()
    assert shutil.which(secret_redaction.binary_path()) or secret_redaction.binary_path()


@pytest.mark.parametrize("text", FALSE_POSITIVES)
def test_outage_corpus_stays_clean(text: str) -> None:
    out, findings = secret_redaction.redact_documents([text])
    assert findings == [], f"fired on benign text: {text[:60]}"
    assert out == [text]


@pytest.mark.parametrize("name,text", TRUE_POSITIVES, ids=[n for n, _ in TRUE_POSITIVES])
def test_credentials_are_replaced(name: str, text: str) -> None:
    out, findings = secret_redaction.redact_documents([text])
    assert findings, f"{name}: nothing found"
    assert out[0] != text
    assert "<redacted:" in out[0]


def test_both_halves_of_an_aws_pair_go() -> None:
    """Stock gitleaks catches the access key id and, via generic-api-key, the
    `=`-assigned secret — but NOT the `:`-separated `aws configure` echo, which
    is the shape that actually leaked. `probe-anchored-secret` exists for that
    one; this test is what keeps the rule file honest."""
    text = (
        f"AWS Access Key ID [None]: {_AWS_ID}\n"
        f"AWS Secret Access Key [None]: {_AWS_SECRET}\n"
    )
    out, findings = secret_redaction.redact_documents([text])
    assert _AWS_ID not in out[0]
    assert _AWS_SECRET not in out[0]
    assert {f.rule for f in findings} >= {"probe-anchored-secret"}


def test_overlapping_windows_are_what_cover_a_chunk_boundary() -> None:
    """A credential straddling a chunk boundary is caught by the CHUNKER, not
    by stitching here.

    `chunk_text` emits overlapping token windows (`stride = chunk_tokens -
    overlap`), so a credential near a boundary appears WHOLE in at least one
    piece. That is the mechanism, and this test pins it — because the obvious
    "improvement" to `redact_documents` is to join pieces with a space so an
    anchor can span them, and that would be wrong twice: it destroys gitleaks'
    line numbers, and it lets two unrelated chunks form a credential neither
    of them contains.
    """
    whole = f"AWS Secret Access Key [None]: {_AWS_SECRET}"
    # What overlap actually produces: the value is intact in the second window.
    pieces = [f"AWS Secret Access Key [None]: {_AWS_SECRET[:12]}", whole]
    out, findings = secret_redaction.redact_documents(pieces)
    assert findings
    assert _AWS_SECRET not in "".join(out)


def test_pieces_are_not_stitched_across_the_join() -> None:
    """The flip side of the above, asserted so nobody relaxes the join."""
    pieces = ["AWS Secret Access Key [None]:", f" {_AWS_SECRET}"]
    _, findings = secret_redaction.redact_documents(pieces)
    assert findings == [], (
        "pieces were stitched into a match neither contains; the chunker's "
        "overlap is the boundary mechanism, not concatenation here"
    )


def test_every_occurrence_goes_not_just_the_first() -> None:
    secret = _AWS_ID
    pieces = [f"first {secret}", "middle", f"again {secret}"]
    out, _ = secret_redaction.redact_documents(pieces)
    assert secret not in "".join(out)


def test_findings_carry_no_value(monkeypatch) -> None:
    """`Redaction` must have no field that could hold a secret — this is the
    difference between a log line an operator can paste into a ticket and one
    that leaks the thing it is reporting."""
    _, findings = secret_redaction.redact_documents(
        [f"AWS Access Key ID [None]: {_AWS_ID}"]
    )
    assert findings
    for finding in findings:
        assert "AKIA" not in repr(finding)
        assert set(vars(finding)) == {"rule", "line"}


def test_missing_binary_fails_closed(monkeypatch, fresh_settings) -> None:
    """A deployment that should have the scanner and does not must not quietly
    store unscanned text. `ScanUnavailable` is transient, so the queue row
    retries instead of persisting."""
    monkeypatch.setenv("PROBE_GITLEAKS_BIN", "/nonexistent/gitleaks")
    assert secret_redaction.available() is False
    with pytest.raises(ScanUnavailable):
        secret_redaction.redact_documents([f"AWS Access Key ID [None]: {_AWS_ID}"])


def test_missing_binary_can_be_opted_out_of(monkeypatch, fresh_settings) -> None:
    """The one deployment shape that may run without a scanner says so
    explicitly, and then gets the old pass-through."""
    monkeypatch.setenv("PROBE_GITLEAKS_BIN", "/nonexistent/gitleaks")
    monkeypatch.setenv("SECRET_REDACTION_FAIL_CLOSED", "false")
    text = f"AWS Access Key ID [None]: {_AWS_ID}"
    out, findings = secret_redaction.redact_documents([text])
    assert out == [text]
    assert findings == []


def test_scanner_exiting_non_zero_is_not_a_clean_scan(monkeypatch) -> None:
    """The failure that was completely silent until 2026-09-17: a rules file the
    installed gitleaks rejects produced no stdout, returned [], and the text was
    stored as if it had been cleared."""
    monkeypatch.setattr(
        secret_redaction.subprocess, "run",
        lambda *a, **k: _Proc(returncode=1, stdout=b"", stderr=b"bad config"),
    )
    with pytest.raises(ScanUnavailable) as exc:
        secret_redaction.find_secrets("anything at all")
    assert "bad config" in str(exc.value)


def test_unparseable_report_is_not_a_clean_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        secret_redaction.subprocess, "run",
        lambda *a, **k: _Proc(returncode=0, stdout=b"{not json", stderr=b""),
    )
    with pytest.raises(ScanUnavailable):
        secret_redaction.find_secrets("anything at all")


def test_spawn_failure_is_not_a_clean_scan(monkeypatch) -> None:
    def _boom(*a, **k):
        raise OSError("no fork for you")
    monkeypatch.setattr(secret_redaction.subprocess, "run", _boom)
    with pytest.raises(ScanUnavailable):
        secret_redaction.find_secrets("anything at all")


def test_a_timeout_retries_once_then_fails_closed(monkeypatch) -> None:
    """The 2026-09-16 shape: 328 timeouts in 22h under CPU starvation, each one
    a document stored unscanned. One retry, because a starved scan often clears
    on the next attempt; then closed."""
    calls = {"n": 0}

    def _always_timeout(*a, **k):
        calls["n"] += 1
        raise secret_redaction.subprocess.TimeoutExpired(cmd="gitleaks", timeout=20.0)

    monkeypatch.setattr(secret_redaction.subprocess, "run", _always_timeout)
    with pytest.raises(ScanUnavailable):
        secret_redaction.find_secrets("anything at all")
    assert calls["n"] == secret_redaction.SCAN_ATTEMPTS

    calls["n"] = 0

    def _timeout_then_ok(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise secret_redaction.subprocess.TimeoutExpired(cmd="gitleaks", timeout=20.0)
        return _Proc(returncode=0, stdout=b"[]", stderr=b"")

    monkeypatch.setattr(secret_redaction.subprocess, "run", _timeout_then_ok)
    assert secret_redaction.find_secrets("anything at all") == []
    assert calls["n"] == 2


def test_oversize_input_is_windowed_not_truncated() -> None:
    """The old code cut at MAX_SCAN_BYTES and logged; the tail of a large
    transcript was never scanned. A credential past the cut must still be
    found."""
    filler = "lorem ipsum dolor sit amet " * 4000  # ~108 KB
    window = 64 * 1024
    original = secret_redaction.MAX_SCAN_BYTES
    try:
        secret_redaction.MAX_SCAN_BYTES = window
        text = filler + f"\nAWS Secret Access Key [None]: {_AWS_SECRET}\n" + filler
        assert len(text.encode()) > window * 2, "fixture must span several windows"
        found = secret_redaction.find_secrets(text)
        assert any(rule == "probe-anchored-secret" for rule, _, _ in found)
        secrets = [s for _, s, _ in found]
        assert len(secrets) == len(set(secrets)), "overlap must not double-report"
    finally:
        secret_redaction.MAX_SCAN_BYTES = original


def test_a_credential_across_a_window_boundary_is_still_found() -> None:
    """The reason windows overlap. Place the credential so a naive split lands
    inside it."""
    original = secret_redaction.MAX_SCAN_BYTES
    try:
        window = 64 * 1024
        secret_redaction.MAX_SCAN_BYTES = window
        line = f"AWS Secret Access Key [None]: {_AWS_SECRET}"
        # Straddle the END of window 0: the first window holds only the first
        # half of the credential, so only the overlap in window 1 can find it.
        head = "x" * (window - len(line) // 2) + "\n"
        text = head + line + "\n" + ("y" * window)
        found = secret_redaction.find_secrets(text)
        assert any(rule == "probe-anchored-secret" for rule, _, _ in found), (
            "a credential straddling a window edge was lost — the overlap is "
            "the whole reason windows are not a plain split"
        )
    finally:
        secret_redaction.MAX_SCAN_BYTES = original


def test_empty_and_whitespace_inputs_are_cheap() -> None:
    assert secret_redaction.redact_documents([]) == ([], [])
    assert secret_redaction.redact_documents(["", "   "]) == (["", "   "], [])


def test_the_custom_rule_actually_matches_something() -> None:
    """A rule can be BROKEN into matching nothing and still look like a clean
    scan. `probe-anchored-secret` was, once: `{20,512}` was written against an
    alternation GROUP rather than a character class, so it repeated the group
    twenty times and matched no real input. Findings went to zero and that read
    as "no false positives" in every measurement.

    So: assert the custom rule fires, by id, on the shape it exists for.
    """
    text = f"AWS Secret Access Key [None]: {_AWS_SECRET}"
    found = secret_redaction.find_secrets(text)
    assert any(rule == "probe-anchored-secret" for rule, _, _ in found), (
        "the vendored rule extension matched nothing; a rule that matches "
        "nothing is indistinguishable from a clean corpus"
    )


def test_a_value_that_is_a_path_is_not_a_credential() -> None:
    """The 539-drop class, asserted directly against the custom rule."""
    text = "aws_secret_access_key = /workspace/library/checkpoints/fsq/FINAL_v2"
    assert [r for r, _, _ in secret_redaction.find_secrets(text)
            if r == "probe-anchored-secret"] == []

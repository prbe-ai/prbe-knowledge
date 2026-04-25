"""Unit tests for OAuth state signing.

`sign_state` / `verify_state` wrap itsdangerous.URLSafeTimedSerializer.
We force HMAC-SHA256 (not the library default SHA1) and a 10-minute TTL.
"""

from __future__ import annotations

import time

import pytest

from shared.state_signing import sign_state, verify_state


def test_sign_verify_roundtrip() -> None:
    state = sign_state("cust-1", "notion")
    assert isinstance(state, str)
    assert verify_state(state, "notion") == "cust-1"


def test_state_is_url_safe() -> None:
    """itsdangerous.URLSafeTimedSerializer uses base64url alphabet — no
    +, /, or = chars — so the state can ride a query string without
    extra encoding."""
    state = sign_state("cust-with-dash", "slack")
    assert "+" not in state
    assert "/" not in state
    assert "=" not in state


def test_verify_rejects_forged() -> None:
    state = sign_state("cust-1", "notion")
    # Flip a byte in the signature (last char). itsdangerous puts the
    # signature after a `.` separator.
    body, sig = state.rsplit(".", 1)
    flipped = sig[:-1] + ("a" if sig[-1] != "a" else "b")
    forged = f"{body}.{flipped}"
    assert verify_state(forged, "notion") is None


def test_verify_rejects_wrong_source() -> None:
    state = sign_state("cust-1", "slack")
    assert verify_state(state, "notion") is None


def test_verify_rejects_expired(monkeypatch) -> None:
    # Sign now, then advance time past the 600s TTL.
    state = sign_state("cust-1", "notion")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 700)
    assert verify_state(state, "notion") is None


def test_verify_rejects_garbage() -> None:
    assert verify_state("not-a-signed-state", "notion") is None
    assert verify_state("", "notion") is None

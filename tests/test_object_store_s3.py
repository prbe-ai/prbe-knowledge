"""AWS S3 support for the engine object store: the boto3 client build (region /
optional endpoint / static-or-ambient credentials + session token) and the
OBJECT_STORE_* env aliases. Pure; no boto3 network, no DB."""

from __future__ import annotations

from types import SimpleNamespace

import boto3
from pydantic import SecretStr

from engine.shared.config import Settings
from engine.shared.storage import _make_client


def _st(**over) -> SimpleNamespace:
    base = dict(
        r2_endpoint_url="",
        r2_region="us-east-1",
        r2_access_key_id="",
        r2_secret_access_key=None,
        r2_session_token=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _capture(monkeypatch) -> dict:
    captured: dict = {}

    def _client(service, **kwargs):
        captured["service"] = service
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(boto3, "client", _client)
    return captured


# --- client build ------------------------------------------------------------


def test_client_ambient_creds_optional_endpoint_and_region(monkeypatch) -> None:
    # No static key/secret -> pass NOTHING so boto3 uses IRSA / instance-profile creds.
    cap = _capture(monkeypatch)
    _make_client(_st())
    assert cap["service"] == "s3"
    assert cap["region_name"] == "us-east-1"
    assert cap["endpoint_url"] is None  # empty endpoint -> AWS default resolver
    assert "aws_access_key_id" not in cap
    assert "aws_secret_access_key" not in cap
    assert "aws_session_token" not in cap


def test_client_static_creds_and_session_token(monkeypatch) -> None:
    cap = _capture(monkeypatch)
    _make_client(
        _st(
            r2_endpoint_url="https://s3.us-east-1.amazonaws.com",
            r2_access_key_id="AKIA",
            r2_secret_access_key=SecretStr("sec"),
            r2_session_token=SecretStr("tok"),
        )
    )
    assert cap["endpoint_url"] == "https://s3.us-east-1.amazonaws.com"
    assert cap["aws_access_key_id"] == "AKIA"
    assert cap["aws_secret_access_key"] == "sec"
    assert cap["aws_session_token"] == "tok"


def test_client_static_creds_without_session_token(monkeypatch) -> None:
    cap = _capture(monkeypatch)
    _make_client(_st(r2_access_key_id="AKIA", r2_secret_access_key=SecretStr("sec")))
    assert cap["aws_access_key_id"] == "AKIA"
    assert "aws_session_token" not in cap  # None token is not passed


def test_r2_endpoint_still_works_as_before(monkeypatch) -> None:
    # Regression: an explicit R2 endpoint + static creds is unchanged.
    cap = _capture(monkeypatch)
    _make_client(
        _st(
            r2_endpoint_url="https://acct.r2.cloudflarestorage.com",
            r2_region="auto",
            r2_access_key_id="k",
            r2_secret_access_key=SecretStr("s"),
        )
    )
    assert cap["endpoint_url"] == "https://acct.r2.cloudflarestorage.com"
    assert cap["region_name"] == "auto"


# --- config: OBJECT_STORE_* aliases + optional creds -------------------------


def test_object_store_env_aliases_win(monkeypatch) -> None:
    monkeypatch.setenv("R2_ENDPOINT_URL", "http://legacy:9000")
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "https://s3.amazonaws.com")
    monkeypatch.setenv("OBJECT_STORE_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "tok")
    s = Settings()
    assert s.r2_endpoint_url == "https://s3.amazonaws.com"  # canonical wins over R2_*
    assert s.r2_region == "eu-west-1"
    assert s.r2_session_token is not None
    assert s.r2_session_token.get_secret_value() == "tok"


def test_secret_is_optional_for_ambient_creds(monkeypatch) -> None:
    # conftest sets R2_SECRET_ACCESS_KEY at import; removing it yields ambient mode.
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("OBJECT_STORE_SECRET", raising=False)
    s = Settings()
    assert s.r2_secret_access_key is None


def test_legacy_r2_secret_alias_still_populates(monkeypatch) -> None:
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "legacy-secret")
    s = Settings()
    assert s.r2_secret_access_key is not None
    assert s.r2_secret_access_key.get_secret_value() == "legacy-secret"

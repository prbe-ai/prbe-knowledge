"""The standalone GitHub App JWT (engine/shared/github_app.py) is minted with
PyJWT. It replaced python-jose, whose hard `ecdsa` dependency carries an
unfixed timing advisory (GHSA-wj6h-64fc-37mp). This pins what GitHub checks:
an RS256 token, signed by the App key, with `iss` = the App id and a lifetime
inside GitHub's 10-minute cap.
"""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from engine.shared.github_app import _build_app_jwt

APP_ID = "123456"


@pytest.fixture(scope="module")
def app_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.mark.parametrize(
    "fmt",
    [serialization.PrivateFormat.PKCS8, serialization.PrivateFormat.TraditionalOpenSSL],
    ids=["pkcs8", "pkcs1"],
)
def test_app_jwt_is_rs256_signed_by_the_app_key(app_key, fmt):
    pem = app_key.private_bytes(
        serialization.Encoding.PEM, fmt, serialization.NoEncryption()
    ).decode()

    before = int(time.time())
    token = _build_app_jwt(APP_ID, pem)
    after = int(time.time())

    assert isinstance(token, str)
    assert pyjwt.get_unverified_header(token) == {"alg": "RS256", "typ": "JWT"}

    claims = pyjwt.decode(
        token,
        app_key.public_key(),
        algorithms=["RS256"],
        issuer=APP_ID,
        options={"require": ["iat", "exp", "iss"]},
    )
    assert before - 60 <= claims["iat"] <= after - 60
    assert claims["exp"] - claims["iat"] == 10 * 60
    assert claims["exp"] - after <= 10 * 60


def test_app_jwt_does_not_verify_under_another_key(app_key):
    pem = app_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with pytest.raises(pyjwt.InvalidSignatureError):
        pyjwt.decode(_build_app_jwt(APP_ID, pem), other.public_key(), algorithms=["RS256"])

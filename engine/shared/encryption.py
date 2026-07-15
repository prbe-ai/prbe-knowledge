"""Fernet wrapper for at-rest encryption of OAuth tokens in `integration_tokens`.

The key lives in settings.token_encryption_key (Fly secret in prod). Rotation
is a two-step process we'll wire up in Tier 7 cron: add new key, re-encrypt,
drop old key. Phase 0 uses a single active key.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from shared.config import get_settings
from shared.exceptions import ConfigError, MissingSecret


def _fernet() -> Fernet:
    key = get_settings().token_encryption_key.get_secret_value()
    if not key:
        raise MissingSecret("TOKEN_ENCRYPTION_KEY is not set")
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except ValueError as exc:
        raise ConfigError(f"invalid Fernet key: {exc}") from exc


def encrypt_token(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_token(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ConfigError("token decrypt failed — key rotation or tampering") from exc


def generate_key() -> str:
    """Convenience for bootstrap scripts."""
    return Fernet.generate_key().decode("ascii")

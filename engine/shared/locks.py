"""Shared advisory-lock key derivation.

Postgres ``pg_advisory_xact_lock($1)`` takes a single bigint (signed
64-bit). Several call sites in this codebase derive that bigint from a
salt + per-key parts using the same sha256 -> low-8-bytes-signed-bigint
recipe; this module is the single home for that helper.

Salts in production:

    custom-ingest-doc : per-(customer, doc_id) lock the Normalizer takes
                        around the custom-ingest door's read-then-write.
    leiden-community  : per-customer lock the Leiden cron takes around a
                        community-detection pass.

Stable across processes — same input bytes always hash to the same
bigint, so locks work cluster-wide without coordination beyond the DB.
"""

from __future__ import annotations

import hashlib


def advisory_lock_key(salt: str, *parts: str) -> int:
    """Hash ``"salt:part1:part2:..."`` to a 64-bit signed bigint.

    Suitable as the argument to ``pg_advisory_xact_lock($1)`` /
    ``pg_try_advisory_xact_lock($1)``. Stable across processes because
    sha256 is deterministic and the byte/sign extraction is fixed.
    """
    composite = ":".join((salt, *parts))
    digest = hashlib.sha256(composite.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


__all__ = ["advisory_lock_key"]

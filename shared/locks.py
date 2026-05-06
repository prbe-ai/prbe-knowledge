"""Shared advisory-lock key derivation.

Postgres ``pg_advisory_xact_lock($1)`` takes a single bigint (signed
64-bit). Several call sites in this codebase derive that bigint from a
salt + per-key parts using the same sha256 -> low-8-bytes-signed-bigint
recipe; this module is the single home for that helper.

Salts in production:

    bootstrap-trigger : per-customer wipe + insert critical section in
                        the trigger route.
    bootstrap-run     : per-(customer, source) defense-in-depth lock the
                        BootstrapWorker takes around the per-source crawl.
    page              : per-(customer, page_slug) lock the wiki agent
                        takes around the read-then-write in
                        update_page / create_page.

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

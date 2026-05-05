"""Synthetic tenant seeding — admin-triggered population of a real-shape
customer workspace with canonical synthetic content.

See docs/superpowers/specs/2026-05-04-synth-plan-4-tenant-seeding-v1-design.md.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_VALID_PREFIXES: tuple[str, ...] = ("cust-eval-", "cust-synth-")


def is_seed_eligible(customer_id: str, metadata: dict | None) -> bool:
    """Return True if customer_id is allowed to receive synth seed data.

    Rules:
    - cust-eval-* / cust-synth-* prefixes are always eligible (existing rule
      from profile.py:42 _VALID_PREFIXES and clean_tenant at bootstrap.py:119).
    - Other prefixes are eligible only when metadata['allow_synth_seed'] is
      Python True. Truthy strings/ints don't count — explicit boolean only.
    """
    if customer_id.startswith(_VALID_PREFIXES):
        return True
    if metadata is None:
        return False
    return metadata.get("allow_synth_seed") is True


def prompt_typed_confirm(expected_customer_id: str) -> bool:
    """Prompt operator to type the customer_id back literally to confirm.

    Returns True iff the typed input (whitespace stripped) exactly matches
    expected_customer_id. Empty input returns False.

    Reads from sys.stdin so monkeypatch can substitute it in tests.
    """
    print(
        f"To confirm seeding {expected_customer_id!r}, type the customer_id back: ",
        end="",
        flush=True,
    )
    typed = sys.stdin.readline().strip()
    if not typed:
        return False
    return typed == expected_customer_id


async def set_allow_synth_seed(customer_id: str, db) -> None:
    """Toggle customers.metadata.allow_synth_seed = true for the named customer.

    Idempotent: re-running on an already-set tenant is a no-op (the UPDATE
    re-writes the same value). Refuses with ValueError if the customer row
    doesn't exist — synth doesn't auto-create real-shape tenants.

    `db` is an asyncpg Pool (matches the signature of bootstrap.py::init_tenant).
    """
    result = await db.execute(
        """
        UPDATE customers
           SET metadata = jsonb_set(
                   COALESCE(metadata, '{}'::jsonb),
                   '{allow_synth_seed}',
                   'true'::jsonb,
                   true
               )
         WHERE customer_id = $1
        """,
        customer_id,
    )
    parts = result.split()
    affected = int(parts[-1]) if parts and parts[-1].isdigit() else 0
    if affected == 0:
        raise ValueError(
            f"customer {customer_id!r} not found in customers table; "
            f"create the tenant via prbe-backend signup first"
        )


def _substitute_customer_id(
    *,
    payload: dict,
    old_key: str,
    old_id: str,
    new_id: str,
) -> tuple[dict, str]:
    """Rewrite a canonical envelope's customer_id field and R2 key for
    a target tenant.

    V1 scope: only the top-level `customer_id` field on the payload is
    rewritten. Nested references (e.g. thread_parent_id segments that
    happen to contain the canonical customer_id) are left untouched —
    no downstream consumer interprets them as customer_ids.

    Idempotent: if old_id is not in old_key but payload has new_id,
    the transformation was already applied; return unchanged.
    Raises ValueError if old_id is not in old_key and payload doesn't
    have new_id (malformed fixture).
    """
    # Affirmative idempotency: payload already transformed AND key already
    # substituted → return unchanged.
    if old_id not in old_key and payload.get("customer_id") == new_id:
        return dict(payload), old_key

    # Malformed input: old_id missing from key but payload not yet transformed.
    if old_id not in old_key:
        raise ValueError(
            f"old_id not found in R2 key: old_id={old_id!r}, old_key={old_key!r}"
        )

    # Normal case: rewrite both payload and key.
    new_payload = dict(payload)
    new_payload["customer_id"] = new_id
    new_key = old_key.replace(old_id, new_id)
    return new_payload, new_key


# ---------------------------------------------------------------------------
# MissingCanonicalError, SeedResult, seed_tenant
# ---------------------------------------------------------------------------


class MissingCanonicalError(FileNotFoundError):
    """Raised when seed_tenant is given a canonical_dir that doesn't exist
    or has no envelope files."""


@dataclass(frozen=True)
class SeedResult:
    envelopes_processed: int
    r2_uploaded: int
    queued: int
    canonical_customer_id: str


async def seed_tenant(
    customer_id: str,
    canonical_dir: Path,
    db,
    bucket,
) -> SeedResult:
    """Replay a canonical envelope set against a target customer.

    Walks canonical_dir/raw/<source>/*.json. For each envelope:
      1. Substitute customer_id in payload + R2 key.
      2. Upload to R2 under the target customer's prefix (overwrites).
      3. INSERT into ingestion_queue ON CONFLICT DO NOTHING.

    Idempotent on re-run: R2 PUT always overwrites; queue INSERT skips on
    the (customer_id, source_system, source_event_id) unique constraint.

    Caller is responsible for gate checks (eligibility, typed-confirm,
    customer existence) — seed_tenant assumes you've done them.

    Bucket creation: this function calls bucket.ensure_bucket() defensively
    because customers created via prbe-backend signup do NOT run synth's
    init_tenant (which is the usual creator of the R2 bucket). Calling
    ensure_bucket() on an existing bucket is a no-op.

    Canonical assumption: all envelopes under canonical_dir/raw/ must share
    the same customer_id (captured from the first envelope walked). Mixed
    canonical_id values across envelopes will silently produce incorrect
    R2 keys for the second-and-later sources.

    Real schema deviations from plan pseudo-code:
    - Column is `source_system`, not `source`.
    - Payload column is `payload_s3_keys TEXT[]`, not `r2_key TEXT`.
    - ON CONFLICT target is (customer_id, source_system, source_event_id).
    - `priority`, `version`, `enqueued_at` are populated explicitly.
    See scripts/synth/output/writer.py::IngestionWriter._flush_queue for
    the authoritative column list.
    """
    raw_root = canonical_dir / "raw"
    if not raw_root.exists() or not any(raw_root.rglob("*.json")):
        raise MissingCanonicalError(
            f"canonical corpus not found at {canonical_dir}; "
            f"generate it first (see scripts/synth/README.md)"
        )

    bucket_name = bucket.bucket_for(customer_id)
    await bucket.ensure_bucket(bucket_name)

    envelopes = sorted(raw_root.rglob("*.json"))
    canonical_id: str | None = None
    uploaded = 0
    queued = 0

    for env_path in envelopes:
        payload = json.loads(env_path.read_text(encoding="utf-8"))
        if canonical_id is None:
            canonical_id = payload["customer_id"]

        # Derive source and event_id from file path:
        # canonical_dir/raw/<source>/<event_id>.json
        rel = env_path.relative_to(raw_root)
        source = rel.parts[0]         # e.g. "slack"
        event_id = rel.stem           # e.g. "std-001"

        # Reconstruct the canonical R2 key then substitute for target.
        old_key = f"raw/{source}/{canonical_id}/synth/{event_id}.json"
        new_payload, new_key = _substitute_customer_id(
            payload=payload,
            old_key=old_key,
            old_id=canonical_id,
            new_id=customer_id,
        )

        # Upload to R2 (overwrites on re-run — idempotent).
        body = json.dumps(new_payload).encode("utf-8")
        await bucket.put(bucket_name, new_key, body)
        uploaded += 1

        # INSERT into ingestion_queue using the real prbe-knowledge schema.
        # Column list mirrors scripts/synth/output/writer.py::_flush_queue.
        # payload_s3_keys is TEXT[] — wrap key in a list.
        # priority=100 matches schema DEFAULT and SynthDoc default.
        # version=1 matches IngestionWriter's convention for new rows.
        # Mirrors IngestionWriter's INSERT contract verbatim — see
        # scripts/synth/output/writer.py for the reference call site.
        # version=1 (not the schema DEFAULT 0) matches the worker's CAS
        # contract that expects rows to start at version >= 1.
        result = await db.execute(
            """
            INSERT INTO ingestion_queue
              (customer_id, source_system, source_event_id, payload_s3_keys,
               status, priority, version, enqueued_at)
            VALUES ($1, $2, $3, $4, 'pending', 100, 1, NOW())
            ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING
            """,
            customer_id, source, event_id, [new_key],
        )
        # asyncpg returns "INSERT 0 N" where N is the number of rows inserted.
        # ON CONFLICT DO NOTHING yields "INSERT 0 0"; a real insert "INSERT 0 1".
        parts = result.split()
        if parts and len(parts) >= 3 and parts[-1].isdigit() and int(parts[-1]) > 0:
            queued += 1

    if canonical_id is None:
        raise MissingCanonicalError(
            f"no envelopes found under {canonical_dir}/raw/"
        )

    return SeedResult(
        envelopes_processed=len(envelopes),
        r2_uploaded=uploaded,
        queued=queued,
        canonical_customer_id=canonical_id,
    )

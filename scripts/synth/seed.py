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
    """Rewrite a canonical envelope's R2 key for a target tenant.

    Real source-specific envelopes (Slack Events API, Notion webhooks, etc.)
    do NOT carry a top-level `customer_id` field — the customer is identified
    by the R2 bucket and the key path's customer_id segment. So payload
    rewrite is opt-in: only modified when the payload happens to already
    have a `customer_id` field (legacy / synth-only envelopes). Real
    envelopes pass through untouched on the payload axis; only the key is
    substituted.

    Idempotent: if old_id isn't in old_key, assume the substitution has
    already happened and return unchanged. Raises ValueError only if old_id
    is missing AND the payload's optional customer_id (if present) isn't new_id
    yet — that combination indicates a malformed input.
    """
    # Affirmative idempotency: key already substituted (and, if a legacy
    # customer_id field is present, it matches the target).
    if old_id not in old_key and payload.get("customer_id", new_id) == new_id:
        return dict(payload), old_key

    # Malformed: key claims un-substituted state but the payload's customer_id
    # field disagrees.
    if old_id not in old_key:
        raise ValueError(
            f"old_id not found in R2 key: old_id={old_id!r}, old_key={old_key!r}"
        )

    new_payload = dict(payload)
    if "customer_id" in new_payload:
        # Legacy / synth-only envelopes that carry an explicit customer_id
        # field get it rewritten too. Real source-API envelopes don't and
        # are left untouched.
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

    Canonical identity: read from canonical_dir/MANIFEST.json which must
    contain {"canonical_customer_id": "<id>", ...}. This is the customer_id
    the canonical was originally recorded against; seed_tenant substitutes
    it for the target customer_id wherever it appears in R2 keys. Reading
    the canonical_id from a manifest (rather than per-envelope payload)
    works for real source-API envelopes that don't carry a customer_id
    field (Slack Events API, Notion webhooks, etc.).

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

    manifest_path = canonical_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise MissingCanonicalError(
            f"canonical MANIFEST.json not found at {manifest_path}; "
            f"the manifest must declare canonical_customer_id (the "
            f"customer_id the canonical corpus was recorded against)"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissingCanonicalError(
            f"canonical MANIFEST.json is not valid JSON: {exc}"
        ) from exc
    canonical_id = manifest.get("canonical_customer_id")
    if not isinstance(canonical_id, str) or not canonical_id:
        raise MissingCanonicalError(
            f"canonical MANIFEST.json missing required 'canonical_customer_id' "
            f"string field; got {canonical_id!r}"
        )

    bucket_name = bucket.bucket_for(customer_id)
    await bucket.ensure_bucket(bucket_name)

    envelopes = sorted(raw_root.rglob("*.json"))
    uploaded = 0
    queued = 0

    for env_path in envelopes:
        payload = json.loads(env_path.read_text(encoding="utf-8"))

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

        # INSERT into ingestion_queue. Column list mirrors the production
        # paths in services/ingestion/backfill_runner.py and
        # services/ingestion/session_completer.py — both write BOTH the
        # legacy `payload_s3_key` (singular, NOT NULL on prod) AND the
        # new `payload_s3_keys` (TEXT[], introduced in migration 0026
        # for claude_code session-coalescing). The legacy column was
        # intentionally not dropped in 0026 to avoid a rolling-deploy
        # race; a follow-up null-allow ALTER never landed, so prod still
        # enforces NOT NULL even though local db/schema.sql shows it
        # nullable. scripts/synth/output/writer.py::_flush_queue has the
        # same drift and would also fail against prod — separate fix.
        # priority=100 matches schema DEFAULT and SynthDoc default.
        # version=1 (not the schema DEFAULT 0) matches the worker's CAS
        # contract that expects rows to start at version >= 1.
        result = await db.execute(
            """
            INSERT INTO ingestion_queue
              (customer_id, source_system, source_event_id,
               payload_s3_key, payload_s3_keys,
               status, priority, version, enqueued_at)
            VALUES ($1, $2, $3, $4, ARRAY[$4], 'pending', 100, 1, NOW())
            ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING
            """,
            customer_id, source, event_id, new_key,
        )
        # asyncpg returns "INSERT 0 N" where N is the number of rows inserted.
        # ON CONFLICT DO NOTHING yields "INSERT 0 0"; a real insert "INSERT 0 1".
        parts = result.split()
        if parts and len(parts) >= 3 and parts[-1].isdigit() and int(parts[-1]) > 0:
            queued += 1

    return SeedResult(
        envelopes_processed=len(envelopes),
        r2_uploaded=uploaded,
        queued=queued,
        canonical_customer_id=canonical_id,
    )

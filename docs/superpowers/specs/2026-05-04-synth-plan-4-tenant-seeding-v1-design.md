# Plan 4 V1 — Synthetic Tenant Seeding (CLI-only, admin-triggered)

**Status:** spec, awaiting implementation plan.

**Reads:** Plan 4 handoff (`2026-05-04-synthetic-tenant-seeding-design.md`),
Plan 3 spec (`2026-05-02-synthetic-narrative-layer-design.md`),
synth README (`scripts/synth/README.md`).

**Stacks on:** PR #72 (`feat/synthetic-eval-corpus-plan3`). Plan 4 V1 depends on
the 5 post-Plan-3 fixes inside #72; either land #72 first or rebase Plan 4 onto
main once #72 merges.

**Repo scope:** prbe-knowledge only. No prbe-backend, no dashboard, no schema
migrations.

---

## Product motion

Customer wants a "playground" — a workspace populated with realistic-feeling
synthetic content so they can experience how Probe handles context across a
fake corporate/engineering environment without first connecting their own
Slack / GitHub / Linear / Notion / Sentry.

The customer never triggers seeding themselves: LLM cost is the gating reason.
An admin (Mahit or the GTM/eng team) runs `synth seed cust-prbe-acme-co` from
prbe-knowledge after the customer signs up normally. Customer logs in to their
primary workspace and sees a populated dashboard.

Mixing risk (synth docs co-resident with the customer's later-connected real
sources) is **explicitly accepted for V1**. Surgical cleanup is deferred to V2.

---

## Architecture

Two phases, shipped together in one PR.

- **Phase A — safety primitives.** Extend the existing `cust-eval-`/`cust-synth-`
  prefix guard to also accept a metadata flag, plus a typed-confirmation
  escape hatch.
- **Phase B — canonical corpus + seed CLI.** Record a templated-only canonical
  corpus once, commit fixtures to git, ship a `synth seed` subcommand that
  replays canonical envelopes against any seed-eligible customer.

No schema migrations. No new infra. The path through prbe-knowledge from
`synth seed` invocation to populated dashboard reuses Plan 3's existing
`IngestionWriter` → `ingestion_queue` → worker → retrieval pipeline unchanged.

---

## Components

Six pieces, all in `scripts/synth/` (plus README and tests):

1. **`is_seed_eligible(customer_id, metadata)` helper** in
   `scripts/synth/bootstrap.py`. Returns True when the customer_id matches
   `cust-eval-`/`cust-synth-` (existing rule, see `profile.py:42`
   `_VALID_PREFIXES`) OR when `metadata.allow_synth_seed = true`. Pure
   function — caller fetches metadata from the DB. Unit-tested independently.
   Consumed only by the new `seed` flow (Component 4); existing `profile.py`
   prefix check at line 81 stays prefix-only (it gates `run`/`init`, not
   `seed`, and seed doesn't load profiles), and existing `clean_tenant`
   prefix check at `bootstrap.py:119` stays prefix-only (V1 preserves clean
   semantics; surgical clean is V2).

2. **`--allow-non-sandbox` typed-confirmation flag** in `scripts/synth/cli.py`.
   Required for any `synth seed` against a non-eval-prefix tenant when the
   metadata flag is not set. Prompts the operator to type the customer_id back
   literally; mismatch aborts before any DB or R2 write.

3. **Canonical corpus fixtures** under `scripts/synth/canonical/v1/raw/<source>/<event_id>.json`,
   committed to git. Generated once via:

   ```
   synth run --integrate --record-llm \
             --archetypes standup,oncall \
             --output-dir scripts/synth/canonical/v1/
   ```

   Templated-only: `standup` and `oncall` archetypes (`validator_level=NAME_ONLY`).
   Plot archetypes (`incident`, `launch`, `big_refactor`) are deferred — they
   require the validator regen loop (`scripts/synth/scenarios.py:202-211` TODO),
   which is separate work.

4. **`synth seed <customer_id>` subcommand** in `scripts/synth/cli.py`. Reads
   canonical envelopes from `scripts/synth/canonical/v1/raw/`, substitutes
   the canonical `customer_id` for the target customer's id (in both the R2
   key path and the JSON payload's `customer_id` field), uploads to R2 under
   the customer's prefix, inserts `ingestion_queue` rows. Idempotent on
   re-run.

5. **`synth allow-seed <customer_id>` subcommand** in `scripts/synth/cli.py`.
   Sets `customers.metadata.allow_synth_seed = true` for the named customer.
   Refuses if the customer row doesn't exist; idempotent if already set.

6. **Runbook update** — append a "Seeding a real-shape tenant" section to
   `scripts/synth/README.md`. Documents: the one-time canonical-record step,
   the `allow-seed` and `seed` subcommands, the two valid paths to seed, and
   an explicit caveat that existing `synth clean` will wipe everything in
   the tenant including any real connector data (V2 surgical clean is
   deferred — admin's responsibility for now).

---

## Gate semantics

For a non-eval-prefix tenant, `synth seed cust-X` succeeds when **either** path
below is satisfied. Customer-existence check applies to both.

- **Path 1 — opt-in flag.** `customers.metadata.allow_synth_seed = true`,
  set previously via `synth allow-seed cust-X`. No flag required at seed time;
  no typed-confirm prompt.
- **Path 2 — escape hatch.** `--allow-non-sandbox` flag passed at seed time
  AND operator types the customer_id back at the prompt. No DB state change
  required.

Eval-prefix (`cust-eval-*`, `cust-synth-*`) tenants keep their existing
zero-friction path — no flag, no prompt. The new gates only apply to
real-shape customer ids.

In all cases, `synth seed` refuses if the `customers` row doesn't exist.
Tenant creation is `init_tenant`'s job (called by prbe-backend signup),
not `seed`'s.

---

## Data flow

### Flow A — Canonical generation (one-time, by Mahit/eng)

```
synth init cust-eval-canonical-v1                       # throwaway eval tenant
synth run --integrate --record-llm \
          --archetypes standup,oncall \
          --output-dir scripts/synth/canonical/v1/
synth clean cust-eval-canonical-v1                      # tear down eval tenant
git add scripts/synth/canonical/v1/raw/
git commit -m "chore(synth): record canonical v1 corpus (templated-only)"
```

Output: `scripts/synth/canonical/v1/raw/<source>/<event_id>.json`. Same
envelope JSON the synth tool would have written to R2 during a normal
`--integrate` run, but captured to disk. ~50–100 docs, ~1–2 MB total.

### Flow B — Per-customer seed (per-customer, by admin)

```
# Path 1 (recommended for any customer seeded more than once):
synth allow-seed cust-prbe-acme-co
synth seed cust-prbe-acme-co

# Path 2 (one-off, no DB state change):
synth seed cust-prbe-acme-co --allow-non-sandbox
```

Inside `synth seed`:

1. Read `customers` row for the target id. Missing → exit 2 with a clear error.
2. Run gate stack (eval-prefix? metadata flag? `--allow-non-sandbox`? typed
   confirm?). Refuse on any failure with a specific exit code and message.
3. Walk `scripts/synth/canonical/v1/raw/<source>/*.json`. For each envelope:
   - Substitute the canonical `customer_id` for the target id in the R2
     key path AND the JSON payload.
   - Upload to R2 at the customer's prefix (unconditional PUT, overwrites
     prior objects at the same key).
   - Insert into `ingestion_queue` with
     `INSERT … ON CONFLICT (customer_id, source, event_id) DO NOTHING`.
4. Log a summary: `seeded N envelopes (X uploaded, Y already-present in R2;
   M queued, K already-queued)`.
5. Exit. Worker picks up queue rows on its next tick → docs/chunks/graph
   populate over the next ~30s–2min.

The seeded tenant's docs are bound to whatever WorldModel the canonical
recording used. Every seeded customer's playground references the same
fictional company (same Alice/Bob/Charlie cast, same `payments-api`/`auth-svc`
services, same `#oncall`/`#general` channels). Per-tenant org-name customization
is a future extension — out of scope for V1.

---

## Error handling

### Gate failures — fail fast, fail loud, no DB writes

Order of checks in `synth seed cust-X` (cheapest first; no typed-confirm prompt
until the call has cleared everything else):

1. **Customer not found** → `error: customer 'cust-X' not found in customers
   table; create the tenant via prbe-backend signup first`. Exit 2.
2. **Canonical fixtures missing** (`scripts/synth/canonical/v1/raw/` empty
   or absent) → `error: canonical corpus not found at <path>; generate it
   first (see scripts/synth/README.md)`. Exit 1.
3. **No path satisfied** (non-eval-prefix, no metadata flag, no
   `--allow-non-sandbox`) → `error: customer 'cust-X' is not seed-eligible.
   Either run 'synth allow-seed --customer cust-X' first, or pass
   --allow-non-sandbox to seed one-off.` Exit 2.
4. **Typed-confirm mismatch** → `error: confirmation mismatch; expected
   'cust-X'. No data written.` Exit 2.

All four exit before any R2 or Postgres write.

Note: the canonical-missing check was promoted ahead of the eligibility/typed-
confirm gates as a UX optimization — an operator using `--allow-non-sandbox`
shouldn't be asked to type the customer_id back only to discover the canonical
fixtures aren't present. The original spec ordering put canonical last; the
implementation in `cli.py::_seed_async` (Task 8 + polish commit 9c70db4) moved
it to slot 2. `seed_tenant` still raises `MissingCanonicalError` as
defense-in-depth in case the dir disappears between the check and the walk.

### Partial-state recovery — idempotency carries the weight

`synth seed` has two side effects per envelope: an R2 PUT and an
`ingestion_queue` row insert. Both are designed for safe re-run:

- R2 upload — unconditional PUT, overwrites prior object at same key.
  Re-running yields byte-identical state.
- Queue insert — `INSERT … ON CONFLICT (customer_id, source, event_id)
  DO NOTHING`. Re-running is a no-op for already-queued items.

If a seed crashes halfway (R2 5xx, network blip, Postgres conn drop), the
operator re-runs `synth seed cust-X` and it picks up where it left off. No
state-tracking column needed.

### `synth allow-seed` errors

- **Customer not found** → fail loud, exit 2. No auto-create.
- **Flag already set** → no-op; log
  `metadata.allow_synth_seed already true for cust-X`; exit 0.

### Forward-compat note on canonical versioning

The canonical fixtures dir is versioned (`canonical/v1/`). When a future
canonical (e.g., once regen lands and plot scenarios survive) is recorded as
`canonical/v2/`, `synth seed` defaults to whatever is latest with a
`--canonical-version v1` flag to pin. That's a 5-LOC follow-up, not V1 work,
but the directory naming sets it up cleanly.

---

## Testing

Three layers, all under `tests/scripts/synth/`. Existing Plan 3 test
infrastructure (docker-compose stack, `pytest -m integration`) covers the
worker / retrieval side — Plan 4 only adds tests for the new surfaces.

### Unit tests (no DB, no R2)

- **Prefix-guard truth table.** `_is_seed_eligible(customer_id, metadata)`
  across the matrix: `cust-eval-*` → True regardless of metadata;
  `cust-synth-*` → True regardless of metadata; `cust-prbe-*` + flag → True;
  `cust-prbe-*` + no flag → False.
- **Typed-confirm logic.** Exact match returns True; mismatch returns False;
  whitespace stripped; empty input rejected.
- **Customer ID substitution.** `_substitute_customer_id(envelope, old, new)`
  rewrites both the R2 key path and the `customer_id` field inside the
  payload; leaves other fields untouched; idempotent on repeat application.

### Integration tests (against the existing tier-3 docker-compose stack)

- **Happy path end-to-end.** `init_tenant cust-prbe-test-X` →
  `synth allow-seed` → `synth seed` → poll worker → assert `documents` table
  populated, `ingestion_queue` empty, retrieval API returns N docs.
- **Re-run idempotency.** Run `seed` twice; assert second run uploads zero
  net-new R2 objects and inserts zero net-new queue rows.
- **Gate-failure exits write nothing.** Four cases (missing customer / no
  path satisfied / confirm mismatch / canonical missing) each assert
  non-zero exit AND empty R2 prefix AND zero `ingestion_queue` rows for
  the target customer.

### Test fixtures

A tiny stand-in canonical (~5 envelopes) committed under
`tests/fixtures/canonical-mini/` for fast integration runs. The real
`scripts/synth/canonical/v1/` is *not* used in tests — it would inflate test
time and couple test correctness to canonical-corpus regen.

### Out of scope for V1 testing

- Worker drain behavior, ACL snapshot construction, graph node/edge creation
  — all existing Plan 3 code, covered by Plan 3's integration suite. Plan 4
  inherits coverage by re-running it post-change as a regression check.
- Retrieval quality on the seeded corpus — that's an eval question, not a
  Plan 4 question. Seed tests assert *that* docs land, not *what* the
  retriever says about them.
- LLM-mode regen path — deferred to the regen-loop work after V1 ships.

---

## Definitions of done

- `synth allow-seed cust-X` toggles `customers.metadata.allow_synth_seed`
  idempotently; refuses on missing customer.
- `synth seed cust-X` succeeds via either Path 1 or Path 2; refuses with a
  specific exit code on each gate failure; writes nothing on refusal.
- Re-running `synth seed cust-X` is a no-op (idempotent R2 PUT + queue
  ON CONFLICT).
- Templated-only canonical corpus committed under
  `scripts/synth/canonical/v1/raw/`; ~50–100 envelopes; <2 MB on disk.
- `scripts/synth/README.md` updated with the seeding runbook including the
  two paths, the canonical-record step, and the V1 cleanup caveat.
- New unit and integration tests pass; existing Plan 3 integration suite
  passes unchanged.
- `cust-eval-*` flow is unchanged — no regression for existing eval tenants.

---

## Out of scope (explicit defer list)

- **Plot archetype canonical content.** Requires the validator regen loop
  (`scripts/synth/scenarios.py:202-211` TODO). The V1 canonical is templated-only
  (standup + oncall). Re-record canonical as `v2/` after regen lands.
- **prbe-backend admin endpoint** (Phase C from handoff). The seed
  motion stays CLI-only in V1.
- **Dashboard "this is sample data" banner** (Phase D from handoff).
- **Surgical clean / V2 per-row provenance tagging.** `synth clean` keeps
  its existing wipe-everything semantics; admin's responsibility to know
  whether the tenant has real connector data.
- **Per-tenant corpus customization** (substituting customer's org name into
  the docs).
- **Self-serve customer-triggered seeding** (signup-flow opt-in, async job
  queue). All gated behind admin CLI for V1.

---

## Resolutions for handoff Q1–Q6

| Q | Resolution | Why |
|---|---|---|
| Q1 mixing model | Accept mixing for V1 | Customer accepts the risk; surgical clean deferred. |
| Q2 trigger placement | CLI-only, admin only | Customer can't trigger; cost-gated; no signup wiring. |
| Q3 allowlist source | Two paths: metadata flag OR `--allow-non-sandbox` | Either is sufficient; flag is the durable opt-in, flag-less is the one-off escape hatch. |
| Q4 customer ID shape | Real-shape (`cust-prbe-*`) | Customer signs up normally; we seed their primary workspace. |
| Q5 fixture roster | Same canonical for everyone | Cost concern forces canonical replay; per-tenant customization deferred. |
| Q6 determinism | Canonical snapshot, recorded once, replayed | Falls out of Q5. |

---

## Reference: post-Plan-3 commits this depends on

These are required for any of Plan 4 V1's work to function (all in
`feat/synthetic-eval-corpus-plan3`, PR #72):

- `89a27ba` — `init_tenant` supplies `api_key_hash` placeholder
- `e005c80` — Anthropic temp param dropped + `ingestion_queue` schema match
- `cc1cac4` — `integration_tokens.access_token_encrypted` Fernet-encrypted
- `93e83eb` — Notion synth bypass (handler reads inline `entity.body_markdown`)
- `9ac6c4c` — `clean_tenant` runs transaction on a Connection, not the Pool

Without these, the integrate path crashes at multiple steps. Plan 4 V1 is
implemented against the PR #72 branch and rebased to main after #72 merges.

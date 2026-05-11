# Disaster Recovery: Known Gaps and Mitigations

_Last reviewed: 2026-05-10 | Reviewed by: Founders_

## What Neon PITR covers

Neon's 30-day point-in-time recovery (project `dark-breeze-70766184`) covers all
Postgres data: `documents`, `chunks`, `graph_nodes`, `graph_edges`, `usage_events`,
`customers`, `integration_tokens`, `neon_auth.*`, and all other tables. Restore
granularity: any second in the last 30 days. Mechanism: branch from a past timestamp
in Neon console → new read/write endpoint → no downtime on prod while drill runs.

**Prod baseline (2026-05-10):**
13,696 active documents · 61,804 chunks · 30,154 graph nodes · 48,865 graph edges · 5,122 usage events

---

## Gap 1 — R2 object storage is not backed up

**What's at risk:** Document body content stored in Cloudflare R2. Neon holds the
metadata rows (`documents.source_url`, etc.) but not the objects themselves. A Neon
restore gives you rows pointing to potentially missing or inconsistent R2 content.

**Current risk:** HIGH if R2 has a corruption or accidental deletion event.

**Mitigation status:** ❌ None. No R2 versioning or cross-bucket replication configured.

**Required action:** Enable R2 object versioning on the production bucket. Optionally
configure a secondary R2 bucket in a different region as a daily sync target. Add to
engineering backlog before Phase 4.

---

## Gap 2 — Integration token encryption keys live outside the database

**What's at risk:** Every row in `integration_tokens` has `access_token_encrypted` and
`refresh_token_encrypted`. These are encrypted with a key in the application environment
(Fly secrets), not in Neon. A Neon restore gives you ciphertext you cannot decrypt if
the encryption key is lost, rotated without migration, or the app environment is also lost.

**Current risk:** MEDIUM. Keys are stable in Fly secrets today. Risk spikes during full
infrastructure failure or if Fly secrets are accidentally overwritten.

**Mitigation status:** ⚠️ Partial. Keys exist in Fly secrets. No formal key backup or
rotation procedure documented.

**Required action:** Export the current DEK and store it in a secure offline location
(break-glass entry in a sealed vault). Document the rotation procedure. This becomes the
customer-held DEK handoff in the managed-isolated tier — solving it now accelerates Phase 4.

---

## Gap 3 — neon_auth infra tables must be preserved across restores

**What's at risk:** `neon_auth.project_config` (1 row) and `neon_auth.jwks` (1 row) are
infrastructure, not customer data. A naive restore that rolls back these tables breaks
sign-in for every user immediately.

**Current risk:** HIGH if a restore procedure doesn't explicitly skip or re-apply these
tables.

**Mitigation status:** ⚠️ Known. Not in a formal runbook.

**Required action:** Restore runbook must verify `neon_auth.project_config` and
`neon_auth.jwks` have valid rows after every restore. If wiped, re-apply from the latest
prod snapshot of just those two tables before cutting over traffic.

---

## Gap 4 — RTO is unknown

**What we don't know:** How long does it take to branch a 30GB database, run sanity
checks, update `DATABASE_URL`, and redeploy? End-to-end RTO is unquantified.

**Current risk:** LOW operationally, HIGH for enterprise contract negotiations.

**Required action:** Run a timed drill against a staging copy before first enterprise
contract. Target RTO < 30 minutes.

---

## Restore procedure (runbook)

1. **Identify restore timestamp** — find the last known-good point:
   ```sql
   SELECT MAX(occurred_at) FROM usage_events WHERE status = 'success';
   ```

2. **Create PITR branch** — Neon console → prbe-knowledge → Branches → New Branch →
   enable "Restore to specific point in time" → set timestamp → Create Branch.

3. **Run sanity queries** on the branch (do NOT cut over traffic until all pass):
   ```sql
   -- Row counts plausible vs prod baseline
   SELECT COUNT(*) FROM customers;                                                -- expect ~23
   SELECT COUNT(*) FROM documents WHERE valid_to IS NULL AND deleted_at IS NULL;  -- expect ≤ prod
   SELECT MAX(occurred_at) FROM usage_events;                                     -- must be < restore timestamp

   -- Auth infra must be intact
   SELECT COUNT(*) FROM neon_auth.project_config;  -- must be 1
   SELECT COUNT(*) FROM neon_auth.jwks;             -- must be 1

   -- Spot-check largest customer
   SELECT COUNT(*) FROM documents WHERE customer_id = 'willow-voice' AND valid_to IS NULL;
   ```

4. **Update DATABASE_URL** — swap the Fly secret to the branch's connection string and
   redeploy.

5. **Verify health** — hit `/health`, run one real query, tail `fly logs` for 5 minutes.

6. **Communicate** — send incident notification with data loss window (prod timestamp →
   restore timestamp).

---

## Action items

| Action | Owner | By when |
|---|---|---|
| Enable R2 object versioning | Engineering | Before Phase 2 |
| Export DEK to offline backup | Founder | This week |
| Timed restore drill on staging | Engineering | Before Phase 4 |
| Add restore runbook to `docs/` | ✅ Done | 2026-05-10 |

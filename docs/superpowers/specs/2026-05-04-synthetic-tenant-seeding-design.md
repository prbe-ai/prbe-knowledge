# Plan 4 — Synthetic Tenant Seeding (handoff)

**Status:** handoff / pre-spec. Decisions still open. Not yet a buildable plan.

**Author of handoff:** generated 2026-05-04 from a working session that finished
verifying Plan 3 end-to-end and surfaced this as the next product motion.

**Reads:** Plan 3 spec (`docs/superpowers/specs/2026-05-02-synthetic-narrative-layer-design.md`),
Plan 3 plan (`docs/superpowers/plans/2026-05-02-synth-plan-3-narrative-layer.md`),
synth README (`scripts/synth/README.md`), and the post-Plan-3 fix commits
(`89a27ba`, `e005c80`, `cc1cac4`, `93e83eb`, `9ac6c4c`).

---

## The product motion this enables

Plan 3 ships a synthetic corpus that can be queried via the retrieval API on a
sandboxed `cust-eval-*` tenant. That's an **engineering eval tool** —
sufficient for offline retrieval-quality benchmarks, useless to product and
GTM because:

- Real users sign up via prbe-backend's auth flow, get a real-shape `customer_id`
  (e.g. `cust-prbe-acme-co`), and their dashboard session is tied to that id
  via the JWT (see `prbe-backend/app/routers/dashboard/tickets.py:76-81`).
- The synth tool today refuses any customer_id that doesn't start with
  `cust-eval-`/`cust-synth-` (`scripts/synth/profile.py:75-79`,
  `scripts/synth/bootstrap.py:96-99`). New users can't see synth data because
  their tenant isn't synth-shaped.
- There's no tenant-switcher UI. Even if a user manually created a side
  `cust-eval-*` tenant, switching to it would require either DB surgery on
  their session or a fresh signup with a different email.

Plan 4's job: **let real users land on a populated workspace on day 1 so they
can poke at the product without first connecting their own Slack / Notion /
GitHub / Linear / Sentry.**

The user motion looks like:

1. User signs up at `app.prbe.ai` (existing flow, prbe-backend owns this)
2. During signup, user opts into "give me sample data to explore"
3. Backend marks their tenant as seed-eligible and triggers a synth seed
4. Synth populates the user's real tenant with ~50–100 docs across all sources
5. User logs in → dashboard shows populated retrieval queries, graph, mentions, etc.
6. User connects their first real source → synth seed is either left in place
   (mixed mode) or cleared (replacement mode), per design decision below

That's the destination. Most of Plan 4 is the safety architecture to get there
without polluting real customer data.

---

## What's already in place from Plan 3 (don't redo this work)

After the post-Plan-3 fixes, the integrate path works end-to-end on a local
docker-compose stack:

```
synth init      → customers row + R2 bucket + Fernet-encrypted integration_tokens
synth run --integrate → SynthDocs → R2 envelopes → ingestion_queue rows
ingestion worker      → drains queue → documents/chunks/graph_*/acl_snapshots
retrieval API         → /retrieve + /query against the populated tenant
synth clean           → removes everything cleanly
```

5 real bugs were caught by the first end-to-end exercise of this path, all
fixed and pushed (see `scripts/synth/README.md` "Common gotchas" section for
the symptom-to-commit mapping). **Plan 4 starts from a known-working
single-tenant integrate flow**; the architectural challenges below are
multi-tenant safety and lifecycle, not getting bytes onto disk.

Specifically these things WORK today and can be reused:

- The `IngestionWriter` already dispatches all 5 plot sources (slack, notion,
  github, linear, sentry).
- The notion synth bypass (`scripts/synth/output/notion.py` inlines
  `entity.body_markdown`, prod handler reads it as fallback) handles the
  no-OAuth-token case.
- `bootstrap.py::init_tenant` is idempotent and properly Fernet-encrypts the
  stub OAuth tokens.
- `bootstrap.py::clean_tenant` runs in a real Postgres transaction.
- `synth run --integrate` produces deterministic output for `(profile, seed)`.

---

## What's NOT in place that Plan 4 needs

### 1. Customer-level "synth seed allowed" signal

Today: prefix-guarded by string match on `customer_id`. Real-shape customer ids
are refused.

Plan 4: a customer-level flag (most likely in `customers.metadata` JSONB —
already exists, no schema migration needed) that synth tooling checks instead
of/in addition to the prefix. Naming options:

- `metadata.allow_synth_seed: bool` (most direct; semantically narrow)
- `metadata.tenant_kind: 'production' | 'sandbox' | 'eval'` (more nuanced;
  unlocks future flavors)

Setting the flag is an explicit act, not a default. The flag's source of truth
matters — see "Trigger model" below.

### 2. Per-row provenance for surgical cleanup

Today: `synth clean` is `DELETE FROM documents WHERE customer_id = X`. Fine
for `cust-eval-*` tenants where everything is synth. Disastrous on a real
customer who has live data.

Three answers, in increasing engineering effort:

| | V1 — snapshot | V2 — provenance tagging | V3 — shadow tables |
|---|---|---|---|
| **Schema change** | none | `synth_run_id TEXT NULL` on `documents`, `chunks`, `ingestion_queue`, `acl_snapshots`, `graph_nodes`, `graph_edges` | new `synth_documents`, `synth_chunks`, etc. |
| **Worker change** | none | propagate `synth_run_id` from queue → all downstream writes | dual-write or routing |
| **Cleanup model** | "delete everything for customer" — only safe before real connectors | `DELETE WHERE customer_id=X AND synth_run_id IS NOT NULL` | `TRUNCATE synth_*` per tenant |
| **Mixing real + synth** | impossible | yes, with surgical removal | yes, with `UNION ALL` in retrieval |
| **Effort** | 0 | ~150 LOC across 6 tables + worker + cleanup CLI | ~400 LOC, doubles every retrieval JOIN |
| **When does it pay off** | seed-once-then-replace product motion | "explore demo data alongside your real Slack" motion | aggressively isolated demo mode |

**Recommendation: ship V1 first, defer V2 until product evidence.** V1 is
implementable in this branch with one CLI command + a metadata flag check. The
rule: synth seed only allowed when `metadata.allow_synth_seed = true` AND
`(SELECT COUNT(*) FROM documents WHERE customer_id = X) = 0`. Connecting a real
source flips the second condition; the next seed attempt is refused. Users
who want to "reset to demo" trigger an explicit clean.

V3 is overkill for the demo-seed motion — the duplication cost compounds
across every retrieval-pipeline optimization that already exists. Don't.

### 3. Trigger model

Today: synth is a CLI tool. Operator runs it manually with API keys in their
shell.

Plan 4: how does seeding actually fire when a user signs up? Three flavors:

| | Manual admin CLI | prbe-backend internal HTTP endpoint | Background job queue |
|---|---|---|---|
| **Engineer effort** | 0 (already works) | ~50 LOC: new endpoint in prbe-backend that calls into prbe-knowledge synth via the same X-Internal-Knowledge-Key path | ~150 LOC: enqueue task on signup, worker process drains it |
| **User experience** | "talk to support, we'll seed your account" | Synchronous spinner during signup ("setting up your workspace...") — 1–2min | Async ("we're populating your workspace, refresh in a few minutes") |
| **Failure mode** | operator notices | user-facing error during signup | retry; eventual consistency |
| **Right when** | early access waitlist | <100 sign-ups/day, willing to slow signup | scaling demands it |

For the first 100 users, **manual admin CLI** is fine and lets us learn what
demo data actually resonates before automating. The CLI command is already
shippable in Plan 4 V1; everything past that is product/infra.

### 4. Cleanup UX

Two questions to answer:

- Can users clear their own seed data? Likely yes — a "remove sample data"
  button in dashboard settings, calls a prbe-backend endpoint that calls into
  `synth clean` (or a V2 surgical-clean variant).
- What happens when a user connects a real source while seed data is present?
  - Replacement mode (V1): block the connection with "this will replace your
    sample data with your real data — confirm?"
  - Mixed mode (V2): allow it, label seed docs with a "Sample" badge in the
    dashboard.

### 5. Dashboard awareness

Today: dashboard reads from prbe-backend's `/dashboard/knowledge/*` endpoints,
which forward to prbe-knowledge with the user's session-derived `customer_id`.
Synth and real data are indistinguishable downstream.

Plan 4 likely needs a UI signal that the workspace is partially or wholly
synthetic, both as a credibility signal ("this is sample data") and as a
discovery signal ("connect your sources to replace it"). Specific decisions
need to land:

- Banner copy + dismissal model (always show? show until first real connector?
  show only on the demo-data tagged docs?)
- Per-doc badge or workspace-level banner?
- Source of truth for "this is synth": `customers.metadata.last_synth_seed`
  timestamp, a boolean `is_demo`, or per-row `synth_run_id` (V2 only)?

These are dashboard decisions, not prbe-knowledge decisions, but the source of
truth has to live somewhere — almost certainly a column or metadata field on
`customers`.

---

## Open product/design decisions for Plan 4 to nail down

Listed in rough dependency order — answers up the chain unblock decisions
below them.

### Q1. Mixing model

Will users keep seed data alongside their real data, or is seed strictly
"one-shot demo, vanishes the moment you connect a real source"?

- **Replacement-only** → V1 schema changes are sufficient. Simpler. Most
  consumer software uses this model (Notion, Linear, Asana — sample workspace
  is replaced or sidelined when you start real work).
- **Mixed mode** → forces V2 (per-row provenance tagging). Lets users compare
  retrieval quality side-by-side ("here's what answers look like with full
  history"). More B2B-y.

Recommendation: **replacement-only for V1.** Defer mixed mode to V2 if user
research shows demand. Replacement-only is also easier to communicate ("Sample
data goes away when you connect Slack") and easier to test.

### Q2. Trigger placement

CLI-only to start, or do you want self-serve "give me sample data" on signup
in V1?

Recommendation: **CLI-only V1.** Self-serve in V2 once you have evidence the
demo data converts trial → activation. This decouples Plan 4 from
prbe-backend signup-flow work.

### Q3. Allowlist source of truth

Where does `allow_synth_seed` actually get set?

- prbe-backend signup flow → on user opt-in
- prbe-backend admin endpoint → manual operator action
- Direct SQL UPDATE → engineer-only escape hatch

Recommendation: **all three exist, allowlist the metadata flag once and that's
the only source of truth.** Synth tooling never bypasses it; the path that
sets it is a prbe-backend concern.

### Q4. Customer ID shape

Do seed-eligible tenants keep their real `cust-prbe-*`-shape ids, or do they
get a marker prefix (`cust-prbe-demo-acme-*`)?

Recommendation: **real-shape ids.** A marker prefix breaks the illusion in
the dashboard URL and would later need to be migrated when the demo becomes
the real workspace. The `metadata.allow_synth_seed` flag carries the
"synth-eligible" property; the id stays clean.

### Q5. Per-tenant fixture roster

Do all seeded tenants get the **same** corpus, or is the corpus parameterized
by the tenant's repo / org info?

- **Same corpus** (literal copy of `tiny_test`-shape output) → cheapest, fastest,
  least personalized. The user sees Alice and Bob and #oncall — recognizable
  as a demo.
- **Per-tenant corpus** (run synth against the user's actual GitHub repo) →
  most realistic, but requires their GitHub OAuth + a per-user run of synth
  (~$1–2 cost + 5 minutes per user). Quickly becomes the bottleneck.
- **Hybrid** (one canonical demo dataset, with the user's display_name swapped
  in via post-render templating) → mid-cost, mid-realism.

Recommendation: **same corpus, generated once and re-uploaded per tenant.**
Plan 4 V1 doesn't need per-tenant generation — that's a runtime cost we
shouldn't take during signup. The corpus is the "Acme Co"-style demo;
parameterization is a V2 concern.

### Q6. Determinism contract for seed

Plan 3's seed → byte-identical guarantee was scoped to a single tenant's
output dir. For multi-tenant seeding, what's the contract?

- Every seeded tenant gets identical doc text but tenant-scoped `doc_id`s
  (just substitute `customer_id` in the doc id)?
- Or every seed runs independently with the same seed value (slightly cheaper,
  but doc texts will diverge over time as model versions drift)?

Recommendation: **canonical corpus snapshot.** Run synth once against a
reference repo, store the resulting envelopes in R2 under
`raw/<source>/_canonical/synth/<event_id>.json`, then per-tenant seeding is
just a copy + customer_id substitution at write time. Stable, fast, and
versionable (`canonical-v1`, `canonical-v2`, etc.).

---

## Concrete tasks Plan 4 should land

These follow a recommended-path-of-V1 + replacement-mode + CLI-only
implementation. Tweak based on Q1–Q6 answers.

### Phase A — safety primitives (no behavior change yet)

1. **Add `metadata.allow_synth_seed` flag check** to `bootstrap.py::init_tenant`,
   `cli.py::_run_async`, and `cli.py::_clean_async`. Default: refuse if flag
   not set on a non-`cust-eval-`/`cust-synth-` tenant. Existing eval tenants
   continue to work without the flag.
2. **Add the empty-tenant guard** for non-eval-prefix tenants: refuse seed if
   `customers` has any existing `documents` rows. Eval tenants bypass this
   guard for re-runs.
3. **Add `--allow-non-sandbox` confirmation flag** with a typed-confirmation
   prompt. Required to seed any non-`cust-eval-` tenant even with the metadata
   flag set. Belt-and-suspenders.

### Phase B — canonical corpus snapshot

4. **Generate a canonical corpus** with `synth run --integrate --output-dir
   canonical/`. Commit the resulting `raw/` envelopes to a known R2 location
   (`raw/_canonical/synth-tiny-test-v1/...`). Cost: one-time ~$1.
5. **Add `synth seed <customer_id>` CLI subcommand** — copies canonical
   envelopes to the customer's R2 prefix with id substitution + queues
   ingestion_queue rows. Idempotent. ~50 LOC.

### Phase C — backend integration (separate repo, separate PR)

6. **Add a prbe-backend admin endpoint** `POST /admin/synth-seed/{customer_id}`
   that calls into prbe-knowledge `synth seed` via X-Internal-Knowledge-Key.
   Auth: admin-only.
7. **Wire the metadata flag into prbe-backend signup flow** behind a feature
   flag. Default off; flip on once dashboard banner ships.

### Phase D — dashboard signal (separate repo, separate PR)

8. **Add a "this workspace has sample data" banner** in prbe-dashboard,
   driven by `customers.metadata.last_synth_seed` timestamp + a "remove
   sample data" CTA.
9. **Add the `synth seed clear <customer_id>` CLI** + corresponding admin
   endpoint, gated on V1's empty-other-tenant guard meaning "if you've
   connected real sources, this fails — you'd need V2 provenance tagging".

### Phase E — eval-extending work (this can ship in Plan 4 or land later)

10. **Real validator regen loop** (Plan 3 deferred). Today plot scenarios
    drop on Pass 1 strictness in real-LLM mode. Until this lands, the
    canonical corpus is templated-only or built from --record-llm fixtures
    once and replayed. Without regen, plot eval questions never reach the
    seeded corpus.

---

## Out of scope for Plan 4

- Mixed-mode seed-with-real-data (V2 — per-row provenance tagging). Defer.
- Self-serve "create demo workspace" UI in prbe-dashboard. Defer to V2.
- Per-tenant corpus generation against the user's actual repos. Defer to V2.
- Cross-region replication of canonical corpus (V3 / production scaling).
- Cost ceiling for synth runs (Plan 3 carry-over).
- 3 remaining plot archetypes (`PERF_REGRESSION`, `DEPENDENCY_BUMP`,
  `CUSTOMER_ESCALATION`) — Plan 3 carry-over.
- 4 meeting archetypes + Granola wrapper — Plan 3 carry-over.

---

## Cross-repo coordination map

Plan 4 spans three repos. Land in this order to minimize coupling:

1. **prbe-knowledge** (this repo) — Phase A + B above. Can ship behind a
   feature flag (`metadata.allow_synth_seed`) with no behavior change for
   existing users. PR can land independently.
2. **prbe-backend** — Phase C. Depends on prbe-knowledge's `synth seed`
   command + X-Internal-Knowledge-Key forwarding. Admin endpoint can ship
   first; signup flag flip is a separate PR with QA.
3. **prbe-dashboard** — Phase D. Depends on prbe-backend exposing the
   `metadata.last_synth_seed` field via the `/dashboard/knowledge/*` BFF.
   Pure frontend work once the backend reads land.

The handoff between sessions / engineers happens at Phase A → Phase C
(prbe-knowledge ready; prbe-backend can now wire it up). Phase B is a
one-shot generation step that doesn't block the integration.

---

## Risks and known unknowns

- **Cost regression on canonical corpus.** Re-generating after model upgrades
  costs ~$1 + latency; doable but the cron / ops process for that hasn't been
  designed.
- **The Plan 3 plot archetype validator strictness issue** is currently
  blocking plot-source content from landing in the corpus. Phase E above is
  a hard dependency on that fix if Plan 4 wants the demo to include incidents
  / launches / refactor scenarios. Without it, the canonical corpus is
  templated-only — slack standups + on-call handoffs only.
- **Real Notion API hydration** still happens during ingestion even for synth
  data — `fetch_supplementary` makes a doomed HTTP call per notion doc that
  401s and then falls back to the inlined content. Wasted round-trip per
  ingestion; not a correctness issue. Worth a small fix in Plan 4 to skip
  the fetch when the integration_token decrypts to `synth-stub`.
- **The `acl_snapshots` table** is populated by the worker for every doc.
  Synth personas (`gh:alice`, `email:bob@example.com`) get inserted into
  the real tenant's ACL. Not catastrophic — they just don't match any real
  user — but worth thinking about whether ACL rows should be tagged with
  `synth_run_id` so they can be removed cleanly.

---

## Definitions of done for Plan 4

- A real-shape `customer_id` (e.g. `cust-prbe-acme-co`) can be seeded with
  synth data via a CLI command, gated on a metadata flag + empty-tenant check
  + typed confirmation.
- The seeded tenant is queryable via the production retrieval API (already
  works today; nothing to do).
- `synth seed clear <customer_id>` removes the seeded data without touching
  customer-owned rows (in V1: refuses if real connectors exist; in V2:
  surgical removal via `synth_run_id`).
- An admin endpoint in prbe-backend can trigger seeding for a tenant.
- A dashboard banner indicates seed data is present, sourced from
  `customers.metadata.last_synth_seed`.
- All unit + integration tests green; existing `cust-eval-*` flow unchanged.
- README updated with the new "seed a real-shape tenant" runbook.

---

## Reference: what the post-Plan-3 fix commits enable

These are required for any of Plan 4's V1 work to function:

- `89a27ba` — `init_tenant` supplies `api_key_hash` placeholder
- `e005c80` — Anthropic temp param dropped + ingestion_queue schema match
- `cc1cac4` — `integration_tokens.access_token_encrypted` Fernet-encrypted
- `93e83eb` — Notion synth bypass (handler reads inline `entity.body_markdown`)
- `9ac6c4c` — `clean_tenant` runs transaction on a Connection, not the Pool

Without these, the integrate path crashes at multiple steps. Plan 4
implementations against an older Plan 3 base will need to either pull
these in or rebase.

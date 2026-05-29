<!--
  ⚠️ PROBE-INTERNAL — DO NOT SHIP THIS FILE IN THE PUBLIC OSS REPO.
  This runbook is deleted as a mandatory step (Step 4) before the public push.
-->

# prbe-knowledge → OSS Publish Runbook (Spec C Part 6)

**Status:** prepared, NOT executed. The public push is a HARD STOP requiring the
owner's explicit go-ahead. **License decided: AGPL-3.0.**

This is the gated final step of open-sourcing `prbe-knowledge`. Specs B, A
(Parts 2–5), C (Parts 1–5), B Part 2 (table drop), and the loose-ends fixes are
all merged + deployed. What remains is making the repo public safely without
diverging from the hosted product.

> **Dual-mode invariant (do not break):** the public repo becomes the CANONICAL
> engine source; the hosted product (`prbe-data-plane-image`) must build from it,
> not from a private fork. Multi-tenant code (RLS, per-tenant routing, gateway
> trust, backend-fetch, LiteLLM gateway) STAYS — it's dormant in community mode,
> not removed.

---

## Pre-flight (state at time of writing)
- Engine is standalone-capable (Spec A) + self-host-ready (Spec C: compose, Helm,
  docs, `.env.example`).
- **Secret scan (working tree):** no real secrets — only synthetic fixtures in
  `tests/test_codegraph_secrets.py`. `gitleaks`/`trufflehog` are NOT installed
  locally (install before Step 3).
- **Internal identifiers to genericize (Step 2):** ~`prbe.ai`×166, `@prbe.ai`
  emails×126, `probe-founders`×46, `willow-voice`×3, cluster names×4 — almost all
  in test fixtures + CI workflows.
- **No phone-home/license** in this repo (`control.prbe.ai`: 0 hits; the license
  heartbeat lives in `prbe-backend`).
- **Deferred (T4 gate):** vendoring `prbe-knowledge-mcp` (Spec A Part 1) + the MCP
  bits of Spec C (mcp compose service, `docs/mcp.md`, mcp Helm deployment) are
  blocked on the domain session's MCP issuer cutover. Do them post-T4; they are
  NOT prerequisites for this publish.

---

## Step 1 — LICENSE + SECURITY.md  (safe, local, reversible)
- [ ] Add `LICENSE` at repo root = full **AGPL-3.0** text (verbatim from
      https://www.gnu.org/licenses/agpl-3.0.txt). Set the copyright line.
- [ ] Add `SECURITY.md` (how to report vulnerabilities; a contact + disclosure
      window).
- [ ] Confirm no telemetry/phone-home remains:
      `grep -rn "prbe.ai\|telemetry\|heartbeat\|pairing" --include=*.py services shared`
      → expect only the device-token pairing flow in `shared/tokens.py` (legit) +
      doc/comment references.

## Step 2 — Genericize internal identifiers  (reviewable diff, low risk)
Replace internal references with generic placeholders on the tree that will
become the public repo. These are overwhelmingly in test fixtures + workflows;
the engine code is unaffected (and CI test runs are disabled).
- [ ] Employee emails: `*@prbe.ai` → `*@example.com` (test fixtures).
- [ ] Customer slugs: `probe-founders`, `willow-voice` → `acme` / `demo-tenant`.
- [ ] Internal hosts/clusters: `*.prbe.ai`, `control.prbe.ai`, `do-sfo3`,
      `probe-managed` → generic examples or remove.
- [ ] Re-scan to confirm:
      `git grep -nIE "prbe\.ai|probe-founders|willow-voice|do-sfo3|probe-managed"`
      → only intentional, generic-safe references remain.
- [ ] Decide INTERNAL-ONLY paths to EXCLUDE from the public tree (they power
      hosted, are not part of the OSS engine — do NOT delete from the live private
      repo, just keep them out of the public snapshot):
      `.github/workflows/{agent-optimization-nightly,nightly-improvement-resume,knowledge-cron}.yml`,
      `.github/workflows/dispatch-data-plane.yml`, `k8s/jobs/`, `STEPS.md`,
      `TODOS.md`, any remaining internal `docs/`.

## Step 3 — Final secret scan  (gate; nothing public yet)
- [ ] Install a scanner: `brew install gitleaks` (or trufflehog).
- [ ] Scan the cleaned working tree: `gitleaks detect --no-git --source .`
- [ ] Resolve EVERY hit. The synthetic keys in `tests/test_codegraph_secrets.py`
      are expected (the secret-redaction test) — allowlist them explicitly, don't
      just ignore.

## Step 4 — ‼️ DELETE THIS RUNBOOK, then create the public repo  (PUBLIC EXPOSURE — owner go-ahead required)
- [ ] **Delete this file:** `git rm PUBLISH_RUNBOOK.md` (it is Probe-internal and
      MUST NOT appear in the public repo). Do this BEFORE seeding the public repo.
- [ ] Snapshot the cleaned tree (excluding the Step-2 internal paths) into a NEW
      public GitHub repo seeded with a SINGLE squashed commit — no private history
      travels:
      ```
      # in a clean export of the cleaned tree (internal paths removed):
      git init && git add -A && git commit -m "Probe knowledge engine (initial public release)"
      git remote add origin git@github.com:<org>/<public-repo>.git
      git push -u origin main
      ```
- [ ] Tag a release (e.g. `v0.1.0`) matching `deploy/helm/Chart.yaml`.
- [ ] Owner confirms the public repo contents before it's made non-private.

## Step 5 — Re-point the hosted build  (CROSS-REPO · PROD-AFFECTING · coordinate, do NOT do solo)
The public repo is now canonical. The hosted product must build from it.
- [ ] In `prbe-data-plane-image`: update `build/versions.lock` `PRBE_KNOWLEDGE_SHA`
      source to pin the PUBLIC repo instead of the private `prbe-knowledge`.
- [ ] Run one hosted deploy from the public source; verify pod image SHA + tenant
      health (RLS isolation + LiteLLM metering unregressed) — the dual-mode
      acceptance test.
- [ ] Retire/lock the private `prbe-knowledge` repo so the two never diverge.
- [ ] Coordinate this with whoever owns the image composition — a bad cutover
      breaks every tenant's deploys.

---

## Verification (post-publish)
- [ ] Fresh clone of the PUBLIC repo → `cp .env.example .env` (provider keys only)
      → `docker compose up` → `make health` 200 → `make seed` + `make query`
      return a result. Zero `*.prbe.ai` egress in logs (the community acceptance
      test, on amd64 — see the arm64 caveat in `docs/self-hosting.md`).
- [ ] Hosted multi-tenant deploy from the public source is unregressed.

## Rollback / safety notes
- Steps 1–3 are local + reversible (no push).
- Step 4 is the public-exposure gate; once pushed public it cannot be fully
  un-published — review the tree first.
- Step 5 is the prod cutover; keep the private repo until the public-sourced
  hosted deploy is verified, then retire it.

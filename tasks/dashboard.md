# Production dashboard (prbe-knowledge)

**Status:** plan locked after `/plan-eng-review` 2026-04-24. Implementation not started.

Authoritative plan for: production Next.js dashboard with Google sign-in (via Neon Auth), team-scoped tenants, invitations to non-existent users via email, and removal of the X-Admin-Key control plane.

---

## Decisions locked

| # | Decision | Rationale |
|---|---|---|
| 1 | Auth: **Neon Auth (Stack Auth)** for Google sign-in, gated on a pre-flight verification | DB already on Neon; managed user-table sync via `neon_auth.users_sync` saves user-mirror code |
| 2 | Schema: composite PK `(customer_id, user_id)` + **`UNIQUE(user_id)`** "for now" | Multi-team-ready data model; one team per user enforced by removable unique. Switcher UI deferred. |
| 3 | Invite delivery: **FastAPI BackgroundTasks** + `notified_at` column + manual resend endpoint | Async send; durability handled via column-driven UI surface, not outbox |
| 4 | Last-owner-leave: **blocked** (409 "transfer or delete first") | Reversibility instinct; team-deletion is a separate ceremony |
| 5 | Team deletion: **soft-delete** (`customers.status = 'deleted'`) + offline reaper | Misclick recovery, R2 + cascade hygiene |
| 6 | Backend: **new `services/dashboard/`** FastAPI service (not router-in-ingestion) | Existing precedent of one-service-per-concern; isolates deploy + secrets |
| 7 | Customer CRUD: **extracted to `shared/services/customers.py`** | DRY between admin routes and dashboard routes |
| 8 | Audit: **`audit_log` wired** for invite, role-change, member-remove, team-delete | Existing table; no schema work |
| 9 | Existing tenants: **migrate all to team-managed**, **kill X-Admin-Key flow** | Single control plane, no drift |
| 10 | Email: **Resend** with verified `prbe.ai` domain (SPF/DKIM/DMARC) | Boring choice; React Email templates |
| 11 | Frontend: **new repo `prbe-knowledge-dashboard/`** forked from `-dash-test` | Production hygiene; preserve test dashboard for ad-hoc admin use during transition |

## What's NOT in scope

- Multi-team UI (team switcher, active-team header) — schema is ready; UI deferred
- Hard-delete reaper for soft-deleted teams (separate cron, future ticket)
- 2FA, hardware keys, SSO/SAML/SCIM, email/password auth (Google-only)
- Email-change reissue of pending invitations (acceptable edge — invitee can request fresh)
- Per-team subscription/billing
- Activity feed UI (audit_log writes are scoped here; reading UI is later)
- Cross-tenant role hierarchies (super-admin) — break-glass via direct DB access until needed
- Existing P0/P1 ingestion security TODOs (orthogonal — see TODOS.md)

## What already exists (reused, not rebuilt)

| Sub-problem | Existing file | Reuse plan |
|---|---|---|
| Customer create/list/integrations/stats/rotate-key/delete | `services/ingestion/admin_routes.py` | Logic extracts to `shared/services/customers.py`; admin routes thin to wrapper, then deleted in Phase 9 |
| asyncpg pool + RLS GUC binding | `shared/db.py:with_tenant()` | Reuse as-is |
| Migration tooling | `db/`, `alembic.ini` | One new revision |
| Audit infra (table only) | `db/schema.sql:audit_log` | Wire writes via new `shared/services/audit.py` |
| UI primitives | `prbe-knowledge-dash-test/components/ui/` | Copy wholesale |
| Themed CSS variables | `prbe-knowledge-dash-test/globals.css` | Copy wholesale |
| Server-side backend client | `prbe-knowledge-dash-test/lib/prbe-client.ts` | Refactor to use Neon Auth session token |
| `customers` table | `db/schema.sql` | Add team_members + invitations FKs; add status filter pervasively |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         User (browser)                              │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               │ HTTPS
               ▼
┌────────────────────────────────────────────────────────────────────┐
│   prbe-knowledge-dashboard (Next.js, Vercel)                        │
│   - @stackframe/stack SDK (Neon Auth client)                        │
│   - Server-side route handlers proxy to backend                     │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               │ Authorization: Bearer <neon-auth-session-jwt>
               ▼
┌────────────────────────────────────────────────────────────────────┐
│   services/dashboard (FastAPI, Fly.io)                              │
│   - Validates Neon Auth JWT (JWKS)                                  │
│   - Extracts user_id from token                                     │
│   - Resolves team_members → tenant_id + role per request            │
│   - Routers: /me, /teams, /teams/:id, /teams/:id/members,           │
│     /teams/:id/invitations, /invitations/:token (peek + accept)     │
│   - Writes audit_log on every mutation                              │
│   - BackgroundTasks → Resend for invite emails                      │
└──────────────┬─────────────────────────────────────┬────────────────┘
               │                                      │
               ▼                                      ▼
┌────────────────────────────┐      ┌────────────────────────────────┐
│   Neon Postgres            │      │   Resend (invite emails)        │
│   - customers              │      │   - prbe.ai verified domain     │
│   - team_members           │      │   - SPF/DKIM/DMARC              │
│   - invitations            │      │   - React Email template        │
│   - audit_log              │      └─────────────────────────────────┘
│   - neon_auth.users_sync   │
│     (FDW, Neon-managed)    │
└────────────────────────────┘
```

### Sign-in & invite-claim sequence

```
new user (no account, has invite)
═══════════════════════════════════
  click invite link → /accept?token=xyz
  ↓ (no session)
  redirect to Neon Auth Google flow
  ↓
  Google OAuth callback (Neon Auth handles)
  ↓
  Neon Auth creates user → users_sync row (FDW eventually consistent)
  ↓
  redirect back → dashboard /post-signin
  ↓
  GET /api/me  →  backend validates Neon Auth JWT
  ↓
  backend: claim_pending_invites(user.email from session token)
    └ INSERT team_members FROM invitation, UPDATE invitation.accepted_at
  ↓
  GET /api/me returns user + team_members → dashboard

existing user (account, no invite, no team)
═══════════════════════════════════════════
  sign in → /post-signin → /me returns no team_members
  → land on /onboarding/create-team
  → POST /api/teams (create customer + team_member owner)

existing user with team, gets new invite
═════════════════════════════════════════
  invite create blocks at backend (409 "already on a team")
  → dashboard surfaces: "this user is on another team; cannot invite"
```

### Schema additions (Alembic revision)

```sql
-- New tables
CREATE TABLE team_members (
  customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
  user_id     TEXT NOT NULL REFERENCES neon_auth.users_sync(id) ON DELETE CASCADE,
  role        TEXT NOT NULL CHECK (role IN ('owner','admin','member')),
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (customer_id, user_id),
  UNIQUE (user_id)                 -- "for now" — drop to enable multi-team
);
CREATE INDEX idx_team_members_user     ON team_members(user_id);
CREATE INDEX idx_team_members_customer ON team_members(customer_id);

CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE invitations (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
  email        CITEXT NOT NULL,
  role         TEXT NOT NULL CHECK (role IN ('admin','member')),
  token        TEXT NOT NULL UNIQUE,                -- 32-byte CSPRNG, base64url
  invited_by   TEXT NOT NULL REFERENCES neon_auth.users_sync(id),
  message      TEXT,
  expires_at   TIMESTAMPTZ NOT NULL,                -- default NOW() + 7d
  notified_at  TIMESTAMPTZ,                          -- set on Resend ACK; NULL = not yet
  accepted_at  TIMESTAMPTZ,
  revoked_at   TIMESTAMPTZ,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_invitations_pending_unique
  ON invitations(customer_id, lower(email))
  WHERE accepted_at IS NULL AND revoked_at IS NULL;
CREATE INDEX idx_invitations_email_pending
  ON invitations(lower(email))
  WHERE accepted_at IS NULL AND revoked_at IS NULL;
CREATE INDEX idx_invitations_customer
  ON invitations(customer_id)
  WHERE accepted_at IS NULL AND revoked_at IS NULL;

-- Soft-delete filter on customers (status enum already exists; ensure 'deleted' allowed)
-- All reads go through service layer that filters status != 'deleted'
```

---

## Phased implementation

### Phase 0 — Neon Auth pre-flight verification

- [ ] Invoke `/neon-postgres` skill to confirm:
  - [ ] Google-only sign-in supported (no forced email/password)
  - [ ] Custom UI supported (no mandatory hosted widget)
  - [ ] FK from app tables to `neon_auth.users_sync.id` is enforced
  - [ ] `users_sync` FDW lag bounds (seconds, minutes?)
  - [ ] Pricing past free tier (active user count)
- [ ] If any blocker: revert auth choice to Auth.js v5; document in this file and re-plan affected phases

### Phase 1 — Schema migration

- [ ] Provision Neon Auth on the project (creates `neon_auth.users_sync`)
- [ ] New Alembic revision `db/migrations/000X_dashboard_schema.py` with the SQL above
- [ ] Update `db/schema.sql` (canonical) with the new tables
- [ ] `pgvector` + `citext` extension declarations
- [ ] Forward + downgrade migration tests in `tests/migrations/`
- [ ] Apply on dev branch; verify FK to `neon_auth.users_sync` works

### Phase 2 — Shared service modules

- [ ] `shared/services/customers.py`:
  - [ ] `create_customer(display_name, owner_email) -> Customer`
  - [ ] `soft_delete_customer(customer_id)`
  - [ ] `list_customer_integrations(customer_id)`, `customer_stats(...)` etc. (extract from `services/ingestion/admin_routes.py`)
  - [ ] All read paths filter `status != 'deleted'`
- [ ] `shared/services/audit.py`:
  - [ ] `log(conn, customer_id, actor_id, action, resource_type, resource_id, details)`
  - [ ] Called inside the same txn as the mutation
- [ ] `shared/services/team.py`:
  - [ ] `add_member(customer_id, user_id, role)`, `remove_member(...)`, `update_role(...)`
  - [ ] `is_last_owner(customer_id, user_id) -> bool`
- [ ] `shared/services/invitations.py`:
  - [ ] `create(customer_id, email, role, invited_by, message) -> Invitation`
  - [ ] `peek(token)`, `accept(token, accepting_user_id, accepting_email)`, `revoke(id)`
  - [ ] `claim_pending_invites_by_email(user_id, email)` for sign-in
  - [ ] Token = `secrets.token_urlsafe(32)`; constant-time compare
- [ ] Refactor `services/ingestion/admin_routes.py` to call the shared modules
- [ ] All existing admin route tests pass

### Phase 3 — `services/dashboard/` FastAPI service

- [ ] New service skeleton: `services/dashboard/main.py`, `Dockerfile.dashboard`, `fly.dashboard.toml`
- [ ] Neon Auth JWT validation middleware (JWKS fetch + cache + `require_user`)
- [ ] Routers:
  - [ ] `auth/` — `POST /auth/post-signin` (claim pending invites + return `/me`)
  - [ ] `me/` — `GET /me` (returns user + active team_member)
  - [ ] `teams/` — `POST /teams` (create), `GET /teams/:id`, `DELETE /teams/:id` (soft, typed-confirmation), `POST /teams/:id/leave` (with last-owner block)
  - [ ] `teams/members/` — `GET /teams/:id/members`, `PATCH /teams/:id/members/:user_id`, `DELETE /teams/:id/members/:user_id`
  - [ ] `teams/invitations/` — `POST` (create + BackgroundTask Resend), `GET` (list pending), `DELETE /:id` (revoke), `POST /:id/resend` (re-fire BackgroundTask)
  - [ ] `invitations/` — `GET /invitations/:token` (peek, no auth), `POST /invitations/:token/accept` (auth required, email-match check)
- [ ] RBAC dependency `require_role(min_role)` per endpoint
- [ ] Rate limit `/invitations/:token/*` at 30 req/min/IP
- [ ] OpenAPI schema generation for typed frontend client
- [ ] CI: tests + mypy + ruff (matches existing `pyproject.toml`)

### Phase 4 — Email (Resend)

- [ ] Resend account, API key in Fly secrets as `RESEND_API_KEY`
- [ ] DNS: SPF, DKIM, DMARC records on `prbe.ai` (`invites@prbe.ai`)
- [ ] React Email template `services/dashboard/email/invite_template.tsx`
- [ ] `services/dashboard/email/resend_client.py` (sync send, called from BackgroundTask)
- [ ] On Resend ACK: `UPDATE invitations SET notified_at = NOW() WHERE id = :id`
- [ ] Bounce-webhook stub (writes to audit_log; full handling deferred)

### Phase 5 — `prbe-knowledge-dashboard/` repo scaffold

- [ ] `git init` new repo
- [ ] Copy from `-dash-test`: `components/ui/`, `globals.css`, `tsconfig.json`, `tailwind.config.*`, route handler patterns
- [ ] Strip: cookie-tenant store, admin-key references, all `/onboard*` and `/tenants/*` routes (replaced)
- [ ] Auth wiring: `@stackframe/stack` provider in `app/layout.tsx`
- [ ] `lib/api-client.ts` — wraps backend calls with Neon Auth session token
- [ ] Vercel project provisioned, env vars set

### Phase 6 — Sign-in & onboarding flows

- [ ] `/sign-in` page (Neon Auth Google button, custom UI)
- [ ] `/post-signin` page → calls `POST /api/auth/post-signin` → routes to:
  - `/dashboard` (has team)
  - `/onboarding/create-team` (no team, no invite)
  - errors surfaced inline
- [ ] `/onboarding/create-team` form + creates team, returns to `/dashboard`
- [ ] `/accept-invite/[token]` — peek invite, show team + role + inviter, "Sign in to accept" button
- [ ] Email-mismatch UI: "this invite is for {email}; sign in as that account"

### Phase 7 — Team UI

- [ ] `/dashboard/team` — members list (with pending invites + "email pending" badge)
- [ ] Invite modal: email + role + optional message
- [ ] Role-change inline (admin/owner only, role gating UI)
- [ ] Remove member action (admin/owner only, role gating UI)
- [ ] Leave team button (with last-owner-blocked toast)
- [ ] `/dashboard/team/settings` — display name, "Delete team" with typed-confirmation modal
- [ ] All existing tenant detail pages from `-dash-test` (integrations, ingestion, query) reauthed for team-managed access

### Phase 8 — E2E tests + deployment

- [ ] Playwright suite (8 user flows from coverage diagram)
- [ ] Backend pytest suite (all branches from coverage diagram)
- [ ] Deploy `services/dashboard/` to Fly.io (dashboard.api.prbe.ai)
- [ ] Deploy dashboard to Vercel (dashboard.prbe.ai)
- [ ] Smoke test the full invite flow with two real Google accounts

### Phase 9 — Migrate existing tenants, kill X-Admin-Key

- [ ] `scripts/claim_existing_tenants.py` takes a CSV of (customer_id, owner_email):
  - If owner_email exists in `users_sync`: insert team_members owner row directly
  - Else: insert a pending invitation that auto-claims on first sign-in
- [ ] Run on each existing tenant
- [ ] Audit: every `customers` row has at least one team_member
- [ ] Remove `X-Admin-Key` middleware from `services/ingestion/admin_routes.py` (or move routes to dashboard service)
- [ ] Decommission `prbe-knowledge-dash-test` (archive)

---

## Test plan (coverage targets)

See full diagram in the eng-review test plan artifact at
`~/.gstack/projects/prbe-ai-prbe-knowledge/<user>-main-eng-review-test-plan-*.md`.

**Backend (pytest):** every branch under `services/dashboard/` and `shared/services/`. Specifically:
- Migration forward + downgrade
- Sign-in: new user / existing user / invite claim / FDW lag
- Invite create: happy / email-on-team-already / pending-dup / non-admin / role=owner reject / BackgroundTask error doesn't roll back invite
- Invite peek: valid / expired / revoked / accepted
- Invite accept: happy / email mismatch / user-on-team-already / expired/revoked / parallel-accept race
- Member CRUD: every role × every action matrix
- Last-owner-leave: 409
- Team delete: happy / non-owner / wrong confirmation / subsequent reads 404

**Frontend (Playwright E2E):** 8 flows
1. First sign-up no invite → create team
2. First sign-up with pending invite → auto-joined
3. Admin invites email → recipient flow → joined
4. Invite link clicked while signed in as wrong account → email-mismatch UI
5. Invite email already on different team → 409 with clear UI
6. Last owner clicks Leave → blocked
7. Owner deletes team → soft-deleted, redirect to create-team
8. Resend down → invite created, retry button works

---

## Failure modes

| Codepath | Realistic failure | Test | Handling | UX |
|---|---|---|---|---|
| Google callback | Google JWKS rotation | Yes | JWKS cache refresh | Clear "sign in again" |
| Invitation claim | FDW lag — users_sync row not visible | Yes | Use user_id from session token, not JOIN | No silent failure |
| Accept race | Two tabs accept same token | Yes | UNIQUE on user_id + advisory lock | One sees "already accepted" |
| BackgroundTask email | Fly redeploy mid-send | Manual | `notified_at` NULL → resend button | Admin sees "email pending" badge |
| Soft-deleted team | Cached frontend reads | Yes | Service-layer status filter | Redirect to create-team |
| Last-owner leave | Owner clicks Leave | Yes | 409 | Clear "transfer or delete first" |
| Token brute force | Attacker probes /accept | Manual | 30 req/min/IP rate limit | 429 |
| Email enumeration via 409 | Admin probes for emails on other teams | Acceptable | Response is consistent (admin is trusted role) | Documented limitation |

---

## Worktree parallelization

| Lane | Steps | Depends on |
|---|---|---|
| **Lane 1 (backend)** | P1 → P2 → P3 → P4 (sequential, share modules) | — |
| **Lane 2 (frontend scaffold)** | P5 (independent until P6) | — |
| **Lane 3 (ops)** | P0 (Neon Auth verify), Resend domain verification, Fly secrets | — |
| **Merge point A** | P6 starts after Lane 1 P3 + Lane 2 P5 + Lane 3 (Neon Auth ready) | |
| **Merge point B** | P7 after P6 ; P8 after P7 ; P9 last | |

Conflict flags: none — Lane 1 (backend repo) and Lane 2 (frontend repo) are physically separate.

---

## Follow-up TODOs (not in this PR; add to TODOS.md)

- [ ] Hard-delete reaper for soft-deleted customers (cron)
- [ ] Audit log retention sweep + email PII redaction (90-day window)
- [ ] Auto-retry sweep for invitations with `notified_at IS NULL AND created_at < NOW() - 5min`
- [ ] Multi-team UI: switcher + active-team header (drop UNIQUE constraint when ready)
- [ ] Resend bounce/complaint webhook full handling
- [ ] Super-admin role for break-glass tenant ops

---

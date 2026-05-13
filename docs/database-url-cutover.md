# DATABASE_URL cutover: `probe` → `probe_app`

Operational runbook for switching prbe-knowledge's runtime Postgres role
from the `probe` superuser to the non-privileged `probe_app` role. Once
in place, FORCE RLS enforces tenant isolation in the database itself
instead of relying solely on `with_tenant()` discipline in application
code.

This is the final step of bug #46 (the Phase 4 RLS hardening track):
the schema/policies/`with_tenant` audit landed in PRs #248, #249, #250,
#252, #253, #254. The connection-role switch is the only remaining
change, and it is an **operator action** — no code deploy is needed.

## TL;DR

```bash
kubectl -n customer-<uuid> patch secret prbe-data-plane-secrets \
  --type=json \
  -p='[{"op":"replace","path":"/data/DATABASE_URL","value":"<base64 of probe_app DSN>"}]'

kubectl -n customer-<uuid> rollout restart \
  deploy/prbe-knowledge-ingestion \
  deploy/prbe-knowledge-retrieval \
  deploy/prbe-knowledge-worker \
  deploy/prbe-knowledge-side-worker \
  deploy/prbe-knowledge-synthesis \
  deploy/prbe-knowledge-poller
```

Then verify (Step 4 below).

## Pre-flight: confirm `probe_app` is correctly provisioned

The `probe_app` role is created by `prbe-postgres`'s bootstrap migrations.
Before the cutover, confirm it has exactly the right grants — no more,
no less.

```sql
-- 1. The role exists and can log in.
SELECT rolname, rolcanlogin, rolsuper, rolbypassrls
FROM pg_roles
WHERE rolname = 'probe_app';
-- Expect: rolcanlogin=t, rolsuper=f, rolbypassrls=f.

-- 2. Table grants: SELECT/INSERT/UPDATE/DELETE on every app table,
--    NO TRUNCATE, NO REFERENCES, NO TRIGGER.
SELECT table_schema, table_name, privilege_type
FROM information_schema.table_privileges
WHERE grantee = 'probe_app' AND table_schema = 'ag_catalog'
ORDER BY table_name, privilege_type;

-- 3. Default search_path includes ag_catalog (so the on_connect hook is
--    defence-in-depth, not load-bearing).
SELECT rolname, rolconfig FROM pg_roles WHERE rolname = 'probe_app';
-- Expect rolconfig to contain
--   search_path=ag_catalog, public, "$user"

-- 4. Sequence USAGE (so SERIAL/IDENTITY columns work).
SELECT sequence_schema, sequence_name
FROM information_schema.sequences
WHERE sequence_schema = 'ag_catalog'
EXCEPT
SELECT 'ag_catalog', c.relname
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'S'
  AND has_sequence_privilege('probe_app', c.oid, 'USAGE');
-- Expect: empty result (every sequence is grantable to probe_app).
```

If any check fails, fix the role at the Postgres-image layer
(`prbe-ai/prbe-postgres`) before continuing. Do NOT hand-grant in the
running database; the next image rebuild would silently un-do it.

## Step 1: dry-run on one tenant

Pick the least-busy customer namespace (today: `probe-founders`) and
test the switch there first.

```bash
NS=customer-<probe-founders-uuid>

# Capture the current DSN so we can roll back in seconds.
kubectl -n "$NS" get secret prbe-data-plane-secrets \
  -o jsonpath='{.data.DATABASE_URL}' | base64 -d > /tmp/pre-cutover-dsn

# Compose the probe_app DSN. Same host/port/database, swap the user
# and password. Production passwords live in 1Password under
# "Neon — probe_app role".
PROBE_APP_DSN='postgresql://probe_app:<password>@<neon-host>/<db>?sslmode=require'
```

## Step 2: patch the secret

```bash
kubectl -n "$NS" patch secret prbe-data-plane-secrets \
  --type=json \
  -p="[{\"op\":\"replace\",\"path\":\"/data/DATABASE_URL\",\"value\":\"$(echo -n "$PROBE_APP_DSN" | base64)\"}]"
```

Patching the secret does NOT roll the pods. The next step does.

## Step 3: rollout-restart every deployment that holds a pool

```bash
kubectl -n "$NS" rollout restart \
  deploy/prbe-knowledge-ingestion \
  deploy/prbe-knowledge-retrieval \
  deploy/prbe-knowledge-worker \
  deploy/prbe-knowledge-side-worker \
  deploy/prbe-knowledge-synthesis \
  deploy/prbe-knowledge-poller
```

Cron jobs (`fly.cron.toml` / `CronJob` resources) pick up the new DSN
on their next scheduled run; no restart needed.

## Step 4: verify

```bash
# (a) Boot log on each deployment should now show:
#     db.role role=probe_app is_superuser=False
# and NOT
#     db.superuser_in_managed_env ...
kubectl -n "$NS" logs deploy/prbe-knowledge-ingestion --tail=200 \
  > /tmp/post-cutover.log 2>&1
grep -E 'db\.(role|superuser_in_managed_env)' /tmp/post-cutover.log

# (b) Sanity-check live Postgres-side:
kubectl -n "$NS" exec deploy/prbe-knowledge-ingestion -- \
  psql "$DATABASE_URL" -c \
  "SELECT usename, count(*) FROM pg_stat_activity \
   WHERE application_name LIKE 'prbe%' GROUP BY usename;"
# Expect: usename = probe_app, NOT probe.
```

## Step 5: smoke checks

Exercise the three durable surfaces in this order:

1. **Health** — `curl https://<slug>.prbe.ai/healthz` returns `{"ok": true}`.
2. **Ingest** — POST a small Custom Ingest payload (the bearer-token
   endpoint exercises `verify_and_touch_custom_ingest_token`,
   `with_tenant`, and a graph_nodes/edges write):
   ```bash
   curl -sS -X POST "https://<slug>.prbe.ai/internal/ingest" \
     -H "X-Internal-Knowledge-Key: $INTERNAL_KEY" \
     -H "X-Prbe-Customer: <customer-id>" \
     -H "Content-Type: application/json" \
     -d @fixtures/custom-ingest-smoke.json
   ```
3. **Retrieve** — `POST /retrieve` with a one-word query, expect 200 +
   a non-empty `results` array for any seeded tenant.

If any step fails, see Rollback below.

## Step 6: roll the rest of the fleet

Once tenant 1 has been stable for ≥30 minutes, repeat Steps 2-5 for
every remaining customer namespace. Steps 2-3 can be wrapped in a small
for-loop.

## Rollback

The bug-#46 audit landed `with_tenant()` everywhere; the cutover is
DSN-only, so rolling back is symmetric.

```bash
# Restore the previous DSN.
kubectl -n "$NS" patch secret prbe-data-plane-secrets \
  --type=json \
  -p="[{\"op\":\"replace\",\"path\":\"/data/DATABASE_URL\",\"value\":\"$(cat /tmp/pre-cutover-dsn | base64)\"}]"

# Roll the deployments back to the prior DSN.
kubectl -n "$NS" rollout restart \
  deploy/prbe-knowledge-ingestion \
  deploy/prbe-knowledge-retrieval \
  deploy/prbe-knowledge-worker \
  deploy/prbe-knowledge-side-worker \
  deploy/prbe-knowledge-synthesis \
  deploy/prbe-knowledge-poller
```

The schema, RLS policies, `with_tenant` wraps, and `WITH CHECK` clauses
are all backward-compatible: the `probe` superuser was already
implicitly bypassing RLS, so rolling back simply restores the
pre-cutover behaviour. No data migration is involved.

## Why the local-dev default doesn't change

`shared/config.py`'s `database_url` default
(`postgresql://prbe:prbe@localhost:...`) deliberately stays on `prbe`
for local Docker Compose, where `prbe` is the only role that exists
and migration discipline (alembic, db init) needs superuser. The boot
log only WARNs when `environment != "local"`.

## Post-cutover: re-running the RLS denial test

`tests/test_rls_cross_tenant_denial.py` skips itself when the
connection is running as superuser (today's local-CI default). Once
the cutover is in place and a CI job exists that runs as `probe_app`,
the skip lifts and the cross-tenant-isolation assertions become live —
the operator-promotion of the "audit says we're fine" story.

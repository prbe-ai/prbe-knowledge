#!/usr/bin/env bash
# Plan A live smoke for the post-approval knowledge flow.
#
# Exercises the writeback + review + visibility surfaces end-to-end
# against a live Docker Postgres + uvicorn (ingestion + retrieval),
# with a mock orchestrator on a local port to receive post-approval
# HTTP dispatches.
#
# Per `feedback_container_smoke_tests`: pytest alone does not cover
# prbe-knowledge features; this script is the "done" gate.
#
# Required on host: docker, jq, psql (PGPASSWORD-friendly).
# Required in repo: .venv/bin/{uvicorn,python,alembic}.
#
# Usage:
#   ./scripts/smoke_post_approval_knowledge.sh
#
# Env overrides (defaults work against docker-compose defaults):
#   KEY            = X-Internal-Knowledge-Key value
#   INGEST_PORT    = uvicorn ingestion port (default 8090)
#   RETRIEVAL_PORT = uvicorn retrieval port (default 8091)
#   ORCH_PORT      = mock orchestrator port (default 9099)
#   CUSTOMER       = customer_id slug for the smoke run
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

KEY="${KEY:-smoke-internal-key}"
BACKEND_KEY="${BACKEND_KEY:-smoke-backend-key}"
INGEST_PORT="${INGEST_PORT:-8090}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8091}"
ORCH_PORT="${ORCH_PORT:-9099}"
CUSTOMER="${CUSTOMER:-smoke-postapproval-cust}"
INGEST_BASE="http://127.0.0.1:${INGEST_PORT}"
RETRIEVAL_BASE="http://127.0.0.1:${RETRIEVAL_PORT}"
ORCH_BASE="http://127.0.0.1:${ORCH_PORT}"

PSQL_DSN="postgresql://prbe:prbe@localhost:5432/prbe_knowledge"
export PGPASSWORD=prbe

# Python DSN for direct on_resolution_event invocations from the script.
# shared.config defaults to this exact URL but exporting it explicitly
# decouples the smoke from whatever ambient DATABASE_URL might be set.
PY_DATABASE_URL="postgresql://prbe:prbe@localhost:5432/prbe_knowledge"

LOG_DIR="/tmp/smoke_post_approval_${CUSTOMER}"
mkdir -p "$LOG_DIR"

INGEST_LOG="$LOG_DIR/ingest.log"
RETRIEVAL_LOG="$LOG_DIR/retrieval.log"
ORCH_LOG="$LOG_DIR/orch.log"

INGEST_PID=""
RETRIEVAL_PID=""
ORCH_PID=""

cleanup() {
    local ec=$?
    set +e
    if [[ -n "${INGEST_PID:-}" ]]; then kill "$INGEST_PID" 2>/dev/null; fi
    if [[ -n "${RETRIEVAL_PID:-}" ]]; then kill "$RETRIEVAL_PID" 2>/dev/null; fi
    if [[ -n "${ORCH_PID:-}" ]]; then kill "$ORCH_PID" 2>/dev/null; fi
    if (( ec != 0 )); then
        echo
        echo "--- FAILED. Tailing logs: ---"
        for f in "$INGEST_LOG" "$RETRIEVAL_LOG" "$ORCH_LOG"; do
            if [[ -f "$f" ]]; then
                echo "==> $f <=="
                tail -50 "$f"
            fi
        done
    fi
    exit $ec
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
psql_q() {
    # Quiet, tab-separated; one-line scalar reads.
    psql "$PSQL_DSN" -At -F $'\t' -c "$1"
}

assert_eq() {
    local got="$1" want="$2" label="$3"
    if [[ "$got" != "$want" ]]; then
        echo "ASSERT FAIL [$label]: got='$got' want='$want'" >&2
        return 1
    fi
    echo "ASSERT OK   [$label]: $got"
}

wait_for_health() {
    local url="$1" label="$2"
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "$label up after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "$label NOT UP after 12s — see logs in $LOG_DIR" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Step 0: Pre-flight + reset smoke customer state
# ---------------------------------------------------------------------------
echo "=== STEP 0: Pre-flight ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "CUSTOMER=$CUSTOMER  KEY=$KEY"
echo "Ports: ingest=$INGEST_PORT retrieval=$RETRIEVAL_PORT orch=$ORCH_PORT"
echo "Logs:  $LOG_DIR/"

# Sanity: Docker Postgres + alembic head.
if ! psql_q "SELECT 1" >/dev/null 2>&1; then
    echo "Cannot reach Postgres at $PSQL_DSN — is docker-compose up?" >&2
    exit 1
fi
HEAD=$(psql_q "SELECT version_num FROM alembic_version" || echo "")
echo "alembic head: $HEAD"
if [[ "$HEAD" != "0086_inv_metadata" ]]; then
    echo "Migrations not at expected head 0086_inv_metadata — run scripts/neon-migrate.sh local" >&2
    exit 1
fi

# Idempotent customer + tabula-rasa of any prior smoke state.
# ON DELETE CASCADE on customers handles everything that FKs back, but
# wiki_review_queue + incident_investigations cascade through customer_id
# so a single DELETE is enough.
psql "$PSQL_DSN" -v ON_ERROR_STOP=1 -c "
    DELETE FROM customers WHERE customer_id = '$CUSTOMER';
    INSERT INTO customers (customer_id, display_name, api_key_hash)
    VALUES ('$CUSTOMER', 'smoke post-approval', 'smoke-hash');
" >/dev/null

# ---------------------------------------------------------------------------
# Step 0a: Mock orchestrator (accepts POST /internal/post-approval-actions)
# ---------------------------------------------------------------------------
echo "=== STEP 0a: Start mock orchestrator on $ORCH_PORT ==="
python3 - <<PY_EOF > "$ORCH_LOG" 2>&1 &
import http.server
import json
import sys
import time


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        sys.stdout.write(
            f"{time.strftime('%H:%M:%S')} POST {self.path} "
            f"len={n} body={body[:300]!r}\n"
        )
        sys.stdout.flush()
        self.send_response(202)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args, **kwargs):
        pass


srv = http.server.HTTPServer(("127.0.0.1", ${ORCH_PORT}), Handler)
print("mock orchestrator listening on ${ORCH_PORT}", flush=True)
srv.serve_forever()
PY_EOF
ORCH_PID=$!
# Give the mock a beat to bind the socket.
for i in 1 2 3 4 5; do
    if curl -fsS -X POST "$ORCH_BASE/ping" -d '{}' -o /dev/null 2>&1; then
        echo "mock orchestrator up after ${i}s"
        break
    fi
    sleep 1
done
if ! kill -0 "$ORCH_PID" 2>/dev/null; then
    echo "mock orchestrator failed to start" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Step 0b: Launch ingestion + retrieval uvicorn services
# ---------------------------------------------------------------------------
echo "=== STEP 0b: Launch uvicorn services ==="

# Common env. SECRET_VALUE pulls through Pydantic SecretStr at import.
export DATABASE_URL="$PY_DATABASE_URL"
export INTERNAL_KNOWLEDGE_API_KEY="$KEY"
export ORCHESTRATOR_BASE_URL="$ORCH_BASE"
export INTERNAL_BACKEND_API_KEY="$BACKEND_KEY"

.venv/bin/uvicorn services.ingestion.main:app \
    --host 127.0.0.1 --port "$INGEST_PORT" \
    > "$INGEST_LOG" 2>&1 &
INGEST_PID=$!

.venv/bin/uvicorn services.retrieval.main:app \
    --host 127.0.0.1 --port "$RETRIEVAL_PORT" \
    > "$RETRIEVAL_LOG" 2>&1 &
RETRIEVAL_PID=$!

wait_for_health "$INGEST_BASE/health" "ingestion"
wait_for_health "$RETRIEVAL_BASE/health" "retrieval"

# ---------------------------------------------------------------------------
# Headers shared across requests
# ---------------------------------------------------------------------------
H_KEY=(-H "x-internal-knowledge-key: $KEY")
H_JSON=(-H "content-type: application/json")
H_RETR=(-H "x-internal-knowledge-key: $KEY" -H "x-prbe-customer: $CUSTOMER")

INC1="pd:incident:SMK-001"
INC2="pd:incident:SMK-002"

# ---------------------------------------------------------------------------
# Step 1: Seed investigation (Plan 4 writeback)
# ---------------------------------------------------------------------------
echo "=== STEP 1: POST /api/incident-investigations (seed v1) ==="
INV_BODY=$(cat <<JSON
{
  "customer_id": "$CUSTOMER",
  "incident_doc_id": "$INC1",
  "source_system": "pagerduty",
  "source_event_id": "$INC1:incident.triggered",
  "version": 1, "mode": "full",
  "title": "Smoke investigation",
  "body_markdown": "## hypothesis\n\nrouter regression caused 502s.\n",
  "evidence": [],
  "narrative": "agent narrative",
  "tool_trace_run_id": "smoke-run"
}
JSON
)
curl -fsS -X POST "$INGEST_BASE/api/incident-investigations" \
    "${H_KEY[@]}" "${H_JSON[@]}" -d "$INV_BODY" | jq -e '.state == "pending_review"' >/dev/null
echo "investigation writeback OK"

# ---------------------------------------------------------------------------
# Step 2: Approve the investigation
# ---------------------------------------------------------------------------
echo "=== STEP 2: POST /api/incident-investigations/{id}/approve ==="
curl -fsS -X POST \
    "$INGEST_BASE/api/incident-investigations/$INC1/approve?customer_id=$CUSTOMER" \
    "${H_KEY[@]}" "${H_JSON[@]}" \
    -d '{"reviewer_id":"smoke-user"}' \
    | jq -e '.state == "approved"' >/dev/null
echo "approve OK"

# After approve, on_approval ran but on_resolution hasn't — guard NULL.
got=$(psql_q "
    SELECT (approved_at IS NOT NULL)::text || '|'
           || (resolved_at IS NULL)::text || '|'
           || (post_approval_dispatched_at IS NULL)::text
    FROM incident_investigations
    WHERE customer_id='$CUSTOMER' AND incident_doc_id='$INC1';")
assert_eq "$got" "true|true|true" "post-approve guards"

# ---------------------------------------------------------------------------
# Step 3: Simulate resolution arrival -> dispatch fires
# ---------------------------------------------------------------------------
echo "=== STEP 3: on_resolution_event -> dispatch ==="
.venv/bin/python - <<PY_EOF
import asyncio, os
os.environ["DATABASE_URL"] = "$PY_DATABASE_URL"
os.environ["INTERNAL_KNOWLEDGE_API_KEY"] = "$KEY"
os.environ["ORCHESTRATOR_BASE_URL"] = "$ORCH_BASE"
os.environ["INTERNAL_BACKEND_API_KEY"] = "$BACKEND_KEY"

from shared.db import init_pool, close_pool
from services.post_approval.dispatch import on_resolution_event


async def main():
    await init_pool()
    try:
        await on_resolution_event("$CUSTOMER", "$INC1")
    finally:
        await close_pool()


asyncio.run(main())
PY_EOF

got=$(psql_q "
    SELECT (resolved_at IS NOT NULL)::text || '|'
           || (post_approval_dispatched_at IS NOT NULL)::text || '|'
           || COALESCE((metadata->>'post_approval_dispatch_failed')::text,'<null>')
    FROM incident_investigations
    WHERE customer_id='$CUSTOMER' AND incident_doc_id='$INC1';")
assert_eq "$got" "true|true|<null>" "dispatch fired (orchestrator received POST)"
# Mock orchestrator log MUST show at least one POST.
if ! grep -q "POST /internal/post-approval-actions" "$ORCH_LOG"; then
    echo "ASSERT FAIL: mock orchestrator did not record post-approval POST" >&2
    cat "$ORCH_LOG" >&2
    exit 1
fi
echo "ASSERT OK   [orchestrator received POST /internal/post-approval-actions]"

# ---------------------------------------------------------------------------
# Step 4: POST an evidence pack
# ---------------------------------------------------------------------------
echo "=== STEP 4: POST /api/incident-evidence-packs ==="
EP_BODY=$(cat <<JSON
{
  "customer_id": "$CUSTOMER",
  "incident_doc_id": "$INC1",
  "evidence_pack": {
    "timeline_events": [],
    "resolution_actions": [],
    "recovery_signals": [],
    "post_resolution_discussion": [],
    "related_doc_ids": [],
    "deploys_in_window": [],
    "similar_past_incidents": [],
    "free_form_findings": "smoke run findings",
    "mode": "full"
  }
}
JSON
)
curl -fsS -X POST "$INGEST_BASE/api/incident-evidence-packs" \
    "${H_KEY[@]}" "${H_JSON[@]}" -d "$EP_BODY" \
    | jq -e ".incident_doc_id == \"$INC1\" and .duplicate == false" >/dev/null
echo "evidence pack writeback OK"

got=$(psql_q "
    SELECT (evidence_pack IS NOT NULL)::text || '|' || (evidence_pack->>'mode')
    FROM incident_investigations
    WHERE customer_id='$CUSTOMER' AND incident_doc_id='$INC1';")
assert_eq "$got" "true|full" "evidence_pack persisted"

# GET it back
curl -fsS "$INGEST_BASE/api/incident-evidence-packs?customer_id=$CUSTOMER&incident_doc_id=$INC1" \
    "${H_KEY[@]}" \
    | jq -e '.mode == "full" and .free_form_findings == "smoke run findings"' >/dev/null
echo "evidence pack GET round-trip OK"

# ---------------------------------------------------------------------------
# Step 5: POST a wiki postmortem artifact (lands as draft)
# ---------------------------------------------------------------------------
echo "=== STEP 5: POST /api/wiki-artifacts (postmortem, full) ==="
WIKI_BODY=$(cat <<JSON
{
  "customer_id": "$CUSTOMER",
  "incident_doc_id": "$INC1",
  "investigation_doc_id": "pd:investigation:SMK-001:v1",
  "artifact_kind": "postmortem",
  "title": "Smoke postmortem",
  "body_markdown": "# postmortem\n\n## Summary\nrouter regression; reverted.\n\n## Timeline\n- 09:00 first 502s\n- 09:08 rollback merged\n",
  "metadata": {"mode": "full", "tool_trace_run_id": "smoke-pm"}
}
JSON
)
ART_RESP=$(curl -fsS -X POST "$INGEST_BASE/api/wiki-artifacts" \
    "${H_KEY[@]}" "${H_JSON[@]}" -d "$WIKI_BODY")
echo "$ART_RESP" | jq .
ART_ID=$(echo "$ART_RESP" | jq -r '.artifact_doc_id')
ART_STATE=$(echo "$ART_RESP" | jq -r '.state')
ART_DUP=$(echo "$ART_RESP" | jq -r '.duplicate')
assert_eq "$ART_STATE" "pending_review" "artifact landed pending_review"
assert_eq "$ART_DUP" "false" "artifact not duplicate"

# Document is draft + chunks are draft.
got=$(psql_q "
    SELECT visibility FROM documents
    WHERE customer_id='$CUSTOMER' AND doc_id='$ART_ID' AND valid_to IS NULL;")
assert_eq "$got" "draft" "document visibility=draft"

draft_chunks=$(psql_q "
    SELECT COUNT(*) FROM chunks
    WHERE customer_id='$CUSTOMER' AND doc_id='$ART_ID' AND visibility='draft';")
if [[ "$draft_chunks" == "0" ]]; then
    echo "ASSERT FAIL: expected >=1 draft chunk, got 0" >&2
    exit 1
fi
echo "ASSERT OK   [draft chunks created]: count=$draft_chunks"

# ---------------------------------------------------------------------------
# Step 6: /sources/{doc_id} 404s for draft (default approved-only filter)
# ---------------------------------------------------------------------------
echo "=== STEP 6: GET /sources/{doc_id} should 404 for draft ==="
http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    "$RETRIEVAL_BASE/sources/$ART_ID" "${H_RETR[@]}")
assert_eq "$http_code" "404" "draft hidden from /sources"

# ---------------------------------------------------------------------------
# Step 7: Approve artifact -> visibility flips to approved on doc + chunks
# ---------------------------------------------------------------------------
echo "=== STEP 7: POST /api/wiki-artifacts/{id}/approve ==="
curl -fsS -X POST \
    "$INGEST_BASE/api/wiki-artifacts/$ART_ID/approve?customer_id=$CUSTOMER" \
    "${H_KEY[@]}" "${H_JSON[@]}" \
    -d '{"reviewer_id":"smoke-user"}' \
    | jq -e '.state == "approved"' >/dev/null
echo "approve OK"

got=$(psql_q "
    SELECT visibility FROM documents
    WHERE customer_id='$CUSTOMER' AND doc_id='$ART_ID' AND valid_to IS NULL;")
assert_eq "$got" "approved" "document visibility flipped to approved"

remaining_draft=$(psql_q "
    SELECT COUNT(*) FROM chunks
    WHERE customer_id='$CUSTOMER' AND doc_id='$ART_ID' AND visibility='draft';")
assert_eq "$remaining_draft" "0" "no draft chunks remain"

approved_chunks=$(psql_q "
    SELECT COUNT(*) FROM chunks
    WHERE customer_id='$CUSTOMER' AND doc_id='$ART_ID' AND visibility='approved';")
if [[ "$approved_chunks" == "0" ]]; then
    echo "ASSERT FAIL: expected >=1 approved chunk, got 0" >&2
    exit 1
fi
echo "ASSERT OK   [approved chunks materialized]: count=$approved_chunks"

# ---------------------------------------------------------------------------
# Step 8: /sources/{doc_id} now returns the approved artifact
# ---------------------------------------------------------------------------
echo "=== STEP 8: GET /sources/{doc_id} returns approved artifact ==="
curl -fsS "$RETRIEVAL_BASE/sources/$ART_ID" "${H_RETR[@]}" \
    | jq -e ".doc_id == \"$ART_ID\"" >/dev/null
echo "ASSERT OK   [/sources returns the approved doc]"

# ---------------------------------------------------------------------------
# Step 9: Reject path — second artifact, reject, durable state flip
# ---------------------------------------------------------------------------
echo "=== STEP 9: Reject path (orchestrator reachable, but we still verify durability) ==="
WIKI2_BODY=$(cat <<JSON
{
  "customer_id": "$CUSTOMER",
  "incident_doc_id": "$INC1",
  "investigation_doc_id": "pd:investigation:SMK-001:v1",
  "artifact_kind": "knowledge_page",
  "title": "Smoke knowledge page",
  "body_markdown": "## kp\n\nDraft body for reject path.\n",
  "metadata": {"mode": "full", "tool_trace_run_id": "smoke-kp"}
}
JSON
)
ART2_RESP=$(curl -fsS -X POST "$INGEST_BASE/api/wiki-artifacts" \
    "${H_KEY[@]}" "${H_JSON[@]}" -d "$WIKI2_BODY")
ART2=$(echo "$ART2_RESP" | jq -r '.artifact_doc_id')
echo "second artifact: $ART2"

# To prove the *durability under failure* invariant explicitly (the
# script's main contribution beyond unit tests), kill the mock
# orchestrator BEFORE the reject so the re-dispatch fails.
echo "Killing mock orchestrator to force re-dispatch failure..."
kill "$ORCH_PID" 2>/dev/null || true
wait "$ORCH_PID" 2>/dev/null || true
ORCH_PID=""

curl -fsS -X POST \
    "$INGEST_BASE/api/wiki-artifacts/$ART2/reject?customer_id=$CUSTOMER" \
    "${H_KEY[@]}" "${H_JSON[@]}" \
    -d '{"reviewer_id":"smoke-user","feedback":"needs more detail"}' \
    | jq -e '.state == "rejected"' >/dev/null
echo "reject OK"

# State must be rejected EVEN though orchestrator is down — that's the
# durability invariant. metadata.re_dispatch_failed must be true.
got=$(psql_q "
    SELECT state || '|'
           || COALESCE((metadata->>'re_dispatch_failed')::text,'<null>')
    FROM wiki_review_queue
    WHERE customer_id='$CUSTOMER' AND artifact_doc_id='$ART2';")
assert_eq "$got" "rejected|true" "reject state durable + re_dispatch_failed flag set"

# ---------------------------------------------------------------------------
# Step 10: Resolution-first ordering smoke (resolution before any investigation)
# ---------------------------------------------------------------------------
echo "=== STEP 10: Resolution-first ordering ==="

# Bring the mock orchestrator back up — Plan A's resolution-first path
# still needs the dispatch to succeed at the end.
python3 - <<PY_EOF > "$ORCH_LOG" 2>&1 &
import http.server, json, sys, time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        sys.stdout.write(f"{time.strftime('%H:%M:%S')} POST {self.path} len={n} body={body[:300]!r}\n")
        sys.stdout.flush()
        self.send_response(202); self.send_header("content-type","application/json"); self.end_headers()
        self.wfile.write(b'{"ok":true}')
    def log_message(self, *a, **k): pass

http.server.HTTPServer(("127.0.0.1", ${ORCH_PORT}), Handler).serve_forever()
PY_EOF
ORCH_PID=$!
for i in 1 2 3 4 5; do
    if curl -fsS -X POST "$ORCH_BASE/ping" -d '{}' -o /dev/null 2>&1; then break; fi
    sleep 1
done

# Resolution arrives FIRST — no investigation row exists yet.
.venv/bin/python - <<PY_EOF
import asyncio, os
os.environ["DATABASE_URL"] = "$PY_DATABASE_URL"
os.environ["INTERNAL_KNOWLEDGE_API_KEY"] = "$KEY"
os.environ["ORCHESTRATOR_BASE_URL"] = "$ORCH_BASE"
os.environ["INTERNAL_BACKEND_API_KEY"] = "$BACKEND_KEY"

from shared.db import init_pool, close_pool
from services.post_approval.dispatch import on_resolution_event


async def main():
    await init_pool()
    try:
        await on_resolution_event("$CUSTOMER", "$INC2")
    finally:
        await close_pool()


asyncio.run(main())
PY_EOF

# Partial row: resolved_at set, approved_at NULL, dispatch NULL.
got=$(psql_q "
    SELECT (resolved_at IS NOT NULL)::text || '|'
           || (approved_at IS NULL)::text || '|'
           || (post_approval_dispatched_at IS NULL)::text
    FROM incident_investigations
    WHERE customer_id='$CUSTOMER' AND incident_doc_id='$INC2';")
assert_eq "$got" "true|true|true" "resolution-first creates partial row"

# Now seed + approve the investigation; dispatch should fire on the
# second timestamp landing.
INV2_BODY=$(cat <<JSON
{
  "customer_id": "$CUSTOMER",
  "incident_doc_id": "$INC2",
  "source_system": "pagerduty",
  "source_event_id": "$INC2:incident.triggered",
  "version": 1, "mode": "full",
  "title": "SMK-002 investigation",
  "body_markdown": "## hypothesis\n\nthird-party outage.\n",
  "evidence": [],
  "narrative": "x",
  "tool_trace_run_id": "smoke-2"
}
JSON
)
curl -fsS -X POST "$INGEST_BASE/api/incident-investigations" \
    "${H_KEY[@]}" "${H_JSON[@]}" -d "$INV2_BODY" \
    | jq -e '.state == "pending_review"' >/dev/null

curl -fsS -X POST \
    "$INGEST_BASE/api/incident-investigations/$INC2/approve?customer_id=$CUSTOMER" \
    "${H_KEY[@]}" "${H_JSON[@]}" \
    -d '{"reviewer_id":"smoke-user"}' \
    | jq -e '.state == "approved"' >/dev/null

# Dispatch should now have fired (approved_at set is the second timestamp).
got=$(psql_q "
    SELECT (approved_at IS NOT NULL)::text || '|'
           || (resolved_at IS NOT NULL)::text || '|'
           || (post_approval_dispatched_at IS NOT NULL)::text || '|'
           || COALESCE((metadata->>'post_approval_dispatch_failed')::text,'<null>')
    FROM incident_investigations
    WHERE customer_id='$CUSTOMER' AND incident_doc_id='$INC2';")
assert_eq "$got" "true|true|true|<null>" "resolution-first: dispatch fires on approve"

# Two POSTs total to the mock orchestrator across the whole run
# (INC1 step 3 + INC2 step 10). Orchestrator log was rotated when we
# restarted the mock; the current log should have >=1 POST for INC2.
if ! grep -q "POST /internal/post-approval-actions" "$ORCH_LOG"; then
    echo "ASSERT FAIL: orchestrator did not receive INC2 dispatch" >&2
    cat "$ORCH_LOG" >&2
    exit 1
fi
echo "ASSERT OK   [orchestrator received INC2 dispatch]"

# ---------------------------------------------------------------------------
# Step 11: Postmortem template routes round-trip
# ---------------------------------------------------------------------------
echo "=== STEP 11: customer postmortem template routes ==="

# Default (no override) — GET effective falls through to the bundled
# default in shared/templates/postmortem.py.
curl -fsS "$INGEST_BASE/api/customer-postmortem-templates/$CUSTOMER/effective" \
    "${H_KEY[@]}" \
    | jq -e '(.body_markdown | length) > 0 and .source == "default"' \
        >/dev/null
echo "GET effective template OK (falls through to default)"

# PUT an inline override and read it back through both GET routes.
PUT_BODY=$(cat <<JSON
{
  "customer_id": "$CUSTOMER",
  "mode": "inline",
  "body_markdown": "# Custom Postmortem\n\n## TL;DR\n{{summary}}\n"
}
JSON
)
curl -fsS -X PUT "$INGEST_BASE/api/customer-postmortem-templates/$CUSTOMER" \
    "${H_KEY[@]}" "${H_JSON[@]}" \
    -d "$PUT_BODY" \
    | jq -e '.customer_id == "'"$CUSTOMER"'" and .mode == "inline"' >/dev/null

curl -fsS "$INGEST_BASE/api/customer-postmortem-templates/$CUSTOMER" \
    "${H_KEY[@]}" \
    | jq -e '.body_markdown | test("Custom Postmortem")' >/dev/null

curl -fsS "$INGEST_BASE/api/customer-postmortem-templates/$CUSTOMER/effective" \
    "${H_KEY[@]}" \
    | jq -e '.source == "inline_override" and (.body_markdown | test("Custom Postmortem"))' \
        >/dev/null
echo "PUT + GET override + effective resolution OK"

# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------
echo
echo "=== ALL SMOKE CHECKS PASSED ==="

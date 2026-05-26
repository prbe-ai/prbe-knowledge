#!/usr/bin/env bash
# Black-box working check (ProductRuntimeSpec.smoke_test). The coding agent sees
# pass/fail ONLY. It asserts the service is UP and /retrieve is SHAPE-valid — never
# anything about retrieval QUALITY. The quality metric is the held-out eval, injected
# into a separate fresh grading sandbox; leaking a quality signal here would let the
# agent reward-hack the smoke gate.
#
# Hermetic by design: passes on an EMPTY corpus with NO LLM gateway. With GOOGLE_API_KEY
# unset, query embedding falls back to the local deterministic hash vector
# (shared/embeddings.py:570); with no gateway, the gatherer short-circuits to an empty
# (but schema-valid) RetrieveResponse (agent/loop.py:1400). Either way: 200 + a list.
set -euo pipefail

BASE="http://localhost:8081"
KEY="${INTERNAL_KNOWLEDGE_API_KEY:-sandbox-internal-key}"
TENANT="${SMOKE_TENANT:-eval-tenant}"

# 1. liveness + DB ping — main.py:117 returns 503 if the PG pool is down.
curl -fsS "$BASE/health" -o /dev/null

# 2. /retrieve answers 200 with a schema-valid RetrieveResponse on a canned query.
#    Internal-key auth: X-Internal-Knowledge-Key + X-Prbe-Customer (auth.py:60,65).
resp="$(curl -fsS -X POST "$BASE/retrieve" \
  -H 'Content-Type: application/json' \
  -H "X-Internal-Knowledge-Key: $KEY" \
  -H "X-Prbe-Customer: $TENANT" \
  -d '{"query":"smoke probe","top_k":5}')"

# 3. shape only: `results` exists and is a list. NEVER assert count/scores/ids.
echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert isinstance(d.get("results"), list), "results not a list"'
echo "[smoke] ok"

# Transcript-extraction spend: plan

Target: the pasted brief "Plan (do not yet ship) cutting transcript-extraction spend"
(2026-09-23). Reviewed with `/plan-eng-review` on branch `extraction-spend-plan`
(prbe-knowledge, worktree off `origin/main` at `2b540de`).

Status: **PR 1 implemented** on this branch (2026-09-23): §3.1-§3.4, the pass log and
tool-declined warning from §3.5, §3.6 tests and the CI list. Differences from the text below,
found while building it: the backfill proves "the last pass mined the current transcript" from
the queue row itself (the last upsert was not a batch: `enqueued_at`'s UTC day is later than the
newest batch key's date, and `completed_at >= enqueued_at`), plus the marker object in R2; it has
no idle filter (the old loop re-touched every row within a day) and takes `--customer` for a
canary. The empty-row guard in the handler applies only to complete rows with no events and no
identity; a live batch without `employee_id` still raises. PR 2 (§3.8, v2 sweep, metrics) follows.

Original status: **plan + engineering review**. Phase 0 is the
deliverable that pays; Phases 1 and 2 are kept as build-ready specs behind a
measurement gate.

---

## 0. The answer in one screen

The bill is not resumed sessions and not degraded passes. It is a loop between the
hourly session-completer cron and the worker:

```
kb/session_completer.py:76-88   "already finalized" == a finalize.marker key is
                                 STILL in payload_s3_keys
engine/ingest/normalizer.py:250 the worker DELETES that key after every
                                 authoritative pass (via claude_code.py:736-738)
```

So every protocol-1 session that has been idle for 24 h (`--idle-minutes 1440`, cron
`0 * * * *`) is re-finalized and fully re-mined **once a day, forever**. The daily
"new sessions" count never mattered because the population being mined is the whole
idle v1 backlog.

| measured 2026-09-23 (research plane) | value |
|---|---|
| gemini-3.7-flash, 30 d | 192,552 calls, $2,843.99, one api key (engine worker) |
| last 24 h | 6,548 calls, $105.77 |
| calls in minutes 00-09 of the hour | 5,165 / 6,548 = **79 %** |
| queue rows completed in 24 h (claude_code+codex+pi) | 5,045 |
| …of which protocol-1 rows first enqueued > 1 day ago | **4,975 (98.6 %)**, median `version` 32 |
| …protocol-2 rows | 70 |
| completer job logs 05:02 / 06:00 / 07:04 UTC | enqueued 356 / 74 / 819 |
| `normalizer.consumed_payload_keys` in 12 h | 2,599 (77 % in minutes 00-09) |
| consume code landed | 2026-08-15, #486 (median version 32 ≈ one pass per day since) |
| documents rewritten by those runs, 12 h | 20,873 added, 28,122 removed, 10,606 reused |
| unit docs written in 24 h: v1 parents vs v2 parents | 14,889 (3,944 sessions) vs 3,312 (51 sessions) |
| `claude_code_extraction.segments_capped` / `segment_failed` in 12 h | 0 / 0 (structlog WARNING renders; 7 warnings in the window, none extraction) |
| managed plane | `managed-worker` (ns `managed`) runs the same handler, but its sweep (`session-finalizer-nightly.yml`) is opt-in and every scheduled run is `skipped`: no loop there; the new completion rule still ships to it on merge |

Share of calls by path:
- **≥ 97 %**: cron re-finalize of idle v1 sessions (4,975 of 5,045 rows; the 21 % of
  calls outside minutes 00-09 are the tail of draining 800-row sweeps, not a second
  path).
- **~1-2 %**: protocol-2 client finalizes (70 rows/day). Of 5,251 v2 sessions completed
  in 30 d, 73 (1.4 %) completed more than once; 461 extra passes out of 5,712.
- **0 %**: the brief's two hypotheses. The tap never emits a `session_end` event (grep of
  research-os `agent/`: only a lease-reason string), v1 finalize keys are consumed after
  one pass, and there were 0 capped / 0 failed segments in 12 h.
- Bonus cost the loop hides: ~40 k document writes and retirements per day, i.e. daily
  re-chunk and re-embed of every unit of every idle v1 session.

What the fix is worth: after Phase 0 the mined population is "sessions that actually
ended today" (~70-175/day at ~1.3-3 calls each) → roughly **$4-8/day instead of ~$95**,
to be measured, not assumed. The per-segment cache (Phase 1) then saves at most the
8 % of post-fix passes that are re-completions: on the order of $10-20/month. That is why
Phases 1 and 2 move behind a measurement gate instead of shipping in sequence.

Outside voice (Codex, gpt-6-astra) found five real gaps in the first draft; all five are
folded in below and recorded in §7: the fix ships as two PRs (v1 cost fix first, v2 sweep
second), the backfill only touches rows with evidence of a completed pass, the v2 "already
ended" predicate no longer depends on a field the cron never sets, both mutations re-check
under the session lock, and a non-authoritative or disabled pass is retried a bounded number
of times instead of being treated as "ended".

### Two things for Richard (everything else auto-decided, recorded in §7)

1. **Stopgap today (your call, reversible, mutates prod):** suspend the research-plane
   CronJob and ~$90/day stops within the hour. Cost while suspended: v1 sessions from
   crashed clients wait for the fix to be mined; v2 is unaffected (the sweep skips it
   already). Command in §4.1.
2. **Scope change vs. your brief:** Phase 1 (cache) and Phase 2 (Jev screen) are deferred
   behind the 48 h post-fix measurement. Their specs are complete below; a logged
   per-segment content hash (§4.6) gives the cache's hypothetical hit rate for free
   before anyone builds it.

---

## 1. Mechanism, confirmed

### 1.1 What the code does today

`kb/handlers/claude_code.py:307-379` (`fetch_supplementary`) computes `session_complete`
from four signals with three different lifecycles:

| signal | where | lifecycle |
|---|---|---|
| any `session_end` event anywhere in the merged stream | :367 | sticky forever (never consumed) |
| v1 client finalize payload (`.../<session_id>.json`, `finalize: true`) | :345-348 | key consumed after an authoritative pass |
| cron `finalize.marker` key | :316-318 (+ appended a second time at :345-348) | key consumed after an authoritative pass |
| protocol 2: highest `batch_seq` payload is a finalize | :368-369, :376-379 | order-aware; keys retained |

`normalize` (:526-739) runs `extract_units_from_session` on every complete pass, over
the whole session, and consumes the finalize keys only if the bundle is authoritative
(:736-738).

`kb/session_completer.py:59-94` finds idle sessions and treats one as "already
finalized" iff a `finalize.marker` or `<session_id>.json` key is still in the array
(:86-87), skipping protocol-2 rows entirely (:89-92, :146-152). Its UPSERT (:108-122)
appends the marker, bumps `version`, resets `status='pending'`.

### 1.2 The loop

```
 day N, 07:04   cron: MAX(enqueued_at) < now-24h AND no marker key  ──► append marker,
                                                                        version+1, pending
 day N, 07:05   worker: marker present ──► complete=True ──► mine all segments
                                        ──► authoritative ──► DELETE marker key
                                        ──► session doc + N unit docs rewritten
 day N+1, 07:04 cron: MAX(enqueued_at) < now-24h AND no marker key  ──► (same row) …
```

One pass per idle v1 session per day. 4,575 claude_code + 1,064 codex + 73 pi v1 rows
exist; 4,463 of them are idle > 3 h. The worker's `consumed_payload_keys` log carries
`count: 2` on every one of 2,599 events because the marker key is appended twice to
`finalize_keys` (`claude_code.py:316-318` and :345-348).

Why nobody saw it: `tests/test_session_completer.py:246-250` still documents the marker
as "sticky"; `scripts/cron_session_completer.py:16-21` says the same; the normalizer
docstring (`normalizer.py:262-264`) explicitly accepts that "the nightly sweep" will
finalize again, written when the sweep was nightly at 360 minutes and the marker filter
still matched. The three test files that exercise this path
(`tests/test_claude_code_extraction.py`, `tests/test_session_coalescing.py`,
`tests/test_claude_code_finalize_e2e.py`) are not on the CI file list
(`.github/workflows/tests.yml:163-182`), and no test anywhere pins "the sweep does not
re-finalize a session it already mined".

### 1.3 Protocol mix (decides what matters)

| | rows | notes |
|---|---|---|
| claude_code v1 / v2 | 4,575 / 3,899 | v2 = any key under `sessions-v2/` |
| codex v1 / v2 | 1,064 / 3,791 | |
| pi v1 / v2 | 73 / 3 | |
| `session_streams` unfinalized (v2) | 70 claude_code, 25 codex; 81 older than 1 day | the sweep never reaches these |
| v1 rows still carrying a marker | 465 claude_code (288 dlq, 177 done), 55 codex, 4 pi | dlq = marker-only rows the sweep INSERTed; the handler raises "missing employee_id" |

Current clients (tap 0.7.0) speak protocol 2. Roughly half the corpus is v1 history
that will never get new batches; it is being paid for daily.

`documents.metadata.session_complete` is NOT a usable "was it mined" signal: 2,258 of
2,483 v2 session docs marked incomplete have live unit children. A completion pass that
adds no events leaves the body hash unchanged, so the SCD2 write is skipped and the
metadata never flips (`claude_code.py:874-876`, `:943`; normalizer skip-if-unchanged).

### 1.4 Deterministic session end: what exists, what is missing

Client side (research-os `agent/plugins/probe-research-tap`, 0.7.0):
- `SessionEnd` hook → shutdown sentinel → daemon enqueues a finalize through the durable
  outbox (`tap/main.py:719-727`, `_enqueue_finalize`). Deterministic on `/exit`, Ctrl-D,
  `/clear`, logout.
- Claude Code process death without the hook: the daemon's orphan check (no process holds
  the transcript open) now falls through to the same finalize path (`tap/main.py:629-650`,
  "BREAK, not return"; `tests/test_lifecycle.py:113`).
- Daemon death: the next daemon on the machine reconciles every transcript and drains every
  outbox row (`tap/reconcile.py`).
- Machine never returns: only a server-side idle sweep can end the session.

Server side:
- The idle sweep exists for v1 only. For v2 the design note at `claude_code.py:376-379`
  ("a cron marker cannot certify a newer protocol stream") is about ORDER, and the
  order-aware rule in §3 satisfies it: a marker appended after the newest v2 batch is a
  valid end; a later client batch reopens the session exactly as today.
- Nothing records WHICH signal ended a session, so the miss rate of each cannot be measured.

---

## 2. What already exists (reuse, do not rebuild)

| existing | used by this plan |
|---|---|
| v2 order-aware completion (`claude_code.py:368-369, :376-379`) | becomes THE rule for all protocols (§3.1) |
| `retire_children_of` + `authoritative` semantics (`claude_code.py:716-727`, `claude_code_extraction.py:249, :958, :1002-1010`) | kept verbatim; new negative tests |
| the emergency stop `CLAUDE_CODE_EXTRACTION_ENABLED` (`claude_code.py:534-549`) | rollback lever for extraction; unchanged |
| `kubectl patch cronjob … suspend` | the stopgap; no code |
| cron `finalize.marker` objects in R2 (consume leaves the object; verified: `raw/claude_code/anthrogen/00261da9-…/finalize.marker`, 135 B, still present) | the backfill re-links keys instead of re-mining |
| `ingestion_queue` has no RLS (`relrowsecurity=f`) | sweep and backfill need no tenant GUC; `documents`/`session_streams` are FORCE RLS |
| `_FETCH_SUPP_R2_CONCURRENCY = 16` semaphore pattern (`claude_code.py:82, :296`) | backfill HEAD/PUT bound |
| `tests/test_session_receipts.py:457` (v2 completion survives the idle sweep) | must be updated when the sweep covers v2 |
| TODOS.md:336 "Per-session parse cache for transcript re-normalization" | related, not blocking; re-prioritize after Phase 0 (98 % fewer runs) |
| Jev client + measured contract (`engine/retrieval/agent/jev.py`, `docs/jev-contract.md`) | Phase 2 shadow, unchanged |

Nothing in this plan needs a schema migration (no engine-pin dependency for kb-migrate),
though the worker image itself ships only when research-os moves its pin.

---

## 3. Phase 0: stop the loop (the bill)

### 3.1 One completion rule

`session_complete` ⇔ **the newest key in `payload_s3_keys` (arrival order) is a finalize
signal**: a v2 finalize batch, a v1 client finalize payload, or a cron marker. A batch
appended after any of them reopens the session. Nothing is consumed.

- Drop the sticky `session_end`-event rule (no producer emits it; a stale one can certify a
  newer stream) and the legacy `:finalize` suffix rule (0 suffixed rows remain in
  `ingestion_queue`).
- Arrival order is what every UPSERT already produces (`payload_s3_keys || EXCLUDED…` in
  `kb/session_receipts.py:_enqueue_agent`, `kb/ingestion_app.py:987`, `session_completer.py:114`);
  protocol 2 additionally enforces sequence order on accept.
- One predicate, one module (`engine/shared/session_signals.py`, name to taste), imported by
  the handler and the sweep. Key-only version for the sweep's SQL (marker or v1 client-finalize
  shape as the last element) plus `session_streams.finalized` for v2; the handler's version
  sees the payloads and can tell a v2 finalize batch.
- A straggler batch that lands after a finalize makes the session live; the sweep re-ends it
  after 24 h idle and it is mined once more. Bounded, and correct: the straggler had content.

### 3.2 Remove consumption

Delete `NormalizationResult.consume_payload_keys`, `Normalizer._consume_payload_keys` and the
`normalizer.consumed_payload_keys` / `consume_payload_keys_failed` events, and the handler's
use (`claude_code.py:728-738`). With §3.1 in place they are a second, contradictory
lifecycle. The double-append at `claude_code.py:345-348` goes with it.

### 3.3 The sweep

- "Already ended" = **the last key is a finalize signal, for both protocols**:
  - v1: last key is a cron marker or a v1 client finalize (`.../<session_id>.json`).
  - v2: last key is a cron marker, OR (`session_streams.finalized` AND the last key is a
    `sessions-v2/` key). `session_streams.finalized` is exactly "the newest ACCEPTED v2 batch
    was a finalize": `accept()` (`kb/session_receipts.py:236-262`) admits a batch after a
    finalize only in sequence, and its UPDATE sets `finalized = bool(payload.finalize)`, so a
    later batch flips it back. The cron marker never touches `session_streams` (it is a server
    observation, not a client claim), which is why the marker must count as an end on its own:
    checking `finalized` alone would re-sweep every cron-ended v2 session daily, i.e. rebuild
    the loop for protocol 2 (outside-voice finding 2).
- Remove the protocol-2 exclusion (`:89-92`) so idle v2 sessions get a marker too. Ships as
  **PR 2**, after the v1 cost fix (PR 1) has been measured (outside-voice recommendation).
- **Recheck under the lock.** Candidates are selected before `_lock` (`:136-145`); a batch can
  land in between. Inside the per-row transaction, the UPDATE carries
  `WHERE enqueued_at < $cutoff AND NOT <last key is a finalize>` and uses `RETURNING`; zero rows
  = skipped, counted separately from failures (outside-voice finding 3).
- UPDATE existing live rows only. Never INSERT a marker-only row: that is the 288 DLQ rows
  (`claude_code.py:772` raises "missing employee_id" on zero events). The handler still
  gets a defensive path: complete + zero events → session doc only, no raise.
- Keep `--idle-minutes 1440`, `--limit 1000`, hourly schedule, `dry_run`, and the `capped`
  log field.

### 3.4 Backfill so the first post-fix sweep does not re-mine ~4,500 sessions once more

`scripts/backfill_finalize_markers.py` re-links the marker key only where there is
**evidence the current transcript prefix was mined by a complete pass** (outside-voice
finding 1: `status='done'` alone also describes a live-batch pass that returned before
extraction, `claude_code.py:526`). A row qualifies iff ALL of:

1. protocol 1 (no `sessions-v2/` key), `status='done'`, idle > 24 h;
2. last key is not a finalize signal (otherwise nothing to do);
3. `completed_at >= enqueued_at`: the worker finished a pass AFTER the last upsert, so the
   pass saw every key now in the array;
4. the R2 object `raw/<src>/<cust>/<sid>/finalize.marker` EXISTS in the tenant bucket
   (`store.bucket_for`): the only way that object exists is a cron finalize, and the only way
   its key is absent from the row is the worker consuming it after an **authoritative**
   complete pass (`claude_code.py:736-738`). A non-authoritative pass leaves the key in place
   and the row fails condition 2 (its retry is §3.8). The script never PUTs a missing marker:
   a missing object is missing evidence, and that row is left for the sweep to mine once.

The UPDATE is conditioned on the observed row (`WHERE queue_id=$1 AND version=$2 AND
status='done' AND completed_at=$3`), `RETURNING queue_id`; a changed row is **skipped and
counted**, never treated as a failure (outside-voice finding 3). No `version`/`status`
change, so no worker pass. `--dry-run` prints candidates, skips, and the count the sweep
will mine once (rows failing 3 or 4). Concurrency 16 for the HEADs. Alternative rejected:
let the sweep do one last pass over everything (~$100 and ~40 k document writes for nothing).

### 3.5 Observability (so the next cost question is answerable from logs)

One INFO event per extraction pass, `claude_code_extraction.pass`: `session_id`, `source`,
`protocol_version`, `completed_by` (`client_finalize` | `process_exit`… as far as known:
`v2_finalize` | `v1_client_finalize` | `cron_marker`), `segments`, `calls`, `authoritative`,
`units`, and the sha256 of each segment's rendered text. The segment hashes are the free
shadow of Phase 1: repeats across passes of one session = the cache's hypothetical hit
rate. Also: a WARNING on the `ToolCallParseError` path
(`claude_code_extraction.py:1205-1209`), today the only silent non-authoritative exit.
Record `completed_by` on the session doc metadata too, and fold `session_complete` into the
session doc's content hash so a no-event completion pass actually writes the flag (§1.3).

### 3.6 Tests (all in the same PR; see §6 for the diagram)

- CRITICAL regression: sweep idempotency after a mined session, for all three finalize
  shapes; batch-after-finalize + idle → exactly one enqueue.
- Completion order for v1 and v2 (positive, negative, stale-`session_end`).
- No consumption; `test_claude_code_finalize_e2e.py:409` rewritten to the order rule.
- Authoritative negatives (non-authoritative → nothing retired; parse error → warning).
- Sweep never INSERTs; handler tolerates zero events.
- Backfill dry-run/live with a fake store.
- Session doc flips `session_complete` on a no-event completion pass.
- Add `tests/test_claude_code_extraction.py`, `tests/test_session_coalescing.py`,
  `tests/test_claude_code_finalize_e2e.py` to the CI list. Comments go ABOVE the step,
  never inside the folded `run: >-` scalar (a `#` inside it comments out every later file).

### 3.7 Rollout and measurement

Two PRs (outside-voice recommendation: keep the money fix's blast radius small):
- **PR 1 (v1 cost fix):** §3.1, §3.2, the v1 half of §3.3 (predicate, recheck, UPDATE-only,
  zero-event path), §3.4, §3.5 logging, §3.6 tests, CI list. No schema change.
- **PR 2 (v2 end + retry):** the v2 half of §3.3, §3.8 (outcome column + bounded retry), the
  `completed_by` / metadata-hash parts of §3.5, and the §4 metrics. One small migration.

1. (Richard) suspend the CronJob now, or not.
2. Merge PR 1. Bump the engine pin in research-os; deploy; the research-plane kb restart is
   manual.
3. With the cron suspended: run the backfill dry-run, compare to `select count(*)` of
   candidates, run live, assert counts.
4. Unsuspend. Watch the first three job logs: `enqueued` should be tens, not hundreds.
5. Measure 48 h: LiteLLM calls/day and $/day for gemini-3.7-flash; minute-of-hour histogram
   (the 79 % spike must be gone); `claude_code_extraction.pass` count by `completed_by`;
   `normalizer.done` added/removed totals.
   **Success:** < 500 calls/day and < $10/day; no session mined more than once without a
   new batch.
6. Rollback: revert + unsuspend restores today's behaviour; `CLAUDE_CODE_EXTRACTION_ENABLED=false`
   stops mining outright; the backfill is additive (keys only).

Blast radius: the worker's completion decision for every coding-agent session, the sweep,
and one queue-column update per idle row. No schema change in PR 1. No other connector reads
`consume_payload_keys` (grep: only `claude_code.py` sets it).

### 3.8 "Ended" is not "mined": bounded retry for non-authoritative and disabled passes (PR 2)

Today and after PR 1 alike, a session whose last complete pass was non-authoritative (a
segment failed, the cap hit, the model declined the tool) or ran under
`CLAUDE_CODE_EXTRACTION_ENABLED=false` keeps a finalize as its last key and is never
revisited unless a new batch arrives (outside-voice finding 4; the "re-enabling re-mines it"
comment at `claude_code.py:534-537` only holds if a batch follows). Fix:

- Migration: `ingestion_queue.extraction_outcome jsonb NULL` (no RLS on this table; DDL
  only; renumber if the alembic number collides). Written by the worker after every complete
  pass: `{"at", "authoritative", "reason": "ok|segment_failed|capped|tool_declined|disabled",
  "keys": <len(payload_s3_keys) at claim>, "retries": n}`.
- Sweep rule 2: a session whose last key is a finalize but whose outcome is missing,
  non-authoritative or `disabled`, idle > 24 h, and `retries < 3` is re-ended (append a marker,
  `version+1`, `pending`, `retries+1`). Three is the bound that keeps a permanently failing
  segment from becoming a new loop; the pass log makes those sessions findable by hand.
- The 177 done v1 rows that still carry a marker today are exactly this population; PR 2's
  first sweep retries them once.

---

## 4. Phase 0b: deterministic end, measured

Delivered inside Phase 0 PR 2 (§3.3 v2 sweep, §3.5 `completed_by`). `completed_by` names
the extraction TRIGGER, not a client miss: an idle-but-open session can get a cron marker
while the client is fine, and a late client finalize can follow a cron mine without adding
content (protocol 2 accepts it, `kb/session_receipts.py:248`) (outside-voice finding 5).
The client miss rate is therefore measured as: **of v2 sessions ended by a cron marker, the
share that never receives a client finalize within 7 days** (a late finalize with an unchanged
prefix is a "late", not a "miss"), plus the share of v2 streams that reach
`session_streams.finalized` within 24 h of their last batch. If misses exceed a few percent,
the tap has a gap worth chasing (pi's `daemon.ts` first: unverified whether it finalizes on
process death the way the CC tap does).

Not now (TODOs): a `reason` on the finalize (the gateway validates a two-field model,
so this is a prbe-backend change); device-heartbeat-driven finalize (the sweep covers it).

---

## 5. Phase 1: per-segment cache (build-ready, deferred)

**Gate:** after Phase 0, `claude_code_extraction.pass` shows repeated segment hashes
accounting for ≥ 25 % of segment calls over a week. Today's estimate is ≤ 8 %
(461 of 5,712 v2 passes are re-completions; the tail segment is new each time).

Spec (per the brief, with the two traps closed):
- Key: `(customer_id, session_id, sha256(rendered segment text), agent, prompt+schema
  version)`. `total`/`(part i of n)` are NOT in the key; drop the part text from the
  prompt or re-stamp `SegmentRef.total` after a hit. Confirm with a test that appending
  events leaves earlier boundaries unchanged (`_split_on_compaction` and `_split_to_budget`
  are prefix-stable by construction: both only ever cut at events already present; the
  cap keeps the LAST 16, so a capped session's earliest segments change identity — cache
  misses, not corruption).
- Store: R2, `raw/<src>/<cust>/<sid>/segments/<sha256>.json`, body = the grounded
  `UnitBundle` for that segment plus `authoritative` and the prompt version. Never write a
  non-authoritative result. No migration, tenant-bucketed, deleted with the session.
- A bundle is authoritative only if every segment (hit or fresh) is.
- `_link_supersessions` re-runs only when the ordered list of decisions changed.
- Measure before/after on the same 20 long v2 sessions: calls per completion pass.

## 6. Phase 2: Jev tail screen (shadow-only, deferred behind Phase 1)

**Gate:** Phase 1 shipped, and the residual is dominated by tail segments that produce
nothing (from the pass log: `units == 0` for the newest segment).

- Screen the DELTA: the newest segment's rendered text, tail-truncated to 100 k chars (the
  conclusion matters most, matching the cap's keep-latest policy); log the truncation so its
  effect on the miss rate is visible.
- Shadow: run extraction anyway; log Jev's probability next to `units` for that segment.
  Nothing is skipped until the miss rate (tails Jev would have skipped that held units) is
  known over ≥ 500 tails.
- A skip is its own outcome (`completed_by`/`skipped_by=jev`), never "mined, found nothing";
  the emergency-stop property at `claude_code.py:534-549` is the model.
- Key is hand-patched into `engine-secrets`; `sync-secrets` full-replaces a Secret.

---

## 7. Review record

### Scope Challenge

Findings (severity, confidence, disposition):

1. **[P0] (10/10)** `kb/session_completer.py:76-88` + `engine/ingest/normalizer.py:250-294` —
   the sweep's "already finalized" predicate is the presence of a key the worker deletes; hourly
   cron + 24 h idle = every idle v1 session re-mined daily. Evidence §0. **Accepted** → Phase 0.
2. **[P1] (9/10)** Phase 1's value after (1) is ≤ 8 % of post-fix passes (73 / 5,251 v2 sessions
   completed more than once). **Deferred** behind the §5 gate (differs from the brief; surfaced).
3. **[P2] (9/10)** Phase 2 likewise. **Deferred** behind §6 gate.
4. **[P1] (9/10)** `.github/workflows/tests.yml:163-182` — the three extraction/coalescing/
   finalize test files are not on the CI list. **Accepted** → Phase 0.
5. **[P2] (9/10)** Managed plane has no engine-worker or completer CronJob in any namespace.
   **No action.**

Complexity gate: 9 files in prbe-knowledge (handler, sweep, normalizer, models, extraction
log, new predicate module, new backfill script, workflow, 5 test files) + research-os pin bump.
One new module (the predicate) and one script. Structure: **Original arrangement** (auto-decided):
the predicate module is what removes the three spellings of "is this a finalize" that let the bug
ship; a "smaller" arrangement that inlines it re-creates the drift. No feature cuts inside
Phase 0. Search check: no new framework pattern or dependency; skipped. Reuse ladder: every
component is rung 1 (existing repo code) except the backfill script.

### 1. Architecture

- **A1 [P1] (9/10)** `claude_code.py:367-379` — four completion signals, three lifecycles. → §3.1. Accepted.
- **A2 [P1] (9/10)** `session_completer.py:89-92, :146-152` — v2 excluded from the sweep; 95 unfinalized v2 streams, 81 > 1 day, never mined. → §3.3. Accepted.
- **A3 [P2] (8/10)** `session_completer.py:99-103, :108-122` INSERTs marker-only rows; `claude_code.py:772` raises on them; 288 DLQ rows. → §3.3. Accepted.
- **A4 [P2] (9/10)** `claude_code.py:874-876, :943` + SCD2 skip-if-unchanged — `session_complete` metadata lies (2,258 / 2,483). → §3.5. Accepted.
- **A5 [P2] (8/10)** Deploy: worker ships only when research-os moves the pin; research-plane restart manual. → §3.7 order. Accepted.
- **A6 [P3] (7/10)** Rollback = revert + unsuspend; extraction emergency stop already exists; no new flag. Accepted.

Failure scenarios for each new path: see §8.

### 2. Code quality

- **C1 [P2] (9/10)** `claude_code.py:316-318` and `:345-348` append the marker key twice (`count: 2` on 2,599 consumes). Removed with §3.2. Accepted.
- **C2 [P2] (9/10)** `claude_code_extraction.py:1205-1209` — `ToolCallParseError` → non-authoritative with no log line. → §3.5. Accepted.
- **C3 [P2] (8/10)** Stale docstrings: `scripts/cron_session_completer.py:16-21`, `tests/test_session_completer.py:246-250`, `normalizer.py:262-264`. Fix in the PR. Accepted.
- **C4 [P3] (8/10)** `_consume_payload_keys` and its result field become dead; delete, do not keep as a second mechanism. → §3.2. Accepted.
- **C5 [P2] (8/10)** DRY: "is this key a finalize" is spelled in three places (`claude_code.py:316, :345`; `session_completer.py:86-87, :91`). → §3.1 predicate module. Accepted.
- **C6 [P3] (7/10)** `_employee_id_from_event` raising for zero-event complete sessions is an error path with no test. → §3.3. Accepted.

### 3. Tests

Framework: pytest (`pyproject.toml`); `live_db` fixture = Postgres + MinIO per
`.github/workflows/tests.yml:39-43`; CI runs a NAMED file list. Existing coverage map and gaps
were read from the six test files (see §6 diagram). Regression contract (IRON RULE) R13.

```
CODE PATHS                                                              STATUS
[~] kb/handlers/claude_code.py  fetch_supplementary
  ├── newest key is v2 finalize            → complete        [★★★ TESTED] receipts:298,357,457 (v2 reopen at :498)
  ├── newest key is v1 client finalize     → complete        [★★  TESTED] fetch_supp:268; finalize_e2e:111 (order not asserted)
  ├── newest key is cron marker            → complete        [★★  TESTED] fetch_supp:183 (order not asserted)
  ├── batch AFTER a v1 finalize            → NOT complete    [GAP] CRITICAL — the rule that ends the loop
  ├── cron marker after the last v2 batch  → complete        [GAP] (new: sweep covers v2)
  ├── stale session_end event in stream    → NOT complete    [GAP] (rule removed; pin it)
  └── no finalize at all                   → NOT complete    [★★  TESTED] fetch_supp:320
[~] kb/handlers/claude_code.py  normalize
  ├── complete + authoritative + units     → retire children [★★★ TESTED] finalize_e2e:320
  ├── complete + NOT authoritative         → retire nothing  [GAP]
  ├── complete + zero events               → session doc only, no raise [GAP]
  ├── extraction disabled                  → un-finalized    [★★★ TESTED] normalize:621
  └── consume_payload_keys                 → REMOVED         [GAP] finalize_e2e:409 rewritten to order rule
[~] kb/session_completer.py  enqueue_idle_session_finalizers
  ├── idle v1, last key not finalize       → 1 enqueue       [★★  TESTED] completer:19 (fresh row untouched)
  ├── idle v1, last key IS finalize        → 0 enqueue       [GAP] CRITICAL — sweep idempotency after a mined session
  ├── idle v1, batch after finalize        → 1 enqueue       [GAP]
  ├── idle v2, session_streams.finalized   → 0 enqueue       [★★  TESTED] receipts:457,506 (pins the OLD exclusion; update)
  ├── idle v2, not finalized               → 1 enqueue (marker) [GAP] (PR 2)
  ├── idle v2, cron-marker-ended, swept AGAIN → 0 enqueue [GAP] CRITICAL (PR 2; the v2 loop guard)
  ├── batch lands between select and lock  → skipped, counted [GAP]
  ├── last key finalize, outcome non-authoritative, retries<3 → re-ended [GAP] (PR 2)
  ├── same, retries==3                     → left alone       [GAP] (PR 2)
  ├── no live row                          → nothing INSERTed [GAP]
  └── limit / dry_run / capped             → [★★  TESTED] completer:276-299
[+] scripts/backfill_finalize_markers.py
  ├── dry-run counts == candidates, skips, sweep-once [GAP]
  ├── appends only where last key not finalize AND completed_at>=enqueued_at AND marker object exists [GAP]
  ├── marker object missing → row left alone, never PUT [GAP]
  ├── row changed between select and update → skipped, counted [GAP]
  └── UPDATE row count asserted            [GAP]
[~] engine/shared/claude_code_extraction.py
  ├── ToolCallParseError → authoritative False + WARNING [GAP]
  ├── segment failure → authoritative False          [GAP] (extraction:239 checks units only)
  ├── cap → authoritative False, keeps LAST 16       [GAP]
  ├── pass log carries segments/calls/hashes/completed_by [GAP]
  └── segmentation, grounding, supersessions          [★★★ TESTED] extraction:176-707
[~] session doc metadata
  └── no-event completion pass writes session_complete=true [GAP]
[+] .github/workflows/tests.yml  three files added        [GAP] (verify by a deliberately failing assert once)

LLM integration: no prompt change in Phase 0 → no [→EVAL]. Phase 2 shadow IS the eval.

COVERAGE (Phase 0 paths): 9/33 tested (27 %) | QUALITY: ★★★:5 ★★:4 | GAPS: 24 (3 CRITICAL, 0 E2E, 0 eval)
```

Legend: ★★★ behavior + edge + error | ★★ happy path | ★ smoke | [→E2E] integration | [→EVAL] LLM eval

Required tests (approved under R13/R10; files match existing names):
- `tests/test_session_completer.py`: idempotency after mined (3 finalize shapes), batch-after-finalize, v2 marker, never-INSERT.
- `tests/handlers/test_claude_code_fetch_supplementary.py`: order rule cases incl. stale `session_end`, marker-after-v2.
- `tests/handlers/test_claude_code_normalize.py`: non-authoritative retires nothing; zero events no raise; no consume field.
- `tests/test_claude_code_finalize_e2e.py:409`: rewrite to the order rule (batch after finalize → 0 extra extraction; second finalize → 1).
- `tests/test_claude_code_extraction.py`: parse-error warning + flag; cap keeps last 16 + flag; pass log fields.
- `tests/test_backfill_finalize_markers.py` (new): fake store, dry-run, HEAD/PUT, row count.
- `tests/test_session_receipts.py:457,506`: v2 sweep now enqueues a marker when `finalized=false` and idle; still 0 when finalized.

### 4. Performance

- **P1 [P1] (9/10)** The loop rewrites ~40 k documents/day (20,873 added + 28,122 removed in 12 h): re-chunk and re-embed of every unit of every idle v1 session, on top of the Gemini bill; unmeasured embedding spend. Removed by Phase 0. Accepted (measure `normalizer.done` added/removed before and after).
- **P2 [P2] (8/10)** Backfill: ~4,500 R2 HEADs; bound at 16 concurrent (≈ 1-2 min). Accepted.
- **P3 [P3] (7/10)** TODOS.md:336 (O(session) parse per run) matters 98 % less after Phase 0; re-prioritize, do not build first. Accepted → TODO edit.
- Suppressed: sweep GROUP BY over ~13 k rows hourly (confidence 5; fine at this size).

### Decision ledger

Records are terse by the reviewer's standing instruction ("go with recommended; surface only
what changes something I said or mutates prod"). Each carries the grid's essential row.

#### R1: Phase order (differs from the brief)
Finding: Scope 1-3. Plan baseline: brief = Phase 1 then Phase 2. Runtime evidence: §0 table.
Grid: A) Phase 0 first, Phases 1/2 gated on measurement — saves ≥ 90 % now, defers ~$10-20/mo of
cache value; B) brief order — builds the cache against a bill that is 97 % something else.
Question D1 — Which order? Recommendation: A because the measured cause is the sweep, not re-mines.
Note: options differ in kind, not coverage. Options: A) Phase 0 first (recommended) B) Brief order.
State: approved (auto, recommended). Actual answer: A. Accepted scope: §3-§6 as written. Surfaced to Richard in §0.

#### R2: Stopgap — suspend the CronJob today
Finding: Scope 1. Plan baseline: none. Runtime evidence: job logs 356/74/819 per hour.
Grid: A) suspend now (~$90/day stops; crashed-client v1 sessions wait) B) leave running until the fix lands (~$95/day for the PR's lifetime).
Question D2 — Suspend now? Recommendation: A because it is reversible and the fix is days away. Note: options differ in kind.
Options: A) Suspend now (recommended) B) Wait for the fix.
State: **pending — Richard's call (mutates prod)**. Actual answer: unanswered. Accepted scope: none until answered.

#### R3: One completion rule (newest key is a finalize)
Finding: A1, C5. Baseline: four signals. Evidence: §1.1. Grid: A) order-aware rule, drop session_end and `:finalize` (10/10) B) keep consumption, add a `mined_at` column (7/10: schema migration through the pinned engine image; two mechanisms remain).
Question D3. Recommendation: A. Completeness: A=10/10, B=7/10. Options: A) Order rule (recommended) B) mined_at column.
State: approved (auto). Answer: A. Scope: §3.1.

#### R4: Remove consumption
Finding: C1, C4. Grid: A) delete field+method+events (10/10) B) leave dead code (3/10).
Question D4. Recommendation: A. Completeness A=10, B=3. State: approved (auto). Answer: A. Scope: §3.2.

#### R5: Backfill vs one last pass
Finding: Scope 1, P2. Grid: A) backfill re-links marker keys, HEAD+PUT, asserted counts (10/10) B) accept one last sweep (~$100, ~40 k doc writes) (7/10).
Question D5. Recommendation: A. Completeness A=10, B=7. State: approved (auto). Answer: A. Scope: §3.4.
History: the original candidate rule (done + idle + last key not finalize) was superseded by R15's evidence rule after outside-voice finding 1.

#### R6: Sweep covers protocol 2 via the marker; `session_streams` untouched
Finding: A2. Grid: A) marker after newest v2 key = complete; streams untouched (10/10) B) keep exclusion, 95 sessions never mined (3/10) C) write `session_streams.finalized` from the cron (5/10: forges a client claim; breaks accept()'s prefix checks).
Question D6. Recommendation: A. Completeness A=10, B=3, C=5. State: approved (auto). Answer: A. Scope: §3.3.
History: the first draft's sweep predicate for v2 ("`session_streams.finalized`") would have re-swept every cron-ended v2 session daily; amended by R16 after outside-voice finding 2.

#### R7: Marker-only rows
Finding: A3, C6. Grid: A) sweep UPDATEs only + handler tolerates zero events (10/10) B) handler-only fix (7/10).
Question D7. Recommendation: A. Completeness A=10, B=7. State: approved (auto). Answer: A. Scope: §3.3.

#### R8: `session_complete` metadata truthfulness
Finding: A4. Grid: A) fold the flag into the session doc content hash (10/10) B) leave; use the pass log only (6/10: every dashboard/query on the flag stays wrong).
Question D8. Recommendation: A. Completeness A=10, B=6. State: approved (auto). Answer: A. Scope: §3.5.

#### R9: Pass-level observability + parse-error warning + segment hashes
Finding: C2, Scope 2. Grid: A) one INFO event with completed_by/segments/calls/hashes + WARNING (10/10) B) warning only (5/10).
Question D9. Recommendation: A. Completeness A=10, B=5. State: approved (auto). Answer: A. Scope: §3.5.

#### R10: CI list
Finding: Scope 4. Grid: A) add the three files (10/10) B) leave (0/10).
Question D10. Recommendation: A. State: approved (auto). Answer: A. Scope: §3.6 last bullet.

#### R11: Phase 1 spec and gate
Finding: Scope 2. Grid: A) build-ready spec, R2-backed, content-keyed, gate ≥ 25 % repeated segment calls B) build now. Note: differ in kind.
Question D11. Recommendation: A. State: approved (auto). Answer: A. Scope: §5.

#### R12: Phase 2 shadow and truncation
Finding: Scope 3. Grid: A) tail-truncate 100 k chars, log truncation, ≥ 500-tail shadow before any skip B) head+tail sampling. Note: differ in kind.
Question D12. Recommendation: A. State: approved (auto). Answer: A. Scope: §6.

#### R13: Regression contract (IRON RULE)
Behaviour to preserve, with assertions: (a) v2 completion unchanged for [b0,b1,finalize] and reopened by [b0,finalize,b2] (`receipts:298,357,498`); (b) `retire_children_of` only when authoritative and units > 1; (c) extraction-disabled leaves the session un-finalized and consumes nothing; (d) session doc coalesces in place while incomplete (`coalesce_into_live`); (e) unit ids stay segment-scoped (`normalize:570`). Intentional changes: no consumption; stale `session_end` no longer completes; sweep reaches v2; marker-only rows are never created.
Question D13 — how to cover? Recommendation: existing tests kept green + the new tests in §7.3. Completeness 10/10. State: approved (auto). Answer: as recommended.

#### R14: PR split (outside voice)
Finding: Codex recommendation. Grid: A) PR 1 = v1 cost fix, PR 2 = v2 end + retry + metrics; B) one PR. Note: differ in kind.
Question D14. Recommendation: A because the money fix should not wait on, or share blast radius with, the v2 sweep. State: approved (auto). Answer: A. Scope: §3.7.

#### R15: Backfill evidence rule (outside voice finding 1)
Finding: Codex P1. Baseline: R5 (backfill on done+idle). Grid: A) require `completed_at >= enqueued_at` AND the R2 marker object exists; never PUT; leave the rest for one sweep pass (10/10) B) original rule (4/10: can end never-mined sessions).
Question D15. Recommendation: A. Completeness A=10, B=4. State: approved (auto). Answer: A. Scope: §3.4. Supersedes R5's rule; R5's choice (backfill over one last pass) stands.

#### R16: v2 "already ended" predicate (outside voice finding 2)
Finding: Codex P1. Baseline: R6 text ("`session_streams.finalized` for v2"). Grid: A) last key is a cron marker OR (finalized AND last key is a v2 key), regression test = two successive sweeps after a v2 marker mine return 0 (10/10) B) original (2/10: rebuilds the loop for v2).
Question D16. Recommendation: A. Completeness A=10, B=2. State: approved (auto). Answer: A. Scope: §3.3. Amends R6.

#### R17: Recheck under the lock; version-conditioned backfill (outside voice finding 3)
Finding: Codex P1. Grid: A) sweep UPDATE with idle+predicate WHERE and RETURNING; backfill UPDATE conditioned on version/status/completed_at; skips counted (10/10) B) trust the pre-lock selection (6/10).
Question D17. Recommendation: A. Completeness A=10, B=6. State: approved (auto). Answer: A. Scope: §3.3, §3.4.

#### R18: Bounded retry for non-authoritative / disabled passes (outside voice finding 4)
Finding: Codex P1. Grid: A) `extraction_outcome` column + sweep retries ≤ 3 in PR 2 (10/10) B) leave the hole, document it (3/10) C) retry without a bound (0/10: a new loop).
Question D18. Recommendation: A. Completeness A=10, B=3, C=0. State: approved (auto). Answer: A. Scope: §3.8.

#### R19: Client miss-rate metric (outside voice finding 5)
Finding: Codex P2. Grid: A) cron-ended sessions that never get a client finalize in 7 d + streams finalized within 24 h of last batch (10/10) B) `completed_by` share (4/10: counts triggers).
Question D19. Recommendation: A. Completeness A=10, B=4. State: approved (auto). Answer: A. Scope: §4.

Approval readiness: PASS for R1, R3-R19 (auto-decisions under the standing instruction; R15 and R16 amend R5/R6 and are recorded in their History). R2 pending (execution step, Richard's call).

### Outside voice (Codex, `gpt-6-astra`, completed)

Five findings, all accepted and folded: (1) backfill could suppress unmined content → R15;
(2) v2 predicate recreated the loop → R16; (3) no stale-candidate protection → R17;
(4) "ended" ≠ "mined", no retry → R18; (5) miss-rate metric invalid → R19. Recommendation
"revise before implementation and separate v2 sweep expansion from the v1 cost fix" → R14.
Cross-model tension: none remaining; the native review had under-specified the v2 predicate
and the backfill's evidence, and Codex's corrections are now the spec.

---

## 8. Failure modes (new paths)

| path | realistic failure | test | handling | user sees |
|---|---|---|---|---|
| order rule | v1 client-finalize key shape mis-detected (`.../<sid>.json` has no `:seq`) → sessions never complete | §7.3 fixtures use the three real key shapes | predicate unit-tested | silent under-mining → **covered by tests** |
| sweep covers v2 | laptop asleep > 24 h, then resumes: marker lands, one extra mine after the next finalize | receipts:498-style reopen test | bounded by the order rule | nothing |
| backfill | R2 marker object gone → next claim of that row 404s → DLQ | HEAD/PUT test | PUT placeholder | nothing |
| backfill | script run while cron unsuspended → race: cron re-appends before script → duplicate marker keys | runbook orders it; idempotent append is harmless | last-key rule tolerates duplicates | nothing |
| no consumption | a v1 session with the marker present AND live batches after it: today re-mined per batch; after: not complete until re-swept | fetch_supp order test | correct by rule | mining delayed ≤ 24 h |
| pass log | structlog `extra=` vs kwargs: the extraction module logs `extra={...}` (renders as a nested key); use kwargs like the normalizer | log-field test | — | none |
| metadata hash | folding the flag changes content_hash for every session doc once → one extra SCD2 version per session on next pass (cheap; chunks reused by hash) | live_db test | expected | none |

Critical gaps (no test AND no handling AND silent): **0** after §7.3 lands; 2 CRITICAL tests are named.

---

## 9. NOT in scope

- Per-segment cache (§5): deferred behind measurement; the pass log gives its hit rate free.
- Jev tail screen (§6): deferred behind Phase 1; shadow design stands.
- Finalize `reason` on the wire: gateway model change (prbe-backend).
- Device-heartbeat-driven finalize: the v2 sweep covers the machine-gone case.
- Re-mining the 288 DLQ marker-only rows: they hold no events; delete or leave.
- TODOS.md:336 parse cache: re-prioritize only.
- pi daemon process-death finalize: verify, separate repo.

## 10. Worktree parallelization

| step | modules | depends on |
|---|---|---|
| S1 predicate module + handler order rule + remove consumption | engine/shared, kb/handlers, engine/ingest, engine/shared/models | — |
| S2 sweep (v2, UPDATE-only, predicate) | kb/session_completer, scripts/cron_session_completer | S1 (predicate) |
| S3 backfill script + test | scripts/, tests/ | S1 (predicate) |
| S4 observability + metadata hash | engine/shared/claude_code_extraction, kb/handlers | S1 |
| S5 CI list + docstring fixes | .github, tests, scripts | — |
| S6 research-os pin bump + deploy + runbook | research-os | S1-S5 merged |

Lanes: `Lane A: S1 → S2 → S4 (sequential, shared kb/handlers)` / `Lane B: S3 (independent after the predicate's interface is fixed)` / `Lane C: S5 (independent)`. Launch B and C once S1's predicate signature is committed; merge all; then S6. Conflict flag: `kb/handlers/claude_code.py` is touched by S1 and S4: keep them in one lane.

## 11. Implementation Tasks

Synthesized from the findings above. Run with Claude Code or Codex; checkbox as you ship.

- [ ] **T1 (P1, human: ~2h / CC: ~10min)** — sweep — *(Richard)* suspend `research-os-engine-session-completer` on `do-sfo3-probe-research` if R2 = A
  - Surfaced by: Scope 1 / R2
  - Files: none (`kubectl patch cronjob … -p '{"spec":{"suspend":true}}'`)
  - Verify: next hour's job does not run; LiteLLM calls in minutes 00-09 drop
- [ ] **T2 (P1, human: ~1d / CC: ~30min)** — engine/shared + kb/handlers — one completion predicate; newest-key rule in `fetch_supplementary`; drop `session_end` and `:finalize` rules
  - Surfaced by: A1, C5 / R3
  - Files: `engine/shared/session_signals.py` (new), `kb/handlers/claude_code.py:307-388`
  - Verify: `pytest tests/handlers/test_claude_code_fetch_supplementary.py tests/test_session_receipts.py`
- [ ] **T3 (P1, human: ~2h / CC: ~10min)** — normalizer + models + handler — delete consumption (field, method, events, handler use, double-append)
  - Surfaced by: C1, C4 / R4
  - Files: `engine/ingest/normalizer.py:250-294, :653-654`, `engine/shared/models.py`, `kb/handlers/claude_code.py:316-318, :345-348, :728-738`
  - Verify: `grep -rn consume_payload_keys` returns nothing; finalize_e2e rewritten test green
- [ ] **T4 (P1, human: ~1d / CC: ~30min)** — sweep — predicate-based "already finalized"; include v2; UPDATE-only; handler tolerates zero events
  - Surfaced by: A2, A3, C6 / R6, R7
  - Files: `kb/session_completer.py:59-122`, `kb/handlers/claude_code.py:759-773`
  - Verify: `pytest tests/test_session_completer.py tests/test_session_receipts.py -k "sweep or completer"`
- [ ] **T5 (P1, human: ~1d / CC: ~30min)** — tests — CRITICAL: sweep idempotency after a mined session (3 shapes); batch-after-finalize; v2 marker; never-INSERT
  - Surfaced by: Test review gaps / R13
  - Files: `tests/test_session_completer.py`, `tests/handlers/test_claude_code_fetch_supplementary.py`
  - Verify: the idempotency test FAILS on `origin/main` (proves it catches today's loop), passes on the branch
- [ ] **T6 (P1, human: ~4h / CC: ~20min)** — scripts — `backfill_finalize_markers.py` with dry-run, HEAD/PUT, asserted counts, concurrency 16
  - Surfaced by: Scope 1, P2 / R5
  - Files: `scripts/backfill_finalize_markers.py` (new), `tests/test_backfill_finalize_markers.py` (new)
  - Verify: dry-run count == `select count(*)` of candidates on research kb before the live run
- [ ] **T7 (P2, human: ~4h / CC: ~20min)** — extraction + handler — `claude_code_extraction.pass` INFO (completed_by, segments, calls, hashes, authoritative, units); WARNING on `ToolCallParseError`; `completed_by` in session doc metadata; fold `session_complete` into the content hash
  - Surfaced by: C2, A4 / R8, R9
  - Files: `engine/shared/claude_code_extraction.py:936-1015, :1205-1209`, `kb/handlers/claude_code.py:860-945`
  - Verify: `pytest tests/test_claude_code_extraction.py tests/handlers/test_claude_code_normalize.py`; one live_db test that a no-event completion writes the flag
- [ ] **T8 (P2, human: ~1h / CC: ~5min)** — CI + docs — add the three test files above the folded scalar; fix the three stale docstrings
  - Surfaced by: Scope 4, C3 / R10
  - Files: `.github/workflows/tests.yml:158-182`, `scripts/cron_session_completer.py:16-21`, `tests/test_session_completer.py:244-252`, `engine/ingest/normalizer.py` (docstring gone with T3)
  - Verify: push a commit with a deliberate failing assert in `test_claude_code_extraction.py` once; CI must go red
- [ ] **T9 (P1, human: ~2h / CC: ~15min)** — research-os — bump the engine pin; deploy; manual restart; runbook §3.7 steps 3-5
  - Surfaced by: A5
  - Files: research-os chart values (pin)
  - Verify: `kubectl -n research logs job/<completer>` shows `enqueued` in the tens; LiteLLM < 500 calls/day after 48 h
- [ ] **T10 (P3, human: ~1h / CC: ~5min)** — TODOS.md — entries per §12
  - Surfaced by: TODO tail
  - Files: `TODOS.md`
  - Verify: entries follow `review/TODOS-format.md`
- [ ] **T11 (P1, PR 2, human: ~1d / CC: ~40min)** — sweep + migration — v2 predicate (marker OR finalized+v2 key), `extraction_outcome` column, bounded retry ≤ 3, worker writes the outcome after every complete pass
  - Surfaced by: Outside voice 2, 4 / R16, R18
  - Files: `kb/session_completer.py`, `db/migrations/versions/<next>_extraction_outcome.py` (new), `engine/ingest/normalizer.py`, `kb/handlers/claude_code.py`
  - Verify: CRITICAL test: two successive sweeps after a cron-marker mine of a v2 session return 0; retry test at 2 and at 3; `tests/test_session_receipts.py:457,506` updated
- [ ] **T12 (P2, PR 2, human: ~2h / CC: ~15min)** — metrics — client miss-rate query (cron-ended v2 sessions without a client finalize in 7 d; streams finalized within 24 h) as a `probe-peek`-style SQL section or a scheduled log line
  - Surfaced by: Outside voice 5 / R19
  - Files: `scripts/` (query), docs
  - Verify: the query returns a number on the research kb for last week

Ordering: T1 → (T2, T3, T5, T6, T8 in PR 1; T4's v1 half in PR 1) → T9 → measure 48 h → (T4's v2 half, T7, T11, T12 in PR 2) → T10 any time.

## 12. TODOS.md updates (dispositions auto-taken)

| TODO | disposition |
|---|---|
| Phase 1 per-segment cache spec + gate (§5) | Add |
| Phase 2 Jev tail screen shadow + gate (§6) | Add |
| Finalize `reason` on the wire (gateway two-field model) | Add |
| pi daemon: verify process-death finalize | Add |
| The 177 done v1 rows still carrying a marker: identify the non-authoritative cause once the pass log exists | Add |
| Re-prioritize TODOS.md:336 after Phase 0 | Edit existing entry |
| Device-heartbeat-driven finalize | Skip (v2 sweep covers it) |

## 13. Suppressed findings (appendix)

- Sweep `GROUP BY` over ~13 k rows hourly (confidence 5): fine at this size; revisit only if `ingestion_queue` grows 10x.
- `worker.cas_retry` 394 in 12 h (confidence 4): unrelated to spend; not investigated.

## 14. Unresolved decisions that may bite you later

- **R2 / D2**: suspend the CronJob now or wait for the fix. Richard's call; nothing else waits on it.

## 15. Completion summary

- Step 0: Scope Challenge — scope reduced per recommendation (Phases 1-2 deferred; Phase 0 added)
- Architecture Review: 6 issues found
- Code Quality Review: 6 issues found
- Test Review: diagram produced, 24 gaps identified (3 CRITICAL)
- Performance Review: 3 issues found (1 suppressed)
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 7 items proposed (dispositions in §12)
- Failure modes: 0 critical gaps flagged (after the named tests)
- Unresolved decisions: 1 in this review (R2, execution step)
- Outside voice: codex, completed, 5 findings, all folded (R14-R19)
- Parallelization: 3 lanes, 2 parallel / 1 sequential
- Lake Score: 14/14 (R3-R10, R13, R15-R19 at 10/10; R1, R2, R11, R12, R14 differ in kind)
- Issues found (all sections + outside voice): 6 + 6 + 24 + 3 + 5 = 44; all mapped to tasks or TODOs

## GSTACK REVIEW REPORT

Target: the pasted brief "Plan (do not yet ship) cutting transcript-extraction spend", materialized as this file. Commit `2b540de`, branch `extraction-spend-plan`, 2026-09-23.

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Review | codex (`gpt-6-astra`) via `/plan-eng-review` | Independent 2nd opinion | 1 | completed | 5 findings, 5 folded (R14-R19) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES OPEN (mapped work) | 44 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE COVERAGE:** provider codex, model `gpt-6-astra`, phase plan-review, completed; 5 findings, all accepted (backfill evidence rule, v2 predicate loop, lock recheck, bounded retry, miss-rate metric) and one sequencing recommendation (PR split) adopted.
- **CROSS-MODEL:** native review (this session's harness; model identity as reported by the host) and codex agree on the mechanism and the fix shape; codex supplied five corrections the native pass under-specified; no disagreement remains.
- **VERDICT:** Eng Review ISSUES OPEN: every one of the 44 findings is mapped to a task (T1-T12) or a TODO and no critical failure gap remains, so this is mapped work, not a failed review; eng review required (re-run after PR 1 lands or if R2 changes the rollout order).

**UNRESOLVED DECISIONS:**
- R2 / D2: suspend `research-os-engine-session-completer` on `do-sfo3-probe-research` now (reversible, ~$90/day) or leave it until PR 1 deploys. Richard's call; nothing in the plan waits on it.

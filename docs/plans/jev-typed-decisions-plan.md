# Jev typed decisions in the retrieval gatherer — Phase 0 plan (eng-reviewed)

Written 2026-09-22 · repo `prbe-ai/prbe-knowledge` · worktree `~/trees/prbe-knowledge/jev-shadow-scoring` (branch `jev-shadow-scoring`, from main `7af68c2`)
Source brief: Richard's task message + the "fold entity extraction in" addendum (same day).
Review: `/plan-eng-review`, decisions auto-taken on the recommended option per Richard's standing instruction; the two forks that need him are in "Open items".

## TL;DR

- **Phase 0 is a REPLAY over stored trace blobs, not a live shadow hook.** Every search already persists a gzip'd trace to R2 (`SEARCH_AGENT_TRACE_SAMPLE_RATE=1.0`) carrying the full pre-fan-out pool WITH content, the delivered chunks, and a `harness_appended` flag per chunk. Research cluster: ~3,600 blobs in the last 30 days (excluding `new-workspace`). No deploy, no key on a pod, no request-path code, no waiting days.
- **Two capture gaps block a complete replay and are the only production change in Phase 0** (log-only, no behaviour change): (1) the terminal `emit_gatherer_output` arguments are never written to the blob (`loop.py:3696` returns without appending the assistant turn), so "what the model actually named" is unrecoverable; (2) the grounding bundle and the extractor's output are not persisted on the gatherer path (`query_traces.grounding_bundle` is NULL on 3,255/3,255 gatherer rows; only the list pipeline sets it). Blob schema v3 adds `terminal_raw`, `grounding`, `extraction`.
- **The gatherer's own curation is nearly absent today.** 36 non-empty sampled traces: median chunks the model emitted = **0** (25/36 emitted zero); median appended by the recall floor = **10**. Worse than the 12% measured 09-11. So "gatherer vs Jev" is trivially won; the real comparison is **Jev vs the recall floor (RRF top-10)**, and that needs a relevance judgment, not just set overlap → LLM judge on the disagreement region + a hand-labelled calibration slice.
- **Extraction addendum, corrected:** the extractor's pick among candidates does NOT narrow the pool. Every grounding candidate is anchored regardless (`loop.py:2870-2882` → `execute_search(entity_ids=…)`, bag p50 = 50 anchors per sub-query). What extraction DOES change: `search_options.sort`, `doc_types`, extra non-candidate entities, person→author filter (gated), and the `sub_queries` text. The extraction shadow measures THOSE.
- **Three arms, not two (from the outside voice):** (A) today's delivery, (B) **floor-only — no model call at all**, (C) Jev topped up by the floor. On the plan's own numbers, arm B alone removes the $293 gatherer line; if the judge cannot tell A from B, the cheapest cutover is "turn the gatherer off" and Jev is a precision play on top. Compared at a fixed budget (10 distinct documents), on held-out queries, with extraction retained.
- **Money:** Cerebras gpt-oss-120b, both clusters, last 30 days ≈ **$375 true** ($319 as LiteLLM reports it): gatherer ≈ $293, extractor ≈ $22, auto-merge (ingest) ≈ $60. Jev for both stages ≈ **$16/month** (+ ≈ $9 for Phase 2 reformulation at 30% fire rate). **Saves ≈ $290/month ≈ 77% of the Cerebras line, ≈ $3.5k/yr.** Latency: gatherer turn p50 1.9s → ≈ 0.3-0.6s; extraction p50 0.8s → ≈ 0.2s.

## Verified facts (read the code / the data, not the brief)

1. **Sequential, not parallel.** Grounding + id-lookup run in one `asyncio.gather` (`loop.py:2767`), then extraction is awaited (`loop.py:2839`). `extractor.py:3-16` docstring agrees. `models.py:375-383` ("parallel with deterministic grounding") is STALE — fix the comment in the Phase 0 PR.
2. **Extraction latency (39 blobs, `timing_ms.extraction_ms`):** p50 805 ms · p90 1,713 · p99 2,486 · max 5,115. Client-side cap is 6.0 s (`SEARCH_AGENT_EXTRACTOR_TIMEOUT_SECONDS`, passed at `extractor.py:409`; `llm.py:541` is a `setdefault`, so the explicit value survives). The proxy rung: research cluster is ALREADY 14 s (`research-os` values.yaml:803 `searchAgentTimeout: 14`, PR research-os#581 merged); managed is still 12 s (`prbe-backend/charts/litellm/values.yaml:180` on main). **Unverified:** whether the LiteLLM SDK forwards `timeout` in-body to the proxy or applies it on the httpx client; if client-side (the documented behaviour), extraction is bounded at 6 s regardless of the rung and the "14 s hang budget" concern does not apply. Either way Jev takes extraction off the rung.
3. **Three Cerebras callers:** gatherer (`loop.py`), extractor (`extractor.py:384`), auto-merge analyzer (`engine/ingest/auto_merge/analyzer.py:406`). Auto-merge is out of scope and stays.
4. **The "grounding gate" only SWAPS, never drops.** `_reconcile_entities_with_bundle` (`router.py:144-187`) replaces an unknown canonical_id with a candidate on substring match; an invented id that matches nothing is KEPT and threaded as an anchor (matches no graph node → harmless no-op, except a `person` under `entity_must_match`).
5. **Pool shape (36 non-empty blobs):** unique chunks median 97.5, p90 121; content ≈ chars/4 → median ≈ 20.6k tokens, p90 ≈ 29.9k, max ≈ 33.2k. Channel budgets are 60 vector + 60 bm25 + ≤60 graph per query (2 sub-queries × 30). So ~90% of pools fit ONE Jev state (32k cap); the rest need 2 batches. The estimate is chars/4 — code/log chunks tokenize denser; batch by a real count with a safety factor, never assume one pass.
6. **Delivered chunks reachable ONLY through the extractor's reformulated sub-query: 23/359 = 6.4%** (upper bound on what dropping `sub_queries` from the happy path can cost).
7. **Graph hits carry `via_entity`** (1,124/1,124 in the sample), so the replay can attribute every delivered graph hit to the anchor that produced it — which is how "would a focused, Jev-picked anchor have kept it" gets measured.
8. **Trace blob contents (v2, `trace_blob.py:54-183`):** `prefanout` (pool with content, per sub-query, per channel, incl. `grounded_entities` = the anchor bag and `query` text), `gathered` (delivered chunks, `harness_appended`), `messages` (system + user + non-terminal turns), timings, `search_options`, seeds. NOT in it: the terminal call's raw args, the grounding bundle as a structure (only rendered as `<grounding>` text lines in `messages[1]`), the extractor's `EntityExtraction`.
9. **Volume, 30 days:** research `query_traces` with a gatherer status, excl. `new-workspace`: 3,255 (probe 1,895 blobs, strand-ai 654, anthrogen 253, bucket-robotics 179, monarcha 92, taan-ai 85). Managed `probe` db, tenant `probe-founders`: 7,164 rows / 6,094 blobs; LiteLLM on managed also shows a `customer-c751dbfa…` key with 8,880 gatherer-shaped calls (its DB/bucket not located from this box). The brief's "~21 runs / 48h" is off by ~50×.
10. **Jev contract (docs only — NOBODY here has called it; no key on this box):** `POST https://api.typesafe.ai/v1/systemone`, `pip install typesafe-sdk` (0.7.1 on PyPI), `TYPESAFE_API_KEY`; question types `Noul` (P(true)), `Choice` (≤255 options, probabilities sum to 1 + confidence), `Score` (2-10 ordinal levels); limits 64k tokens per request, 32k for state; 1,200 req/min, 250k tok/s; $0.042/M input, output free; 70-500 ms. Also exposed as Cloudflare Workers AI `typesafe/jev` (pricing not listed on the CF pricing page). `docs.typesafe.ai` redirects 308 → `/introduction`; the API endpoint answers 403 unauthenticated (it exists). First Phase 0 step is a live contract probe: max questions per request, real tokenizer count, error shapes.

## Cost model (LiteLLM_SpendLogs on both clusters, 30 days to 2026-09-22; true rate = prompt×$0.35/M + completion×$0.75/M — LiteLLM prices Cerebras cached tokens at $0, Cerebras does not)

| cluster | calls | prompt tok | completion tok | true $ | LiteLLM $ |
|---|---|---|---|---|---|
| managed, ≥8k prompt (gatherer turns) | 15,953 | 570.5M | 23.9M | 217.6 | — |
| managed, <8k (extractor ≈5.6k calls + auto-merge ≈82k calls) | 89,708 | 166.1M | 21.0M | 73.9 | — |
| managed total | 105,679 | 736.6M | 44.9M | **291.5** | 240.96 |
| research, ≥8k (gatherer turns) | 6,264 | 195.2M | 8.9M | 75.0 | — |
| research, <8k (extractor) | 3,541 | 18.3M | 2.5M | 8.3 | — |
| research total | 9,805 | 213.5M | 11.4M | **83.3** | 78.39 |
| **both** | | | | **374.8** | 319.35 |

Attribution is by prompt size (the shared key carries no stage tag): gatherer ≈ $293 (78%), extractor ≈ $22 (6%), auto-merge ≈ $60 (16%). Per gatherer turn: 35.8k in + 1.5k out = $0.0136; ≈1.8 turns/search → ≈ $0.024/search; extractor ≈ $0.0025/search. ≈ 22.2k gatherer turns / 30d ≈ 12k searches.

**Jev:** selection ≈ 23k tokens/search (pool ≈ 20.6k + ≈100 Noul stubs) → $0.0010; extraction ≈ 6-8k (query + ≤106 candidates + one Choice) → $0.0003. 12k searches × $0.0013 ≈ **$16/30d**. Phase 2 reformulation at 30% × 12k × ≈$0.0025 (query + top-scored snippets, NOT the 35k pool prompt) ≈ **$9**. Post-cutover ≈ $85 (auto-merge $60 + Jev $16 + reformulation $9) vs $375 → **≈ $290/month saved**. If the reformulation naively re-sends the full gatherer prompt: 30% × 12k × $0.0136 = $49 → still ≈ $250 saved. Phase 0 replay itself: 3,600 blobs × 23k ≈ 83M tokens ≈ **$3.50**; judge on disagreements ≈ $2-5 on gemini flash-lite.

## Phase 0 design — one replay harness, two stages

```
R2 prbe-research/<tenant>/search-traces/<date>/<trace>.json.gz   (v2 blobs, today)
        │  rclone copy (bounded, per tenant/date)             + v3 blobs after the capture PR
        ▼
scripts/jev_shadow_replay.py  ── per blob ──▶  engine/retrieval/agent/jev_shadow.py (pure)
        │                                        ├─ pool(prefanout)          → P  (unique chunk_id, content)
        │                                        ├─ rendered(prefanout)      → R  (re-run _render_prefanout_budgeted, budget 19,100)
        │                                        ├─ delivered(gathered)      → G (model) ∪ B (harness_appended)
        │                                        ├─ terminal_raw (v3 only)   → G_raw (what the model NAMED pre-coercion)
        │                                        ├─ to_jev_batches(P, query) → ≤32k-token states, one Noul per chunk
        │                                        ├─ select(scores, θ)        → J_θ  for θ ∈ {0.5, 0.7, 0.9} and top-10
        │                                        └─ extraction: candidates (v3 `grounding`, else parse <grounding>),
        │                                             extractor pick (v3 `extraction`, else pod-log join on trace_id),
        │                                             Jev Choice over candidates + "none of these", Jev Choice sort∈{relevance,recency}
        ▼
TypeSafeClient (typesafe-sdk, ≤8 concurrent, 5s timeout, retries on 5xx/429, exc-name logged)
        ▼
results.jsonl (one row per trace × θ) ──▶ report.md
   selection: |P| |R| |G| |G_raw| |B| |J_θ|, J∩G, J∩B, J∖(G∪B), (G∪B)∖J, J∩(P∖R) ("outside what the model could see"),
              per tenant × status; Jev determinism (3 re-scores of 50 blobs); P(relevant) histogram; weak-score rate for Phase 2 sizing
   judge:     disagreement region only (J_θ ⊕ B) → gemini flash-lite "does this chunk help answer <query>" → precision of J-only vs B-only;
              calibrated on a 40-query × ~20-chunk hand-labelled slice (Richard/Mahit), report judge-vs-human agreement
   extraction: pick == Jev pick? Jev confidence on disagreements; sort disagreement; extractor non-candidate entities and
              whether any delivered graph hit has via_entity ∈ them; share of delivered chunks reachable only via sub_queries
```

**Question design (verify live in step 1):** one `Noul` per chunk ("This chunk answers or directly supports the query", criteria true/false) — independent, calibrated, thresholdable. NOT one `Choice` over ≤255 chunks: a distribution summing to 1 cannot be thresholded per chunk. Extraction: `Choice` over candidate canonical_ids + `none_of_these`; abstain when confidence < τ (τ swept in the report), fall back to grounding-only anchors — the documented fallback when extraction fails.

**Exclusions:** tenants `new-workspace`, `oneleet-*`, `test-prod`, `testing`, `probe-demo`; statuses `zero_recall_short_circuit`, `id_lookup_short_circuit`. Stratify the report by tenant and by status — `loop_timeout` / `schema_violation` / `output_truncated` (321 of 3,255 rows) is where Jev's upside is largest because the gatherer delivered nothing of its own.

**The decision Phase 0 reports:** (a) Jev-only vs backfill-only precision on the disagreement region (judge, calibrated); (b) recall of J_θ over G∪B as a floor check; (c) how much of J lies outside R (the render budget ceiling the gatherer cannot see past); (d) extraction disagreement rate with Jev confidence; (e) weak-score rate → Phase 2 fire rate and cost. Cutover (Phase 1) is proposed only if (a) ≥ backfill's precision AND (b) ≥ 0.9 on `ok` traces.

## Decisions (auto-taken on the recommended option; Richard's standing rule)

- **D1 Replay over R2 blobs, not a live shadow hook** — zero request-path code; queries are real; ~3,600 blobs available now. Live scoring in-path only if Phase 1 needs forward data. Completeness A=9/10 (replay) vs B=6/10 (live hook: days of data, key on pods, sync-secrets trap).
- **D2 Capture PR (blob v3): persist `terminal_raw` (the emit args pre-coercion), `grounding` (bundle jsonable), `extraction` (EntityExtraction dump + reconciled bag); set `request.state.grounding_bundle` on the gatherer path so `query_traces.grounding_bundle` stops being NULL.** Log-only, bump `TRACE_BLOB_SCHEMA_VERSION` to 3, `trace_analyzer/digest.py` uses `.get` so v2 stays readable. Rolls the retrieval pods once (a backend merge; merge between generation batches per the deploy memory).
- **D3 Decision criterion = judge on the disagreement region + hand-labelled calibration slice.** Set overlap alone cannot rank J against B. Changes Richard's "that comparison is the entire decision" by one step — surfaced in Open items.
- **D4 Question shape: Noul-per-chunk for selection; Choice+none for extraction; threshold sweep, not a fixed θ.**
- **D5 Batch by measured tokens with a 0.75 safety factor (≤24k per state), never one pass; verify the per-request question cap and the tokenizer live before the bulk run.**
- **D6 Recompute the rendered set R with the real `_render_prefanout_budgeted`** so a Jev win is split into "inside what the gatherer saw" vs "outside it". No re-implementation.
- **D7 Extraction shadow measures what actually shapes the pool** (sort, doc_types, non-candidate anchors, sub_queries-only recall) — not "did it pick the right candidate", which the harness never acts on. Surfaced in Open items because it corrects the addendum's premise.
- **D8 Jev cannot write `sub_queries`.** The end state's happy path is raw-query-only fan-out; the 6.4% sub-query-only share is the number Phase 0 firms up. Phase 2's reformulation call is the only text generator left.
- **D9 Pure logic lives in `engine/retrieval/agent/jev_shadow.py` with tests in `tests/retrieval/agent/test_jev_shadow.py`** — that directory is already on the CI file list (`tests.yml:227`), so no list edit and no silent non-gate. The HTTP client is a thin module with a `httpx.MockTransport` test; the replay script stays in `scripts/`.
- **D10 Vendor route: TypeSafe direct (documented limits/pricing) over Cloudflare Workers AI (undocumented pricing).** Key handling: Phase 0 needs only a shell env var on the devbox; Phase 1's pod secret goes through the secrets source of truth, never `kubectl edit` (sync-secrets full-replaces the Secret). Needs Richard — Open items.
- **D11 Jev determinism check:** re-score 50 blobs 3× and report P(relevant) drift; "one search is never a measurement" applies to the pool (frozen here) and to the scorer (checked here).
- **D12 Managed-cluster blobs are a follow-up**, not a Phase 0 blocker: bucket access from this box is unproven (`r2-env` credential cannot list buckets; `probe-founders` bucket name lives in `pg-managed-1`/probe `customers.r2_bucket`). Phase 0 runs on research (3,600 blobs) and says so.
- **D13 Fix the stale `models.py:375-383` "parallel" comment in the capture PR.**
- **D14 (outside voice) Extraction is RETAINED in Phase 0 and Phase 1.** Jev is scored on the pool as built. The raw-query-only end state is measured separately: re-fuse the pool from `sub_query[0]` alone (the blob has per-sub-query hits) and report the delivered delta — rank shifts included, not just the 6.4% sub-query-only chunks. Zero-recall traces stay IN the extraction report (a wrong `doc_types` filter is one way to get zero recall).
- **D15 (outside voice) Compare delivered results, not raw score sets.** Arms A/B/C above, each cut to the same budget (10 distinct documents, the floor's unit), judged per document. θ is chosen on a tuning split and reported on held-out queries. Judge-vs-human agreement on the calibration slice below 0.8 INVALIDATES the gate; it is not a footnote.
- **D16 (outside voice) Units: documents.** The floor counts distinct documents; Jev scores chunks. Document score = max P(relevant) over its chunks; chunk-level numbers are reported only for span/why_relevant coverage. `B` is not "RRF top-10": it fills the slots the model left, may skip (`conditional` mode), and is deduplicated afterwards — arm B is reconstructed as the floor with an EMPTY model output, which is what 25/36 sampled traces already were.
- **D17 (outside voice) Historical fidelity needs the blob to carry it.** `_fuse_prefanout_docs` decays by `datetime.now(UTC)` and per-request `recency_half_life_days`; the rendered doc-id set and `recall_floor_mode` are not in v2 blobs. v3 persists `rendered_doc_ids`, `request_recency_half_life_days`, `request_recall_floor_mode`, and the fused doc order actually used. For v2 blobs the replay passes `ref_now = timestamp_utc` and flags every row as approximate; "recomputed B equals blob B" is a read-the-blob-right check, not a claim about R.
- **D18 (outside voice) Extraction question shapes:** one `Noul` per candidate (independent multi-select, not a single `Choice` — the extractor can pick several); `Choice` for `sort`; one `Noul` per `doc_types` class from the registry. Jev produces no non-candidate anchors — those come from grounding recall, a separate lever. Phase 0 measures labels and `via_entity` provenance; retrieval-under-replacement needs a live fan-out and is a Phase 1 A/B, said so in the report.
- **D19 (outside voice) Weak scores are necessary, not sufficient, for Phase 2.** High P on present chunks cannot see the missing chunk. Phase 0 reports the score distribution AND a judge answerability label per query ("does the delivered set answer it?"); the Phase 2 trigger is designed against the second, later. In-loop tool-call evidence (≈40% of sampled traces made one) is reported as a separate pool the pre-fan-out replay does not see.
- **D20 (outside voice) Capture BOTH terminal paths and the raw string.** `_parse_terminal_args` is called from the normal return (`loop.py:3696`) and the forced-termination path (`loop.py:3777`); stash `state.terminal_raw` inside `_parse_terminal_args` so no path is missed, and persist the raw argument STRING (capped, e.g. 256 KB) — malformed payloads are exactly the model-vs-parser cases the study needs.
- **D21 (outside voice) Order of work: quality test before vendor plumbing.** Arm A vs B needs no Jev at all and no capture PR — it runs on v2 blobs today with the judge. Do it first; it bounds what Jev can add before a client, an extraction replacement, or a blob bump is built.

## NOT in scope

- Phase 1 cutover (Jev replaces emission; delete the content-echo contract and `_coerce_lenient`'s drop path) — only after Phase 0 numbers.
- Phase 2 conditional reformulation — sized here (fire rate, cost), not built.
- `engine/retrieval/synthesis.py` — untouched by instruction.
- Auto-merge analyzer's Cerebras calls — unrelated caller, stays.
- A focused (Jev-picked) graph anchor — a Phase 1 design option the `via_entity` attribution makes measurable; not built.
- Managed-cluster replay (D12).
- Any change to the recall floor.

## What already exists (reused, not rebuilt)

- `trace_blob.py` + R2 layout + `trace_analyzer/loader.py` / `fetch_one.py` — blob fetch and decode; the replay reuses `fetch_one`'s decode path and rclone for bulk copy.
- `_render_prefanout_budgeted`, `_fuse_prefanout_docs`, `_all_prefanout_doc_ids` — deterministic pool/render/backfill order; imported, not copied. `_backfill_recall_floor` recomputed = self-test that the harness reads the blob the same way production did.
- `RecallFloorOutcome.rejected / unexamined` — the same split Jev's win must be reported in.
- `harness_appended` on `GatheredChunk` — the G/B split is already persisted.
- `agent.entity_extract_complete` log (entities with confidence, sub_queries, sort) — the extractor's pick for the ~30 h of pod logs that exist; joined on `trace_id` for a first look before v3 blobs accumulate.
- `tests/retrieval/agent/test_recall_floor_conditional.py` `_pool()` fixture shape — test fixture pattern for pools.
- `_seed_for_query` — extraction is seeded and `temperature=0`, so an offline re-run of the extractor reproduces its pick if the log join is insufficient (fallback, not the plan).

## Test plan (unit unless marked)

```
CODE PATHS                                                      COVERAGE
[+] engine/retrieval/agent/jev_shadow.py
  ├── pool_from_prefanout()          [GAP] dedupe by chunk_id, first content wins, skip non-dict/none-content hits, empty prefanout
  ├── rendered_set()                 [GAP] equals what _render_prefanout_budgeted fills for the same blob (pins D6)
  ├── delivered_split()              [GAP] G/B by harness_appended; missing flag = model (v1 blobs)
  ├── to_jev_batches()               [GAP] ≤ cap per state, order stable, a single oversized chunk → truncated not dropped, 0 chunks → 0 batches
  ├── select()                       [GAP] θ sweep, ties, missing answer for a chunk → excluded + counted
  ├── compare_sets()                 [GAP] all seven counts on a hand-built 3-set example; J∩(P∖R)
  ├── extraction_candidates()        [GAP] v3 field first; else parse <grounding> lines incl. "(no entities matched)" and bare_id
  └── backfill_recompute() self-test [GAP] equals blob's B for `ok` traces (REGRESSION guard on reading blobs correctly)
[+] engine/retrieval/agent/jev_client.py
  ├── happy path                     [GAP] MockTransport → answers parsed, per-chunk map
  ├── 429 / 5xx retry then give up   [GAP] bounded retries, exception NAME logged (httpx timeouts stringify empty)
  ├── timeout                        [GAP] raises typed error, never hangs the replay loop
  └── malformed body                 [GAP] typed error
[+] scripts/jev_shadow_replay.py     [→E2E, key-gated] one real blob end-to-end; skips loudly without TYPESAFE_API_KEY
[+] trace_blob.py v3                 [GAP] build_trace_blob carries terminal_raw/grounding/extraction; None-tolerant; v2 digest still reads
[+] loop.py capture                  [GAP] terminal args stashed on LoopState before _parse_terminal_args; grounding_bundle set on request.state
LLM: [→EVAL] judge prompt vs the 40-query hand-labelled slice — report agreement, not a pass/fail gate
COVERAGE today: 0/16 (new code)  |  all 16 are required in the Phase 0 PR  |  E2E: 1 (key-gated)  |  eval: 1
```

## Failure modes (new code paths)

| path | realistic failure | test | handled | user sees |
|---|---|---|---|---|
| Jev call | 429 storm / 5xx / timeout | yes | retry-then-typed-error, trace skipped and counted | replay report lists skipped traces — never silent |
| Jev call | state over 32k because chars/4 undercounted | yes (safety factor) + live probe | batch by measured count | — |
| Jev call | question cap lower than pool size | live probe first | split questions across requests | — |
| replay | blob v1/v2 lacks a field | yes | `.get` with defaults, counted as "unavailable" | report column |
| replay | `_render_prefanout_budgeted` drifts from the budget the blob ran under | self-test vs blob B | env has no override today; record the constant in results | — |
| capture PR | `terminal_raw` is large (chunk content echoed) | yes (size test) | store ids + why_relevant + spans, not content (content is in the pool already) | — |
| capture PR | blob write fails | existing (`persist_trace_blob_to_r2` return-or-log) | unchanged | — |
| judge | judge disagrees with humans | eval slice | reported agreement; no gate | — |
**Critical gap flagged:** none silent — every skip is counted in the report.

## Parallelization

| step | modules | depends on |
|---|---|---|
| A capture PR (blob v3 + grounding_bundle on gatherer path + stale comment) | engine/retrieval/agent/{loop,trace_blob,models}.py, tests/retrieval/agent/ | — |
| B pure replay logic + client + tests | engine/retrieval/agent/jev_shadow.py, jev_client.py, tests/retrieval/agent/ | — |
| C replay script + report + judge | scripts/ | B, key |
| D hand-labelled calibration slice | docs/ (40 queries) | — (Richard/Mahit) |
Lane A ∥ Lane B ∥ Lane D; C after B. A and B both touch `engine/retrieval/agent/` — separate files, no conflict expected; land A first so v3 blobs start accumulating.

## Implementation Tasks
- [ ] **T1 (P1, human ~1d / CC ~30m)** — jev_client — live contract probe: auth, max questions/request, tokenizer count for a 25k-char state, error shapes; record in `docs/jev-contract.md`. Surfaced by: Step 0 (nobody has called it). Files: docs/jev-contract.md. Verify: probe script output committed.
- [ ] **T2 (P1, ~2d / ~45m)** — trace capture — blob v3 (`terminal_raw` as the raw string from BOTH parse sites, `grounding`, `extraction`, `rendered_doc_ids`, `request_recency_half_life_days`, `request_recall_floor_mode`, fused doc order), `request.state.grounding_bundle` on the gatherer path, fix `models.py:375` comment. Surfaced by: facts 8, 1; D17, D20. Files: engine/retrieval/agent/loop.py:3696+3777 (`_parse_terminal_args`), trace_blob.py, models.py, tests/retrieval/agent/test_trace_blob.py. Verify: `pytest tests/retrieval/agent/` green; a live blob shows the three fields.
- [ ] **T3 (P1, ~3d / ~1h)** — jev_shadow — pure functions + tests (coverage list above), document-level aggregation (max chunk P), arm B reconstruction (floor with empty model output), `sub_query[0]`-only re-fusion (D14, D16). Files: engine/retrieval/agent/jev_shadow.py, tests/retrieval/agent/test_jev_shadow.py. Verify: pytest.
- [ ] **T4 (P1, ~1d / ~30m)** — jev_client — thin client on `typesafe-sdk`/httpx, MockTransport tests. Files: engine/retrieval/agent/jev_client.py, tests/retrieval/agent/test_jev_client.py. Verify: pytest.
- [ ] **T0 (P1, ~1d / ~40m)** — arm A vs arm B FIRST — today's delivery vs floor-only on v2 blobs with the judge, budget-matched, held-out split; no Jev, no capture PR (D21). Verify: report section with per-tenant precision + answerability.
- [ ] **T5 (P1, ~2d / ~1h)** — replay — `scripts/jev_shadow_replay.py`: rclone copy by tenant/date, exclusions, three arms at a 10-document budget, θ on a tuning split / reported held-out, determinism re-score, results.jsonl + report.md. Verify: run on 09-14..09-22 research blobs.
- [ ] **T6 (P1, ~1d / ~30m)** — judge — per-document relevance + per-query answerability on gemini flash-lite via the gateway; agreement vs the calibration slice; agreement < 0.8 invalidates the gate (D15, D19). Files: scripts/jev_shadow_judge.py. Verify: report section.
- [ ] **T7 (P2, ~0.5d / ~20m)** — extraction stage in the replay (Noul-per-candidate vs the extractor's picks, `sort` Choice, `doc_types` Nouls, non-candidate anchors via `via_entity`, `sub_query[0]`-only delivered delta; zero-recall traces included) (D14, D18). Files: jev_shadow.py, replay script. Verify: report section.
- [ ] **T8 (P3, ~0.5d / ~15m)** — managed blobs — locate `probe-founders` bucket + credential path; extend the replay. Verify: blob count from managed.

## Open items for Richard (the only two)

1. **Jev key + vendor route.** TypeSafe direct (recommended: documented limits and pricing) or Cloudflare Workers AI (`typesafe/jev`; team already has the CF account; pricing not published). For Phase 0 the key is a devbox env var only. Nothing can be scored until this exists; T2-T4 proceed without it.
2. **The extraction premise.** Today the extractor's candidate pick never narrows the pool (every grounding candidate is anchored, p50 50). The addendum's "scoring the wrong pool perfectly" happens only through `sort`, `doc_types`, extra anchors and `sub_queries`. Phase 0 measures those; a *focused* anchor (Jev picks, harness anchors only on the pick + its neighbours) is a Phase 1 design option, not the status quo being replaced.

## Outside voice (Codex, `gpt-6-astra`, read-only on the repo) — 8 findings, all folded

1. Replay scores a pool the extractor already improved; removing a sub-query shifts fusion ranks; zero-recall exclusion hides filter failures → **D14**.
2. Production delivers Jev ∪ floor, not `J_θ`; judge the delivered set at a fixed budget on held-out queries; a judge that fails calibration invalidates the gate → **D15**.
3. `B` is not RRF top-10; it fills remaining DOCUMENT slots, may skip, and is deduplicated; Jev scores chunks → **D16**.
4. `_fuse_prefanout_docs` uses `datetime.now` + per-request recency; rendered ids not in the blob; recomputing B proves nothing about R → **D17**.
5. Extraction replacement unspecified (multi-select, doc_types, non-candidate anchors); T7 measures labels not retrieval → **D18**.
6. Weak scores cannot see missing evidence; in-loop tool evidence is outside the pre-fan-out → **D19**.
7. Forced-termination path (`loop.py:3777`) also parses terminal args; store the raw string → **D20**.
8. The cheapest alternative — drop the gatherer call, keep extraction, floor-only delivery — is missing; test it before building vendor plumbing → **D21, arm B, T0**.

Codex's closing line: "Narrow Phase 0 to a held-out, budget-matched comparison of Jev against deterministic backfill with extraction retained." Accepted.

CROSS-MODEL TENSION: none left open. Finding 8 reframes what Phase 0 is for (is Jev needed at all, or just the floor?) — accepted as arm B and T0 rather than argued.

## Suppressed findings (confidence < 7)
- LiteLLM SDK `timeout` kwarg may be forwarded in-body rather than applied client-side (5/10) — matters only for the "14 s hang" argument, which Jev moots either way.
- `customer-c751dbfa…` big calls on managed could include a non-gatherer caller (4/10) — attribution by prompt size; changes the gatherer share by at most the 4,987 localhost calls ($68).

## GSTACK REVIEW REPORT

| Run | Status | Findings |
|---|---|---|
| Step 0 scope | scope changed: replay not live hook; capture PR added; extraction measures pool-shaping outputs | 3 |
| Architecture | 21 decisions folded (D1-D21) | 6 |
| Code quality | reuse over re-implementation (D6, D9); stale comment (D13) | 3 |
| Tests | 16 paths, all required; 1 E2E key-gated; 1 eval | 16 gaps → tasks |
| Performance | Jev limits vs our peak: trivial; batching by measured tokens (D5) | 1 |
| Outside voice | Codex completed (read-only, repo-grounded); 8 findings, 8 folded (D14-D21) | 8 |

VERDICT: CLEARED for Phase 0 as specified in this document — arm A vs B first (T0), then Jev; no cutover until the report exists.
OUTSIDE COVERAGE: completed — provider codex, host claude, 8 findings folded. CROSS-MODEL: no open tension.

**UNRESOLVED DECISIONS:**
- Jev key / vendor route (Open item 1) — blocks scoring, not building.

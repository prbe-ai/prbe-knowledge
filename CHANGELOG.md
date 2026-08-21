# Changelog

All notable changes to the Probe knowledge engine are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **0118's backfill inserted nothing, and reported success.** Its `ALTER TABLE
  ADD COLUMN` landed (DDL is not row-filtered) and both data statements were
  silently filtered to zero rows. Production sat at head `0118` with
  `situations.classifiable` present and not one `misc` row in the database.

  `situations`, `clauses` and `clause_situation_edges` all carry FORCE ROW LEVEL
  SECURITY, and migrations run as `app` — the table OWNER, but not a superuser
  and without BYPASSRLS. FORCE is exactly the flag that subjects an owner to the
  policy. With no `app.current_customer_id` bound, the policy compares against
  NULL, so `SELECT DISTINCT customer_id FROM situations` returns nothing and the
  INSERT consuming it writes nothing. No error, no warning.

  The tests could not catch it: the local and CI role is a SUPERUSER and bypasses
  RLS, so the backfill worked perfectly under test, and the drift guard compares
  schema rather than rows. 0119 repairs it by binding the tenant GUC per tenant,
  driving the loop from `customers` (deliberately not row-secured).

  **The rule this establishes: a data migration touching an RLS-FORCED table must
  bind the tenant GUC per tenant.** There is no ambient admin here. The first
  draft of this very repair asked `WHERE EXISTS (SELECT 1 FROM situations ...)`
  in its tenant query and was therefore empty for everyone — the same bug wearing
  a different hat — and the new test under the non-superuser `prbe_rls_test` role
  is what caught it.

### Fixed

- **The classifier's LLM tie-break never once worked in production.**
  `_TIEBREAK_MAX_TOKENS` was 256 -- generous for a reply that is ~15 tokens of
  JSON, and nowhere near enough for a THINKING model, whose reasoning is billed
  against the same budget before it emits a single visible character. Measured
  against the live model:

      max_tokens=256   completion_tokens=252   '```json\n{\n  "slug": null'
      max_tokens=2048  completion_tokens=533   '{"slug": null, "confidence": 0.0}'

  So every ambiguous classification truncated, failed to parse, degraded to
  `unknown`, declined to attach a situation, and left the clause unreachable --
  the upstream cause of the orphaned-rule bug the `misc` bucket below exists to
  catch. Nothing errored; the log line said `tiebreak_unusable_response`, which
  reads like a flaky model rather than a budget we set.

  Now 2048, sized for the reasoning rather than the answer. Turning thinking off
  would be cheaper and is MODEL-SPECIFIC -- it would stop applying the day
  somebody swaps the model in `constants.py`, which is the kind of coupling that
  produced this. Code fences were investigated and exonerated: the parser strips
  them, and there is now a test pinning that so the next person does not go
  looking there.

### Added

- **A `misc` situation, so a rule nobody could classify is not unreachable.**
  `declare` attached a situation only when it HAD one, and the classifier
  honestly answers `unknown` for prose it cannot place. Those two correct
  behaviours composed into a silent loss: the clause was written, got a real id,
  returned 200, appeared in an unfiltered `probe rule list` -- and was invisible
  to every situation-scoped read, which is the only read this store exists to
  serve. Production hit it on the very first rule anyone declared: it was plainly
  about opening a pull request, and asking for `open-pr` rules returned nothing.

  A declaration with no situation now lands in `misc`, and says so
  (`situation_fallback`) rather than reporting a filing that did not happen. The
  bucket is EXCLUSIVE: giving a clause a real situation later evicts its `misc`
  edge, so the bucket keeps meaning "still has no home" instead of filling with
  rules that already found one.

  `misc` is NOT a thirteenth classifier label, and the new `situations.
  classifiable` column is what keeps it out of the label space. The reason is
  mechanical: the classifier embeds each description, and "anything that does not
  fit the others" has no situation in it to embed -- so as a label it either
  matches nothing (bucket unreachable) or weakly matches everything and starts
  winning near-floor ties against real situations, filing rules under misc BY
  classification rather than for want of a home. A column rather than a reserved
  slug because `situations` is per-tenant and editable, so a rule keyed on the
  literal string breaks the moment somebody renames it.

  Serving offers the bucket in exactly one case: a situation-scoped read that
  matched NOTHING. Topping up a partial result was the tempting version and is
  wrong -- at limit=3 it would spend a third of the agent's attention on rules we
  know do not apply, diluting an answer that was already correct. Bucket cards
  carry `from_fallback` so a reader can tell "your team decided this" from
  "nothing fit, here is an unfiled rule"; render them alike and the safety net
  becomes a source of confident irrelevant advice.

  Migration 0118 backfills the bucket for every tenant that already has a
  vocabulary and adopts the clauses that were already stranded.

### Fixed

- **Workflow memory's write path never worked in production, and the log line
  could not say why.** Every `/procedures/preview` and `/procedures/declare`
  502'd from the day the feature shipped. `WFMEM_STRUCTURING_MODEL` named
  `anthropic/claude-sonnet-4-6`, and the research-os cluster's LiteLLM gateway
  serves Fireworks, Cerebras and `gemini-*` — no Anthropic deployment at all —
  while the engine has no `ANTHROPIC_API_KEY` to go direct with. Both spellings
  fail and they fail differently: the prefixed id 404s (an Anthropic-native
  request against an OpenAI-shaped proxy) and the bare id gets "no healthy
  deployments for this model". The comment claiming a `claude-*` modelList entry
  was true of the *prbe-backend* proxy, a different deployment.

  Default is now `gemini/gemini-3.6-flash`, verified reachable against the live
  gateway rather than inferred, and env-overridable per the
  `SEARCH_AGENT_INFERENCE_MODEL` convention. **This is a deliberate downgrade**
  from the frontier-class model the original note argued for — the structuring
  pass has to refuse to embellish, and a polished invention is not caught
  downstream because people approve what looks right. Restoring one is a values
  change once that gateway serves it, not a release.

  The structuring failure log now carries `status_code` and `provider` (the
  classifier's tie-break too). It carried only the exception class, so a 404
  from the gateway — a config bug fixable in a minute — was indistinguishable
  from a provider outage, and diagnosing it took a shell in the pod and a dump
  of the gateway's model list. `str(exc)` stays out: the request here is
  unscanned prose a human pasted, and a provider's message can quote it back.
  `status_code` and `provider` come off the HTTP layer and cannot.

### Added

- **Workflow memory: a store for the team's procedure rules, off by default.**
  A person declares a rule once and their teammates' coding agents can be handed
  it when they are in the situation it applies to. Five tables (`situations`,
  `clauses`, `clause_situation_edges`, `clause_evidence`, `serve_ledger`), each
  with `FORCE ROW LEVEL SECURITY` and composite `(customer_id, id)` foreign keys
  — simple ones would let a tenant probe for another tenant's clause ids, since
  Postgres RI checks bypass row security by design.

  Three endpoints on the retrieval service: `/procedures/preview` structures the
  prose, classifies it and looks for near-duplicates without writing anything;
  `/procedures/declare` writes what a human confirmed; `/procedures/query`
  serves. A fourth, `/procedures/publish`, puts an existing rule in front of the
  team or withdraws it.

  Every cell is gated per-tenant and fails closed, and each response carries the
  three-state `enabled` / `entitled` / `upgrade_url` envelope rather than a
  status code — "nobody turned it on" and "the plan does not include it" are
  different answers, and a client that cannot tell them apart reports the switch
  as broken. Nothing is reachable until a tenant is explicitly enabled, which is
  also what seeds their twelve situations.

  Three properties are load-bearing and fail silently if broken, so each has
  direct tests: evidence is stored BY REFERENCE (there is no quote column, and a
  CHECK refuses quote-shaped keys in `source_ref`), a credential-shaped rule is
  refused at ingest across every author-supplied field, and a clause backed by
  only one human stays private to that human until a second, untainted human
  appears. That last rule has an escape hatch — a named person can publish on
  their own authority — because without one a lead declaring twenty existing
  team rules produces twenty clauses nobody can see. Published rules are
  labelled with who published them and how many humans back them, never
  laundered into looking corroborated.

  Every delivery that returns rules writes one `serve_ledger` row in the same
  transaction as the read. Exposure is unbackfillable, so a delivery that wrote
  no row is not a gap that can be repaired later.


### Fixed

- **Entity merges that hit a duplicate edge lane no longer roll back silently.**
  `merge_cluster`'s collision handler runs inside `with_tenant`'s explicit
  transaction, so the `UniqueViolation` it catches had already aborted that
  transaction — the recovery `INSERT` died with `InFailedSQLTransactionError`
  and the whole merge was thrown away. Signature on the managed plane: 88
  duplicate-key errors in 24h paired 1:1 with 88 "current transaction is
  aborted", and not one completed merge. The UPDATE is now wrapped in a
  SAVEPOINT, and the handler matches `graph_edges_unique_lane` by name rather
  than catching any unique violation. Letting the path run for the first time
  exposed a second layer: it writes `operation = 'deleted_duplicate_lane'`,
  which the column's CHECK never permitted (migration 0113 widens it).
- **`system_settings` exists on every plane again, so the global ingestion
  killswitch works.** A database bootstrapped from `schema.sql` + `alembic stamp
  head` never replays the chain, so the table migration 0025 created was absent
  from managed-shared — 4,271 errors/24h — and its reader fails open, meaning
  the master switch for halting all plugin ingestion silently did nothing there.
  Migration 0112 repairs it; `scripts/check_schema_drift.py` makes the next
  instance loud.
- **Cross-source vector similarity actually runs.** `inferred_edges/bundle.py`
  named a CTE `similar`, a reserved word, so the statement was a syntax error and
  the query had never executed once in production — 752 failures/24h, swallowed
  by a bare `except Exception` that logged at DEBUG and blamed a missing
  embedding. Renamed, and the handler now catches `asyncpg.PostgresError` and
  logs at WARNING with the real error type.
- **`neon_auth` lookups stop erroring on managed** (1,514/24h). Migration 0112
  grants the app role USAGE + SELECT, guarded so it no-ops where the schema is
  absent. This silences the error; it does not make the feature work — the table
  is an empty shim and every active managed customer has a NULL
  `organization_id`. See TODOS.md.

### Changed

- **The /knowledge stats header no longer takes seconds to answer.**
  `GET /api/stats/ingestion` ran four uncached aggregates, and neither of the
  two that matter had an index behind it. Measured on the research plane for a
  19.5k-document tenant: the document count took 5,187 ms cold and bitmap-scanned
  10,494 heap pages to answer for 19,573 rows; the chunk count took 2,936 ms and
  touched 148,808 buffers, 77% of them a Memoize'd nested loop into `documents`
  resolving `source_system` one chunk at a time. Migration 0111 adds a partial
  covering index to each side, so both aggregates are now Index Only Scans and
  the same tenant costs 28 ms and 92 ms. The join to `documents` in the chunk
  count is deliberately kept: ~3% of live chunks belong to a superseded or
  soft-deleted document and must not be counted.
- **The same endpoint is cached 30s per tenant and runs its four queries
  concurrently.** `?refresh=true` bypasses the cache; the dashboard's Refresh
  button sets it. The four queries now use a connection each, so the four
  results are no longer one consistent snapshot — safe for display counters,
  documented at the call site, and not a pattern to copy elsewhere. The cache
  lock is per tenant, so one slow tenant cannot stall another's header.
  The `refresh` bypass only reaches here once research-os relays it
  (research-os v0.211.0.0); until that ships, Refresh returns the cached
  payload for up to 30s. Order of deploy does not matter — FastAPI ignores the
  unknown parameter in one direction and defaults it to false in the other.

- **The rebuild button can no longer empty a wiki that nothing can
  rebuild.** `POST /api/wiki/backfill/trigger`, its `/bootstrap/trigger`
  alias, `POST /api/wiki/synthesize/trigger` and `PUT /api/wiki/settings`
  answer 410. Wiki generation was switched off across the fleet on
  2026-08-18, which removed the triage, synthesis and backfill workers —
  but a rebuild retires every compiled page *before* re-crawling, so the
  wipe still ran and the rebuild never arrived. research-os refuses its own
  `POST /v1/wiki/rebuild`, and the dashboard reaches this API through the
  prbe-backend BFF without passing through research-os, so this refusal is
  the one that closes that door.
- **The nightly wiki trigger is gone** from `knowledge-cron.yml`. It was
  the 02:00 UTC `pg_notify` that woke the worker app; with it and the
  manual synthesize trigger both closed, nothing can wake synthesis.
- Still working, deliberately: every read, page writes (unreachable from
  outside now that research-os refuses them), the DLQ reset (an operator
  lever that only moves rows between statuses), and
  `POST /api/wiki/backfill/undo` — the recovery path, and the only way home
  for a tenant a rebuild already stranded.

### Added

- **`search_knowledge` grew a `detail` dial — and its default response got
  leaner without losing a fact.** Measured on live production responses,
  60% of a search payload was structure and metadata rather than evidence,
  and because responses run into the ~20KB byte budget (9 of 10 measured
  calls truncated), that overhead was being paid for with dropped search
  results, not just tokens. Two changes:

  - Compaction now collapses measured repeats unconditionally: chunk-level
    `retriever_scores` (empty in 69/69 production chunks), `canonical_id`
    when byte-identical to `doc_id` (69/83), a chunk's `matched_via` when it
    duplicates its document's (19/69), empty `graph_evidence`/`why_relevant`,
    and null-valued keys inside provenance entries. The empty-value collapses
    pass populated values through untouched; chunk `retriever_scores` is
    deny-listed outright like its document-level twin (`verbose=True` keeps
    it). Absent keys mean "nothing here", never "unknown" — with one scoped
    exception: a chunk's `matched_via` omitted as identical to its
    document's is covered by the document's copy.
  - `detail: "ids" | "evidence" | "full"` (default `"evidence"`) projects
    each result row: `evidence` keeps identity + chunk content and drops
    per-doc audit metadata (`author_id`, timestamps) and non-graph
    provenance; `ids` is the no-content triage shape; `full` is the prior
    compact response. The envelope (`degraded`, `truncated`,
    `confidence_breakdown`, `total_candidates`, ...) is identical at every
    detail — a leaner answer can never masquerade as a healthier one — and
    profiles run before the byte budget, so leaner details keep more hits
    under the cap. Replayed against the captured production corpus:
    ~−26%/call at the default, −71% at `ids`; `verbose=True` still returns
    the raw upstream payload untouched. `query_knowledge`'s evidence rows get
    the same redundancy collapse (it has no `detail` parameter; its rows match
    search's detail="full"), and the byte budget's trim stage now POPS
    `graph_evidence` from tail chunks instead of writing `[]` — an empty list
    is a shape the new contract says cannot exist, and emitting it would have
    read as "no graph grounding" on a chunk that had some.

- **The wiki now covers documents that arrived before a tenant turned it
  on.** The nightly trigger reconciles the synthesis queue for every
  enabled tenant (batched, idempotent, timestamp-agnostic — backdated
  bulk imports are covered), and flipping `wiki_generation_enabled` on
  seeds the tenant's existing corpus immediately in the background. The
  Normalizer's enqueue-failure swallow finally has the safety net its
  comment always claimed.
- `GET /api/wiki/backfill/preview`: read-only counts (pages the wipe
  deletes, documents a rebuild re-derives) for the rebuild confirmation
  dialog. The trigger response now carries `eligible_documents` /
  `seeded` / `reset`, and `PUT /settings` carries `catchup_started`.
- `WIKI_AGENT_GLOBAL_CONCURRENCY` is env-overridable (validated; the
  only real synthesis throughput knob — replicas are a no-op under the
  per-customer advisory locks).

### Changed

- **The wiki no longer has `project` or `person` pages.** Both restated
  what the platform already holds — projects, experiments and runs are
  live entities in research-os, and authorship, review and ownership are
  graph edges the ingestion pipeline maintains continuously — so each
  page was a second copy that went stale silently with no way for a
  reader to tell which copy was current. `decision` stays: why X was
  chosen over Y exists in no system of record. Migration 0107 retires
  (does not delete) every page that still carries one of the two kinds,
  so their bodies and version history remain on disk. `[[person: X]]`
  LINKS are unaffected — a wiki link points at a graph entity, and
  several kinds it can name have never had pages of their own.
- Migration 0108 redoes 0107's retire per customer. `documents` carries
  FORCE ROW LEVEL SECURITY, which subjects even the table owner to the
  tenant policy, and a migration binds no `app.current_customer_id` — so
  0107's single unscoped UPDATE matched zero rows and reported success on
  any deployment whose migration role lacks BYPASSRLS. It applied on
  managed-shared (role `probe`, bypasses) and silently did nothing on
  research-os's kb hook (role `app`, does not). Any future migration
  writing `documents` globally needs the same per-customer loop.
- Undo of a wiki rebuild never restores a page whose kind has since left
  `WikiType`. The time bound alone did not cover a taxonomy change, and a
  live page carrying a retired kind is unreachable by construction.
- The GitHub backfill crawler shares the real `wiki_type` tool schema
  instead of its own copy. The copy had drifted into a free-form string
  inviting the model to invent a kind "if the corpus calls for it" —
  kinds the crawler's own argument validator has refused since the set
  was closed, so each one cost a turn and was then rejected.
- "Rebuild wiki" reseeds the daily pipeline in the same transaction as
  the wipe: seed missing queue rows + reset terminal rows for live
  eligible docs, so pages derived from sources with no crawler
  (transcripts, custom ingest) come back instead of being lost. dlq rows
  stay parked unless the catchup CLI's `--include-dlq` redrives them.
- `scripts/wiki_synthesis_catchup.py` is a thin CLI over
  `kb.synthesis.persistence`; `--dry-run` now reports the REAL
  would-insert/would-reset split (it previously always printed
  `already_queued = eligible`).

### Fixed

- A tenant enabled through the engine's own `PUT /settings` (which
  writes the JSONB **string** `"true"`) was ON for every SQL consumer
  and OFF for the Python gate — the Normalizer silently stopped
  enqueueing their new documents. Both sides now accept the same two
  shapes, pinned by a parity test.
- The catchup seed excluded only `wiki`, not `code_graph`, so catchup
  runs mass-enqueued AST-extraction docs the pipeline deliberately
  skips. All seed paths now consume `WIKI_ENQUEUE_EXCLUDED_SOURCES`.


### Changed

- The gatherer now cuts a stalled Cerebras turn at **5s** (was 70s) and
  finishes the run on Fireworks, sticky for every remaining turn. Cerebras
  latency is bimodal — measured over 105 production turns, 87.6% land at
  mean 971ms (p90 1.6s, max 3.9s) and 12.4% stall at 59.5-63.8s with nothing
  in between — so a 5s deadline separates the two modes cleanly without
  truncating healthy traffic. Waiting the stall out was the old strategy; at a
  mean 2.23 turns/retrieval a 12.4% per-turn stall compounds to ~30% of
  retrievals degrading, and research-os abandons /v1/search at 30s, so a 60s
  "success" was already an empty result set for that caller. The gateway's own
  Cerebras -> Fireworks route fallback provably never fires (stalled turns
  carry `x-litellm-attempted-fallbacks: 0`), so the loop owns the hop:
  `agent.provider_failover` records it. Set
  `SEARCH_AGENT_FALLBACK_INFERENCE_MODEL=""` to disable failover on
  single-provider installs.
- `SEARCH_AGENT_LOOP_TIMEOUT_SECONDS` is now **60s** (was 90s) and bounds the
  WHOLE gatherer stage, not just the loop: grounding + extraction +
  pre-fan-out (~4s) are subtracted from it rather than added to it, so the
  number is a ceiling a caller can size its own HTTP timeout against. On
  expiry the stage still returns the deterministic pre-fan-out evidence via
  `_backfill_recall_floor` — a real result set, not an error.

### Fixed

- `sources` is now an applied filter on `/retrieve` and `/query`. It was
  accepted and enum-validated on `QueryRequest` but never threaded into the
  gatherer path — `execute_search` had no `sources` parameter at all, so the
  only consumer was `list_pipeline`, which is dead for `/retrieve` post-
  cutover. Live proof: `sources=["claude_code"]` returned 10 github docs and
  zero transcripts. The loop now carries `request_sources` into the pre-fan-out
  and injects it into every in-loop `search` dispatch (same contract as
  `source_keys` / `doc_types`: the agent reformulates queries but cannot widen
  the caller's scope), all four channels filter on `d.source_system` before the
  LIMIT, and the adapter's scope gate re-verifies it. Responses now echo
  `applied_sources` so a dropped filter is visible instead of silent.
- Keyless tolerance now reaches the scope GATES, not just the channels.
  `source_keys_include_keyless` admitted connector docs (github, claude_code)
  into the retrieval channels while `_doc_scope_sql` and the adapter's
  `_enforce_scope_on_chunks` kept applying the hard filter, so the agent
  retrieved github hits it was then refused permission to read
  (`agent.fetch_doc_scope_refused`) and every one was dropped at the response
  choke point (`dropped=10, kept=0`). A mixed keyed+keyless request therefore
  returned an empty result set with a non-zero candidate count, and the loop
  reported `degraded=true, degraded_reason="schema_violation"`.
- Full-source reads now remove standard 64-token chunker overlap while
  provenance-gating pre-chunked `code_graph` rows so their boundaries remain
  intact. Reconstruction also repairs Unicode replacement characters created
  when a token window bisects UTF-8, keeps source-view line spans aligned with
  the reconstructed body, and runs off the async request loop.
- `per_source_top_k` no longer loses a whole corpus to the final `LIMIT`. The
  window function gave each `source_system` its own slot budget and the query
  then applied `ORDER BY score DESC LIMIT top_k`, handing every slot back to
  whichever source scores highest in the absolute — undoing the guarantee one
  line after computing it. Measured on the research cluster with
  `per_source_top_k=20, top_k=30`: the response contained
  `{github: 4, claude_code: 20, code_graph: 6}` and **no `custom_ingest` at
  all**, which first appeared at rank 61. Scores are not comparable across
  sources — a terse structured projection never out-scores a chatty transcript
  on a natural-language query — so the final ordering now leads with the
  per-source rank, interleaving sources instead of letting one sweep the limit.
  Callers that do not pass `per_source_top_k` keep a straight score ranking.

### Added

- BM25 matches document TITLES, not only chunk content (migration 0099). The
  title was unreachable by keyword search: `chunks.content_tsv` is a GENERATED
  column, and a generation expression may only reference its own row, so it can
  never reach `documents.title` — the two are in different tables. A file named
  `model.ckpt`, or a PR titled "Fix the retry loop", was findable by keyword
  only if those words also appeared in the body. Titles are now indexed in a
  `documents.title_tsv` generated column, weighted `A` against unweighted
  content so a title hit outranks a body hit under `ts_rank_cd`.

  The title is indexed twice over — verbatim, and with path punctuation
  flattened to spaces. Postgres' `english` parser emits `model.ckpt` as a
  single `file` lexeme while the retriever splits queries on alphanumeric runs,
  so a verbatim-only index would have missed exactly the case this exists for.

  Title-only matches return one representative chunk (`chunk_index = 0`). A
  title belongs to the document rather than any chunk, so matching without that
  restriction would return every chunk of the document at an identical score
  and a long document would bury everything else. Chunks matching on their own
  content are unaffected.

- Per-turn `finish_reason` and `usage.completion_tokens` are recorded on the
  gatherer loop state, logged on `agent.turn_complete`, carried in the trace
  blob (schema v2), and summarised in the trace digest as
  `completion_tokens_max` / `was_truncated`. The gatherer emits each chunk's
  content verbatim, so emit size scales with results and cannot be inferred
  from a synthetic probe — this is the measurement needed before capping
  `max_tokens`. `finish_reason` is also the truncation alarm whose absence
  from the extractor's failure logs once kept a blown token cap invisible.
- `LLM_TPM_BUDGET` / `LLM_TPM_MAX_WAIT_SECONDS`: an optional per-process
  token-rate limiter in `engine/shared/llm.py`, covering every provider call
  (`import litellm` appears in exactly one engine file). Limits tokens rather
  than requests, because the quota is denominated in tokens and this engine's
  requests differ by ~50x. **Disabled by default** (`0`), and it **fails
  open** — if budget is not available within the max wait it proceeds and
  lets the provider decide, because blocking longer than the caller's own
  deadline turns a fast 429 into a slow timeout that returns nothing.
- Internal ingestion stats can now break Claude Code and Codex totals down by
  device, including historical derived documents linked through their parent
  session, so downstream dashboards can show trustworthy per-laptop counts.
- `POST /api/github/connect` (X-Internal-Knowledge-Key gated): seeds a GitHub
  App installation (customer_source_mapping + `installation:<id>` token row,
  validated by a dry-run mint) and enqueues its historical backfill, which the
  deployed BackfillWorker drains. This is the invokable equivalent of the
  `scripts.github_seed_token` + `scripts.backfill` runbook, so a downstream
  consumer (research-os) can backfill a repo the moment a team claims the
  installation instead of running the manual runbook. The seed logic is
  extracted to `kb.github_seed.seed_github_installation`, shared by the
  endpoint and the CLI.

### Changed

- The turn-1 pre-fan-out dump is rendered as compact JSON instead of
  `indent=2`. Indentation is input tokens carrying no information, re-sent on
  every turn, and it was un-budgeted: the budget counts `content` only, so
  whitespace rode on top of the cap rather than inside it.
- `SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET` lowered 40,000 → 18,000 and made
  env-overridable. This is the highest-leverage knob on provider quota
  because the turn-1 dump is charged on every turn, while a completion is
  charged once. Cerebras enforces 250,000 tokens-per-minute at the
  **organization** level — shared by every deployment of this engine — and
  reserves `input + max_completion_tokens` before running a request, which
  admitted only ~2-3 concurrent searches before `429 token_quota_exceeded`.
  This is a payload cap, not a recall cap: `SEARCH_AGENT_VECTOR_TOP_K` and
  `_BM25_TOP_K` are deliberately unchanged, so the candidate pool and the
  recall ceiling are untouched and only the post-fusion payload shrinks.
  Measured in production, ~280 candidates were rendered to return 10-16
  results.
- Query synthesis now advertises Gemini 3.6 Flash and Gemini 3.5 Flash Lite,
  keeps the previous picker IDs as compatibility aliases, and uses each new
  model's supported thinking level without retired sampling controls.

### Fixed

- Entity extraction no longer silently returns nothing on reasoning models.
  `max_tokens` bounds reasoning *and* visible content together, so the old
  600-token cap was consumed by the reasoning trace and truncated the JSON
  mid-object; the caller swallowed the parse error and searched with zero
  entity anchors while still reporting `state:"ok"`. Raised to 2000 and made
  it env-overridable, since the right value follows the model.
- Extraction failure logs now carry `finish_reason` and `content_len`, which
  distinguish "the model emitted bad JSON" from "we cut the model off".
- Retrieval trace ids no longer collide. They were a bare millisecond
  timestamp, so the several concurrent `/retrieve` calls a single search fans
  out into shared one id — making interleaved log lines unattributable and
  letting their R2 trace blobs overwrite each other at the same key.
- Claude Code session extraction now sends gateway-configured model aliases
  over the proxy's OpenAI-compatible transport while preserving direct
  Anthropic routing, preventing gateway URLs from becoming
  `/v1/v1/messages` and leaving finalized sessions in the ingestion DLQ.
- Retrieval queries now return citable pre-fan-out evidence with low
  confidence when the gateway exhausts its providers on a transient timeout,
  rate limit, or server error. Permanent provider failures and responses
  without citable evidence continue to fail closed, while phase-specific
  deadlines keep the fallback inside the MCP request envelope.
- Gatherer responses now honor `top_k_related`, including returning no related
  entity payload when callers set it to zero.
- Gateway-routed retrieval entity extraction and gatherer turns now make a
  single client attempt, so a failed provider chain is not replayed after the
  LiteLLM gateway has exhausted its configured failover routes. Direct
  self-hosted provider calls retain their normal transient retries.
- Probe MCP retrieval calls now preserve upstream and transport diagnostics,
  use phase-specific HTTP timeouts with enough read headroom for the search
  agent, and return failures with the MCP error flag plus a trace ID.
- Large MCP retrieval responses now emit one compact JSON payload, enforce a
  24 KB hard wire limit, and reject oversized `get_source(mode="full")` calls
  with bounded-mode guidance instead of allowing responses up to the retrieval
  service's 100 MB ceiling.

## [0.1.0] - 2026-06-18

Initial public release of the open-source community edition.

### Added

- Self-hosted, single-tenant knowledge engine: ingestion, worker, retrieval,
  synthesis, and MCP services.
- Source connectors for GitHub, Slack, Linear, Notion, and Sentry, plus a
  custom-ingest API.
- Hybrid retrieval (vector + BM25 + graph) fused via RRF, exposed as raw chunk
  retrieval (`/retrieve`) and LLM-synthesized cited answers (`/query`).
- Knowledge-page synthesis (optional `cron` profile).
- MCP server exposing `search_knowledge`, `query_knowledge`, and `get_source`
  (static and OAuth auth modes).
- Turnkey self-hosting via Docker Compose and a community Helm chart, backed by
  Postgres (pgvector) and S3/R2/MinIO object storage.

[Unreleased]: https://github.com/prbe-ai/prbe-knowledge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/prbe-ai/prbe-knowledge/releases/tag/v0.1.0

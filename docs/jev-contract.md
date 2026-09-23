# Jev (TypeSafe) — the contract, as measured

Probed live on 2026-09-22 from prbe-devbox with `scripts/jev_shadow/probe_contract.py`.
Server model id: **`jev-1.13.0`**. Raw probe output: `t1-probe.jsonl` (scratchpad).

Everything below is MEASURED. Where a published figure disagrees with a measured
one, the measured one governs — the vendor's docs describe a "32k token state"
and "255 options", and both turn out to mean something narrower than they sound.

## The four numbers the batcher needs

| | measured | what the docs implied |
|---|---|---|
| Questions in ONE request | **≥ 300, no cap reached** (300 Nouls answered, 284 ms) | "multiple"; the 255 figure is options-per-Choice, NOT questions-per-request |
| Token cap | **~32k TOTAL**, request-level. 29,415 passed; the next rung (131,348 chars) returned `400 {"detail":{"error_type":"max_tokens_exceeded"}}` | "64k request, 32k state" — there is no separate 64k headroom to spend |
| Cost of a question | **~35 input tokens per Noul**, on top of the state | not documented |
| Chars per input token (state) | **~3.77** on our corpus (3.60 at 40k chars, 3.77 at 110k) | n/a |

So the size rule is:

```
tokens ≈ state_chars / 3.77  +  35 × n_questions        (cap ≈ 32,000)
```

The question overhead is the part a chars/4 estimate misses entirely: at 100
chunks it is ~3,500 tokens, more than a tenth of the budget.

### What that means for our pools

Measured pool sizes (39 trace blobs): median 97.5 unique chunks / ~98k chars,
p90 121 chunks / ~130k chars.

| pool | state tokens | question tokens | total | fits one request? |
|---|---|---|---|---|
| median (98 chunks, 98k chars) | 25,995 | 3,430 | **29,425** | just — no margin |
| p90 (121 chunks, 130k chars) | 34,483 | 4,235 | **38,718** | **no** |

**Batching is mandatory, not a contingency.** Budget 24,000 tokens per request
(0.75 of the cap) and split by measured size, never by chunk count: roughly 70%
of searches go in one request, the rest in two. A batch boundary is safe here
because each Noul is independent — unlike a Choice, whose probabilities must sum
to one across the whole option set.

## Jev is NOT deterministic

Three byte-identical requests (30 chunks, 30 Nouls), same `input_tokens` all
three times:

```
max drift  0.060      mean drift  0.029      identical: false
```

This is the single most consequential finding for Phase 1's design, and it cuts
against the obvious threshold implementation:

- **A hard threshold θ is a coin flip for any chunk within ±0.03 of it.** A chunk
  at 0.71 against θ=0.70 is not reliably selected; re-run the same search and it
  may vanish. The team already has the scar for this — "one search is never a
  measurement" was written about the retrieval pipeline's own nondeterminism, and
  this is a second, independent source of it inside the scorer.
- **Phase 0 must therefore report the score distribution near θ**, not just the
  selected set: how many chunks per search land in the ±0.03 band decides whether
  a threshold is usable at all.
- **Mitigations to weigh in Phase 1** (not decided here): pick θ in a sparse part
  of the distribution; take top-N by score instead of a threshold (rank is far
  more stable than the value); or accept the flicker because the recall floor
  tops the result up anyway. The last one is probably enough, which is an argument
  for keeping the floor rather than deleting it.

Note the usage counts are identical across runs while the probabilities are not,
so this is sampling inside the model, not a differing request.

## Shapes

**Noul** (the selection question) — one per chunk, independent, thresholdable:

```python
Noul(instructions="Chunk c17 answers, or directly supports an answer to, the query.")
# -> {"type": "noul", "noul": 0.84}
```

**Choice** (the extraction question) — 121 options including an explicit
`none_of_these` returned a clean abstain on nonsense candidates:

```
choice="none_of_these"  confidence=0.98  probabilities={none_of_these: 0.99, cand_0: 0.01, ...}
```

The abstain path the addendum asked for works: the model does not have to pick a
real id when none fits, and `confidence` is available to gate on. Every option
gets a probability and they sum to one, so `Choice` cannot be used to select a
SET — that is why selection uses Noul-per-chunk.

## Latency and cost

- 88–309 ms across every shape probed, including the 300-question request. The
  published 70–500 ms holds; a full-pool scoring is ~0.3 s, against the gatherer
  turn's measured p50 of 1,909 ms.
- 29,057 input tokens for a 300-question request ≈ **$0.0012** at $0.042/M.
  Output tokens are returned in `usage` (5,594 on that request) but are not
  billed per the published pricing — treat that as unconfirmed until an invoice
  shows it.

## Client notes

- `pip install typesafe-sdk` (0.7.1). `TypeSafeClient(api_key=...)`; the SDK does
  NOT read `TYPESAFE_API_KEY` itself — pass it explicitly.
- Errors are typed: `TypeSafeBadRequestError`, `TypeSafeAuthenticationError`,
  `TypeSafeAPITimeoutError`, `TypeSafeAPIConnectionError`,
  `TypeSafeAPIResponseValidationError`. `str(exc)` carries the method, URL,
  status, body and a `request_id` — log the class name AND the string; the
  request_id is what a vendor ticket needs.
- `RetryPolicy(max_retries=...)` is built in; set it to 0 when probing a limit,
  or a retried 400 just slows the answer down.

## Entity auto-merge (Choice + `none_of_these`), measured 2026-09-23

`engine/ingest/auto_merge/jev_judge.py` asks ONE Choice per judgment: the
analyzer's ≤10 filtered candidates as `c0..cN`, plus `none_of_these`. The
replay (`scripts/jev_automerge/`) rebuilt 477 managed-plane decisions with the
analyzer's own candidate SQL and asked Jev and the production gpt-oss prompt the
same questions on identical inputs.

| | Jev | gpt-oss |
|---|---|---|
| same real-world entity as gpt-oss (all / everyday traffic) | 94.8% / 98.8% | — |
| agrees with the STORED past gpt-oss decision (same entity) | 89.5% | 84.8% (its own re-run) |
| failed calls | 0 / 954 | 33 / 477 (reasoning ate `max_tokens=512`) |
| latency p50 / p90 | 145 / 197 ms | 383 / 625 ms |
| input tokens p50 / max | 1,773 / 3,154 | — |
| cost per decision | $0.000078 | $0.000703 |

**Precision, not agreement.** Every auto-merge each gate would make was checked
against hard identity evidence (same repo + PR/issue number, shared UUID, shared
email or login, a human-approved merge):

| gate | auto-merges | verified | known false | name-only (needs a human) |
|---|---|---|---|---|
| gpt-oss `high` | 107 | 106 | 0 | 1 |
| Jev Choice p ≥ 0.95 | 84 | 83 | 0 | 1 |
| Jev Choice p ≥ 0.95 + Person shared-identifier guard | 83 | 83 | 0 | 0 |
| Jev pairwise Noul (one per candidate) ≥ 0.95 | 58 | 58 | 0 | 0 |

Pairwise Noul looks safer but found 58 of the 107 verified duplicates against
83 for the Choice, and its score cannot separate the one risky pair (0.80, while
verified pairs go as low as 0.78). So the Choice stays, and the name-only class
is handled by a deterministic gate instead.

**Document nodes are never judged.** A node whose id is one of the tenant's
documents is that document's graph node -- retrieval joins
`documents.doc_id = graph_nodes.canonical_id` -- so the analyzer skips it before
the candidate search, as it does a node whose properties carry a `doc_type` (a
stub some writers create before the document row). It may still be the primary
another node merges into. `merge_cluster` re-checks both under its lock when
the caller sets `refuse_document_aliases`, which auto-merge and both suggestion
approve routes do.
On the managed plane 661 of the 704 auto-merges made before 2026-08 folded a
document's node into its mention (above all a PR's `github:o/r:pr:N` document
into the bare `o/r#N`), and on the replay 68 of Jev's 82 would have too; 875
live documents are detached that way today (TODOS.md).

**The shipped gate.** A `high` answer (p ≥ 0.95) executes only when
`jev_judge.execution_evidence` finds identity evidence the pair shares: an exact
email or login (a login only within one source system) for people; for other
labels also the same repo + PR/issue number, the same LEAF UUID (the id's last
segment, when it is a UUID: sibling issues share their workspace UUID, and
every upload id carries the tenant's), the same repo-style slug up to case and
`-`/`_` (dots are kept: `v1.1` is not `v11`), or the same repo name with a
compatible owner. Otherwise it is written as a suggestion; an answer from any
model but `AUTO_MERGE_JEV_MODEL` (or naming none) is too. Re-scored on the
frozen T7 set (463 decisions) with the production code: 263 skipped as document
nodes; 14 auto-merges -- 3 verified, 11 people confirmed only by the shared
email/login the gate itself requires -- 0 known false, 0 needing a human;
`score.py` prints `ACCEPTANCE: PASS`. For non-Person pairs "verified" is mostly
the same evidence the gate checks, so it shows agreement, not independent
truth; the day-one review of live merges is the independent check.

**Two properties of the Choice probability to design around:**

- **It splits across copies of the same entity.** When the graph holds the
  same repo three ways (`org/x`, `x`, `wiki:repo:x`), the mass divides between
  them: in 98 of 215 picks another candidate that is the same entity held ≥ 0.05.
  That lowers p, which errs toward a suggestion, never toward a wrong merge.
- **It drifts ±0.02–0.03 between identical calls** (20 repeats on three
  near-threshold pairs: 0.87–0.92, 0.89–0.92, 0.91–0.95). Every upsert re-queues
  its node, so a pair is judged many times, and under that churn a 0.95 cut
  behaves like "averages ~0.92". The fix for that is not re-judging unchanged
  nodes (TODOS.md).

`max_tokens_exceeded` is permanent for its input (same state, same answer), so
`post_choice` raises `JevRequestTooLarge` and does not trip the breaker; string
property values are trimmed at 500 characters and lists at 50 items before that
can happen, keys never (no replayed property held a list, so the list cap
changes no measured request).

**Failure classes.** 400/413/422 are refusals of one request
(`JevRequestRejected`): the node is dropped from the queue as an error. Only a
request-SCHEMA refusal (a list `detail`) counts against `MERGE_BREAKER`, so a
server that changed its schema opens it; other refusals do not, because the
breaker is shared by every tenant. `400 api_usage_error` -- what an unknown or retired model
gets (checked 2026-09-23) -- and every other non-200 -- 401/403 (a revoked
key), 404 (a wrong URL), 408, 429, 5xx -- and any malformed answer count
against the breaker and DEFER the node (5 min doubling to 1 h, parked after
12). An answer naming a
model other than `AUTO_MERGE_JEV_MODEL` may suggest but never auto-merges: the
bands were measured on that model.

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

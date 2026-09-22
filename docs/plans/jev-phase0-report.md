# Phase 0 — the comparison, measured

Run overnight 2026-09-22 on prbe-devbox. Worktree `~/trees/prbe-knowledge/jev-shadow-scoring`.
Raw data in the session scratchpad; code committed on branch `jev-shadow-scoring`.
Plan this answers: `~/plans/jev-typed-decisions-plan.REVIEWED-2026-09-22.md`.

## The answer, in three lines

1. **The gatherer's LLM call is not earning its keep.** Today's results are not
   better than the recall floor alone — 6.0% vs 7.4% useful documents in the
   region where they differ (p=0.06, and the floor is nominally *ahead*), and
   identical on whether the result set answers the query (10.9% vs 10.6%, p=0.92).
   That call costs **$293/month**.
2. **Jev used as a RANKER beats both, decisively.** Top-10 by Jev score: **15.5%**
   useful vs the floor's 7.4% (2.1x, p=6e-18) and **15.8%** of result sets answer
   the query vs 10.6% (p=0.037). Replicated on two disjoint samples and on a
   split-half of the pooled set.
3. **Jev used as a THRESHOLD does almost nothing.** At θ=0.7, 75% of searches have
   zero documents clearing; at θ=0.9, 96% do. The plan's thresholding design is
   wrong and should be dropped — see "What this changes in the plan".

Cost of the winning arm: **$19/month** (measured, 38,128 tokens per search at
$0.042/M) against $293 — and it is ~0.3s where the gatherer turn is p50 1.9s.

## What was compared

3,048 real searches replayed from stored R2 trace blobs (six tenants, 2026-08-23
to 09-22; `new-workspace`, `oneleet-*`, `test-prod`, `testing`, `probe-demo`
excluded, plus 115 zero-recall and 18 empty-pool traces, all counted in the log).
Four arms, each cut to the same 10-document budget:

| arm | what it is |
|---|---|
| **A** | what the search actually delivered (gatherer picks + recall floor) |
| **B** | the recall floor alone, no model — reconstructed by running the real `_backfill_recall_floor` against an empty output |
| **C** | Jev scores ≥ θ, topped up by the floor |
| **D** | top 10 by Jev score, no threshold |

A judge (claude-haiku-4.5) graded the **disagreement region** — documents one arm
delivered and another did not — with two labels: is this document useful, and
does the delivered SET answer the query.

## The judge is not vacuous

Asked the same question about three kinds of document, it separates them 4.6x:

| documents | useful |
|---|---|
| both arms delivered (the agreed core) | 17.5% |
| contested (one arm only) | 6–16% |
| in the pool, no arm delivered (control) | **3.8%** |

Absolute precision is low everywhere, and that is partly the corpus: many queries
are agent-generated lookups (`attempt-0010`, `attempt 6 val_bpb 0.997208`) where
"would a person consider this useful" fits badly. The *relative* comparison is
unaffected — every arm faces the same queries.

**This is the one number a human still has to check.** The 40-query hand-graded
slice was not done, so judge-vs-human agreement is unmeasured and the gate the
plan defined (agreement ≥ 0.8) is **not yet satisfied**. Treat the ranking of the
arms as solid and the absolute percentages as provisional.

## Results (current pipeline era, 2026-09-12 onward, 368 distinct traces judged)

| arm | contested docs | useful | precision | sets answering the query |
|---|---|---|---|---|
| A — today | 2,215 | 133 | 6.0% | 10.9% |
| B — floor only | 2,285 | 169 | 7.4% | 10.6% |
| **D — Jev rank** | 2,285 | **355** | **15.5%** | **15.8%** |

| comparison | precision | p | answerable | p |
|---|---|---|---|---|
| D vs B | 15.5% vs 7.4% | 6e-18 | 15.8% vs 10.6% | 0.037 |
| D vs A | 15.5% vs 6.0% | 9e-25 | 15.8% vs 10.9% | 0.047 |
| A vs B | 6.0% vs 7.4% | 0.062 (ns) | 10.9% vs 10.6% | 0.92 (ns) |

Split-half of the pooled set: D 14.6% / 16.5%. Two independent samples (seeds 7
and 11) gave D 15.8% / 16.4%. This does not rest on one run.

## Five things the replay found that the plan had wrong

**1. `harness_appended` did not exist before 2026-09-12**, and treating its absence
as "the model chose this" produced a clean, false cliff on the day the field
shipped — every older trace read as 100% model-supplied. Caught by its own test;
the harness now reports provenance as *unknown* for those traces. Where it IS
recorded (818 traces): the model supplies **5.9%** of delivered documents and the
floor 94.1%, and in **71.6%** of searches the model picks nothing at all. The
team note's "88% floor / 12% gatherer" (09-11, chunk-level) is now 94/6 at
document level.

**2. Thresholding is the wrong rule for this scorer.** 58% of pool documents score
≤0.1 and only 5.2% score ≥0.8, so a cutoff is empty on most searches. The *order*
is what Jev is good at. This also dissolves the nondeterminism problem from T1
(±0.03 drift run-to-run): rank is far more stable than the value.

**3. Jev is not deterministic.** Three byte-identical requests drifted a mean 0.029
and max 0.060 per probability, with identical token counts. Full contract in
`docs/jev-contract.md`.

**4. The token cap is on the whole request, and questions cost ~35 tokens each** —
at 100 chunks that is 3,500 tokens a chars/4 estimate never sees. 67 of 3,048
traces blew a budget set at 0.75 of the cap, because code and log chunks tokenize
denser than the prose the ratio was measured on. The batcher now halves and
retries on the server's own `max_tokens_exceeded`; after that, **100% pool
coverage on all 3,048 traces, zero errors**. 46% of searches need two requests.

**5. Questions-per-request has no cap at 300.** The documented "255" is
options-per-Choice. A whole pool is one or two requests, not N.

## Cost and latency, measured

| | today | with Jev ranking |
|---|---|---|
| per search | $0.027 (1.8 gatherer turns + extractor) | **$0.0016** |
| per month at ~12k searches | $293 gatherer + $22 extractor | **$19** |
| selection latency | p50 1,909 ms | 88–309 ms measured across every probe shape |

Phase 0 itself spent **$9.65 of Jev** (two full passes over 3,048 traces) and
roughly **$5 of judge** on the team Anthropic key — the in-cluster LiteLLM
gateway needs a port-forward this host will not keep alive, so the judge went
direct and its cost does not appear in the proxy's spend log.

## What this changes in the plan

- **Drop the threshold design (D4, and arm C).** Ship top-N by rank. This also
  removes the θ-tuning step and the ±0.03 instability it inherited.
- **Keep the recall floor** as the top-up under the ranking — it is also the
  fallback if Jev is down, and it costs nothing to keep.
- **Arm B alone is a live option.** It is statistically indistinguishable from
  today and free. If you want the $293 back this week without any Jev dependency,
  deleting the gatherer call is defensible on this data. D is the better end state.
- **Phase 2 trigger, sized:** the best document scores <0.5 in **44%** of searches,
  <0.4 in 28%, <0.3 in 15%. Pick the rung against the answerability label rather
  than by feel; at 30% the reformulation call is ≈$9/month.

## Not done

- **The 40-query hand-graded calibration slice** — yours or Mahit's hour. Until it
  exists the gate is unsatisfied and the absolute percentages are provisional.
- **T2, the capture PR**, is written but NOT merged and NOT deployed, per the goal.
- **T7 extraction comparison** — the Jev `Choice` abstain path is verified working
  (121 options, clean `none_of_these` at 0.98 confidence), but the extraction arm
  was not run; selection was the load-bearing question and the budget went there.
- **T8 managed-cluster blobs** — not attempted.
- Pre-09-12 traces (2,230 of them) are in the corpus for the A-vs-B set comparison
  but carry no provenance, and the pipeline changed on 09-12, so the headline
  numbers above are the current era only.

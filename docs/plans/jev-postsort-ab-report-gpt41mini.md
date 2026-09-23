# Post-sort A/B report: research-os transcript demotion vs Jev order

Judge: `gpt-4.1-mini` (effort medium), Phase 0 prompt verbatim, per document, position-blind. Traces judged: 400 (unjudged/incomplete: 0).

Origin: {'live': 115, 'phase0': 285}. Tenants: {'anthrogen': 38, 'bucket-robotics': 21, 'monarcha': 14, 'probe': 203, 'strand-ai': 124}.

## Primary and guardrail endpoints (paired per trace, B − A)

| metric | A (today) | B (proposed) | mean diff | 95% CI | wins/losses/ties | p one-sided B>A | p two-sided |
|---|---|---|---|---|---|---|---|
| NDCG@10 (n=378) | 0.868 | 0.907 | +0.039 | [+0.027, +0.052] | 237/75/66 | 4.96e-21 | 9.92e-21 |
| NDCG@8 (n=378) | 0.797 | 0.860 | +0.062 | [+0.046, +0.079] | 231/72/75 | 6.81e-21 | 1.36e-20 |
| NDCG@5 (n=378) | 0.743 | 0.826 | +0.083 | [+0.062, +0.106] | 194/64/120 | 1.08e-16 | 2.16e-16 |
| NDCG@4 (n=378) | 0.754 | 0.842 | +0.088 | [+0.064, +0.112] | 170/51/157 | 1.89e-16 | 3.78e-16 |
| P@4 (n=400) | 0.647 | 0.728 | +0.081 | [+0.060, +0.101] | 147/39/214 | 3e-16 | 6e-16 |
| P@5 (n=400) | 0.620 | 0.688 | +0.068 | [+0.052, +0.084] | 152/45/203 | 4.48e-15 | 8.96e-15 |
| P@8 (n=400) | 0.593 | 0.622 | +0.029 | [+0.021, +0.037] | 111/33/256 | 2.29e-11 | 4.58e-11 |
| P@10 (n=400) | 0.575 | 0.575 | +0.000 | [+0.000, +0.000] | 0/0/400 | 1 | 1 |
| MRR (n=400) | 0.833 | 0.865 | +0.032 | [+0.011, +0.054] | 41/28/331 | 0.074 | 0.148 |

## Set answerability (judge on the delivered SET, per arm)

| k | A answerable | B answerable | traces (sets differ) | traces (sets same) |
|---|---|---|---|---|
| 5 | 53.8% (215/400) | 59.5% (238/400) | 363 | 37 |
| 8 | 61.2% (245/400) | 62.2% (249/400) | 307 | 93 |

## Q3: are the documents the multiplier promotes better than the ones it demotes?

- Documents the partition moved UP (non-transcripts): 1805, useful 50.0%
- Documents the partition moved DOWN (transcripts): 1246, useful 66.5%
- All transcripts in the engine's top-10: 1521, useful 64.2%; all non-transcripts: 2444, useful 55.9%
- Rank-1 document useful: A 77.5%, B 81.8%

## Grader agreement

- gpt-4.1-mini self-agreement (repeat pass): raw nan, kappa nan, n=0
- gpt-4.1-mini vs claude-opus-5-5: raw 0.765, kappa 0.486, n=3964
- gpt-4.1-mini YES rate over judged documents: 59.1%
- Answered by: gpt-4.1-mini 3965 (100.0%)
- `gpt-4.1-mini` refusals seen (stop_reason=refusal, re-asked on the fallback), by document source: {'claude_code': 15, 'codex': 15, 'custom_ingest': 5}
- Human calibration (40-query slice): NOT DONE unless `calibrate --human` was scored; see below.

## Per-origin and per-tenant NDCG@10 diff (B − A)

| slice | n | A | B | diff | wins/losses/ties |
|---|---|---|---|---|---|
| origin=live | 102 | 0.810 | 0.865 | +0.055 | 55/22/25 |
| origin=phase0 | 276 | 0.889 | 0.922 | +0.033 | 182/53/41 |
| tenant=anthrogen | 37 | 0.901 | 0.947 | +0.046 | 25/9/3 |
| tenant=bucket-robotics | 21 | 0.765 | 0.866 | +0.101 | 13/7/1 |
| tenant=monarcha | 12 | 0.799 | 0.882 | +0.083 | 7/2/3 |
| tenant=probe | 186 | 0.854 | 0.885 | +0.031 | 104/37/45 |
| tenant=strand-ai | 122 | 0.903 | 0.938 | +0.035 | 88/20/14 |

## Decision rule (plan §5), evaluated

- Branch 1 (judge calibration): human slice not scored → the human gate is OPEN; model cross-agreement stands in (see above).
- Primary NDCG@10: diff +0.039, CI [+0.027, +0.052], p=4.96e-21 → significant
- Guardrail P@4: diff +0.081, CI lower bound +0.060 → holds
- Guardrail P@5: diff +0.068, CI lower bound +0.052 → holds
- Guardrail P@8: diff +0.029, CI lower bound +0.021 → holds
- **Branch 2: primary significant, guardrails hold → ship PR 1.**

## Cost

- claude-opus-5-5: $38.30
- claude-opus-5: $1.43
- gpt-4.1-mini: $0.57
- judge calls: 9985

## Method notes

- Arms are pure functions of the engine's delivered 10-document list: A = partition (non-transcripts first) → `_dedupe_by_session` → cut; B = engine order → `_dedupe_by_session` → cut. Both ported verbatim from research-os.
- Not replayed: session self-exclusion (not recorded in a trace) and the workspace/project lenses (membership filters, identical across arms, absent on unscoped searches). They can change which document is cut at k, never the order.
- NDCG@k uses binary gains with the ideal computed over the same 10 documents; traces with zero useful documents are excluded from NDCG (undefined) and kept in P@k/MRR.
- Phase 0 rows use arm D (top-10 by Jev, offline, same scorer) as the engine list; live rows use the delivered order recorded in the trace (`gathered.chunks`), which equals `selection.ranked`.

# Post-sort A/B report: research-os transcript demotion vs Jev order

Judge: `claude-opus-5-5` (effort medium), Phase 0 prompt verbatim, per document, position-blind. Traces judged: 399 (unjudged/incomplete: 1).

Origin: {'live': 114, 'phase0': 285}. Tenants: {'anthrogen': 37, 'bucket-robotics': 21, 'monarcha': 14, 'probe': 203, 'strand-ai': 124}.

## Primary and guardrail endpoints (paired per trace, B − A)

| metric | A (today) | B (proposed) | mean diff | 95% CI | wins/losses/ties | p one-sided B>A | p two-sided |
|---|---|---|---|---|---|---|---|
| NDCG@10 (n=364) | 0.934 | 0.954 | +0.020 | [+0.009, +0.031] | 142/58/164 | 1.29e-09 | 2.58e-09 |
| NDCG@8 (n=364) | 0.896 | 0.926 | +0.030 | [+0.016, +0.045] | 136/60/168 | 2.92e-08 | 5.83e-08 |
| NDCG@5 (n=364) | 0.877 | 0.922 | +0.045 | [+0.027, +0.064] | 104/35/225 | 1.93e-09 | 3.86e-09 |
| NDCG@4 (n=364) | 0.884 | 0.929 | +0.045 | [+0.026, +0.065] | 81/27/256 | 9.54e-08 | 1.91e-07 |
| P@4 (n=399) | 0.776 | 0.815 | +0.039 | [+0.025, +0.055] | 70/19/310 | 2.48e-08 | 4.96e-08 |
| P@5 (n=399) | 0.761 | 0.797 | +0.036 | [+0.023, +0.049] | 79/25/295 | 5.27e-08 | 1.05e-07 |
| P@8 (n=399) | 0.743 | 0.755 | +0.011 | [+0.005, +0.018] | 69/32/298 | 0.000148 | 0.000296 |
| P@10 (n=399) | 0.716 | 0.716 | +0.000 | [+0.000, +0.000] | 0/0/399 | 1 | 1 |
| MRR (n=399) | 0.865 | 0.884 | +0.019 | [+0.001, +0.037] | 20/15/364 | 0.25 | 0.5 |

## Set answerability (judge on the delivered SET, per arm)

| k | A answerable | B answerable | traces (sets differ) | traces (sets same) |
|---|---|---|---|---|
| 5 | 53.9% (215/399) | 59.4% (237/399) | 362 | 37 |
| 8 | 61.2% (244/399) | 62.2% (248/399) | 306 | 93 |

## Q3: are the documents the multiplier promotes better than the ones it demotes?

- Documents the partition moved UP (non-transcripts): 1802, useful 70.2%
- Documents the partition moved DOWN (transcripts): 1239, useful 76.0%
- All transcripts in the engine's top-10: 1514, useful 74.4%; all non-transcripts: 2441, useful 72.8%
- Rank-1 document useful: A 84.5%, B 86.5%

## Grader agreement

- claude-opus-5-5 self-agreement (repeat pass): raw 0.965, kappa 0.909, n=1327
- claude-opus-5-5 vs gpt-4.1-mini: raw 0.765, kappa 0.486, n=3964
- claude-opus-5-5 YES rate over judged documents: 73.5%
- Answered by: claude-opus-5-5 3873 (97.7%), claude-opus-5 91 (2.3%)
- `claude-opus-5-5` refusals seen (stop_reason=refusal, re-asked on the fallback), by document source: {'claude_code': 15, 'codex': 15, 'custom_ingest': 5}
- Human calibration (40-query slice): NOT DONE unless `calibrate --human` was scored; see below.

## Per-origin and per-tenant NDCG@10 diff (B − A)

| slice | n | A | B | diff | wins/losses/ties |
|---|---|---|---|---|---|
| origin=live | 91 | 0.904 | 0.953 | +0.049 | 35/12/44 |
| origin=phase0 | 273 | 0.944 | 0.954 | +0.010 | 107/46/120 |
| tenant=anthrogen | 37 | 0.935 | 0.947 | +0.012 | 19/8/10 |
| tenant=bucket-robotics | 21 | 0.823 | 0.913 | +0.090 | 13/5/3 |
| tenant=monarcha | 12 | 0.925 | 0.961 | +0.036 | 3/3/6 |
| tenant=probe | 170 | 0.927 | 0.947 | +0.021 | 56/31/83 |
| tenant=strand-ai | 124 | 0.963 | 0.970 | +0.007 | 51/11/62 |

## Decision rule (plan §5), evaluated

- Branch 1 (judge calibration): human slice not scored → the human gate is OPEN; model cross-agreement stands in (see above).
- Primary NDCG@10: diff +0.020, CI [+0.009, +0.031], p=1.29e-09 → significant
- Guardrail P@4: diff +0.039, CI lower bound +0.025 → holds
- Guardrail P@5: diff +0.036, CI lower bound +0.023 → holds
- Guardrail P@8: diff +0.011, CI lower bound +0.005 → holds
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

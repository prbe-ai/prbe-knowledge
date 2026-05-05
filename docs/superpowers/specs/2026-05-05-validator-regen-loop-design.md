# Validator Regen Loop (handoff)

**Status:** handoff / pre-spec. Decisions still open. Not yet a buildable plan.

**Author of handoff:** generated 2026-05-05 from the Plan 4 V1 PR session
(PR #96, `feat/synth-tenant-seeding-plan4-v1`). Plan 4 V1 had to ship with
a templated-only canonical because plot archetypes drop on validator
strictness in real-LLM mode. This handoff documents what's needed to fix
that and unblock the larger synthetic-data-generation roadmap.

**Reads:** Plan 3 spec
(`docs/superpowers/specs/2026-05-02-synthetic-narrative-layer-design.md`),
Plan 4 V1 spec
(`docs/superpowers/specs/2026-05-04-synth-plan-4-tenant-seeding-v1-design.md`),
`scripts/synth/validator.py`, `scripts/synth/llm/validator_pass2.py`,
`scripts/synth/scenarios.py:202-211` (the deferred TODO this work targets).

---

## The problem

Plot archetypes (`incident`, `launch`, `big_refactor`) get silently dropped
during real-LLM scenario generation when their LLM-written docs fail the
strict validator. The drop happens in
`scripts/synth/scenarios.py:202-211`:

```python
if result.should_drop:
    # TODO(plan3-cleanup): implement validator regen loop (max 2 rounds,
    # surgical doc-level replacement preserving thread_parent_id wiring)
    log.warning(
        "plot_scenario_dropped",
        scenario_id=spec.id,
        archetype=name,
        failing=result.failing_doc_ids,
    )
    continue
```

This is the only gate between "synth produces rich plot content"
(incidents → postmortems → fix PRs threaded across Slack/Linear/GitHub)
and "synth produces only daily standup + on-call handoff noise." Plan 4
V1 ships with templated-only canonical for exactly this reason — the
playground a customer sees is shallow because plot archetypes never made
it into the recording.

Empirical drop rate: Pass 1 (NAME_ONLY) catches LLM-invented names not in
the WorldModel. Pass 2 (LLM consistency) catches cross-doc contradictions
and uses a 30% violation threshold over the scenario's docs. Real-LLM mode
loses most plot scenarios.

---

## What's already in place from Plan 3 (don't redo this work)

The validator infrastructure is mature; everything below already exists
and works:

- **`scripts/synth/validator.py::validate`** orchestrates Pass 1 + Pass 2.
  Returns `ValidationResult(should_drop: bool, failing_doc_ids: tuple[str, ...])`
  per scenario. `failing_doc_ids` is the union of Pass 1 and Pass 2 violators.

- **`scripts/synth/validator.py::Violation`** carries Pass 1 violations:
  `doc_id` + `out_of_world` (the offending tokens).

- **`scripts/synth/llm/validator_pass2.py::validate_pass2`** runs ONE
  `generate_structured` call per scenario asking the LLM "do these docs
  tell a consistent story?" Returns `Pass2Result(passed, violations)`
  where each `Pass2Violation` has `doc_id` + `issue` (free-text reason).
  Threshold: violations > 30% of docs forces `passed=False`.

- **`scripts/synth/scenarios.py:202-211`** — the drop site. Already has
  `result.failing_doc_ids` and (for Pass 2) the violation reasons in
  scope. The caller just doesn't use them.

- **Plot writer**: `scripts/synth/llm/writer.py` is the LLM call that
  generates plot doc bodies given a scenario spec + cast + services +
  channels as constraint. Operates over the whole scenario today.

- **Plot builder loop**: `scripts/synth/scenarios.py` async-iterates plot
  builders that yield `(spec, docs_list)` tuples. Each spec is fully
  constrained before the writer runs.

What's NOT in place is the regen loop itself: the ability to take a list
of `failing_doc_ids` + their failure reasons and ask the writer to
regenerate just those docs while preserving the rest.

---

## What's NOT in place that the regen loop needs

### 1. Surgical doc-level replacement

When validation flags doc 3 of a 5-doc scenario, regen needs to:
- Keep docs 1, 2, 4, 5 as-is.
- Re-ask the writer for a NEW doc 3 with the same `source_event_id`,
  `thread_parent_id`, `occurred_at`, etc. — anything other docs reference.
- Splice the new doc 3 back into the scenario without breaking threading
  (parent IDs, source_event_id references, timestamp ordering).

The writer's current API generates a whole scenario at once. Regen needs
either a new writer entry point (`regenerate_docs(spec, prior_docs,
failing_doc_ids, violation_reasons)`) or an option on the existing entry
point to "regenerate docs N, M with these constraints, holding the rest
fixed."

### 2. Failure context as input

The writer needs to know WHY doc 3 failed, otherwise it's just rolling
dice again:
- Pass 1: `"doc 3 referenced 'Sarah' (out_of_world); cast is {Alice, Bob, Charlie}"`
- Pass 2: `"doc 3's stated root_cause contradicts doc 1's debugging conclusion"`

The failure messages need to be threaded into the regen prompt as
explicit "fix these violations" instructions, not just "regenerate doc 3."

### 3. Round budget + termination

The TODO suggests "max 2 rounds." Need to define:
- What counts as a round (one writer call → one validation pass → either
  pass or repeat).
- What happens when rounds exhaust (drop scenario? Templated fallback?
  Surface for operator?).
- Whether the budget is per-scenario or per-run.

### 4. Pass 2 retry (orthogonal axis)

Pass 2 is itself an LLM call and can be wrong — false positives or
nondeterministic verdicts. Should regen first re-run Pass 2 on the
original docs before regenerating the writer's output? This is a
SEPARATE axis from "regen the writer." Bundling them obscures whether
the LLM writer or the LLM validator was the source of failure.

### 5. Cost observability

Each regen round adds: writer call + Pass 2 call. For a 5-doc scenario
with 2 regen rounds worst case: 5 + 2 + 2 = 9 LLM calls (writer) vs.
5 + 1 = 6 baseline. Plus Pass 2 is per-scenario so +1 baseline, +2 with
2 regen rounds. Net: roughly 50–60% cost increase per scenario in worst
case. Need observability (structured logs / metric counters) to know
what regen actually costs vs. saves.

---

## Open product/design decisions

Listed in rough dependency order — answers up the chain unblock decisions
below them.

### Q1. Per-doc regen vs per-scenario regen

Two architectures:

- **Per-doc (TODO suggestion):** regenerate only the failing docs, splice
  them back into the existing scenario. Preserves scenario shape across
  rounds. Surgical splicing is meaningful work — must hold cross-doc
  references stable.
- **Per-scenario:** regenerate the entire scenario with a different RNG
  seed, validate again. Simpler to implement. Loses determinism — each
  scenario has a specific flavor (cast, services, root_cause) that we
  lose by rerolling the whole thing.

**Recommendation: per-doc.** Matches the TODO. Preserves canonical
"shape" so re-recording is reproducible. Surgical splicing is the price
of admission.

### Q2. Failure context format

What does the regen prompt include?

- **Just violation messages** — minimal. May not give the writer enough
  context to fix.
- **Full prior scenario** — every doc as context. The writer can see
  what it produced and what's surrounding the failing doc.
- **Relevant excerpts only** — the offending lines highlighted, neighbors
  trimmed.

**Recommendation: full prior scenario + an explicit "regenerate doc {id}
to fix these violations: {list}" instruction.** LLMs do better with
context than with constraints alone.

### Q3. Round budget

- 1 round: cheapest. May not converge on hard cases.
- 2 rounds (TODO): probably the sweet spot.
- 3+ rounds: diminishing returns; if 3 rounds don't converge, the writer
  probably can't generate a valid scenario for this prompt at all.

**Recommendation: 2 rounds**, profile-configurable via
`regen.max_rounds`.

### Q4. Termination behavior

When rounds exhaust:

- **Drop scenario** (current behavior, even pre-regen). Log structured
  event with round-by-round failure messages so operators can debug.
- **Templated fallback** — synthesize a templated version of the archetype
  as a placeholder. Keeps scenario count stable but breaks the
  "all-LLM-text" property.
- **Pass with warning** — accept the validator's complaints, log them,
  ship anyway. Defeats the purpose of validation.

**Recommendation: drop scenario** (V1). Add a `--regen-fallback templated`
flag later if drop rate is too high in practice.

### Q5. Cost gate

- **Always on** — every plot scenario can regen up to budget. Cost surge
  on flaky-LLM days.
- **Opt-in via `--regen` flag** — operator decides per-run.
- **Always on with per-run ceiling** — abort regen if cumulative LLM
  spend > threshold.

**Recommendation: always on** for V1. The whole point of regen is to make
plot generation viable; opt-in defeats it. Add a cost ceiling later if
runs blow up.

### Q6. Pass 2 retry

If Pass 2 itself returns different verdicts on the same docs across runs,
should we re-run Pass 2 N times and majority-vote before regenerating?

**Recommendation: accept first Pass 2 verdict** (V1). Validator
nondeterminism is a separate concern. Bundling Pass 2 retry with writer
regen makes failure attribution harder. If Pass 2 flakiness becomes
load-bearing, address it as a follow-up.

---

## Concrete tasks the regen plan should land

These follow the recommended path of per-doc regen + 2 rounds + always-on.
Tweak based on Q1–Q6 answers.

1. **Add `regenerate_docs(spec, prior_docs, failing_doc_ids, violations)`
   entry point** in `scripts/synth/llm/writer.py`. Returns a fresh list
   of docs (only the regenerated ones; caller splices).

2. **Add a regen prompt file** under
   `scripts/synth/llm/prompts/writer_regen.txt` with the failure-context
   format from Q2.

3. **Refactor `scripts/synth/scenarios.py:202-211`** — replace the
   `continue` with a regen loop. Up to `regen.max_rounds` iterations.
   Re-validate after each. On success: yield. On exhaustion: structured
   log + drop.

4. **Splice helper**: a small utility (probably in `scripts/synth/scenarios.py`
   or a new `scripts/synth/regen.py`) that takes original docs +
   regenerated docs and returns a merged tuple, preserving doc order and
   thread wiring. Unit-tested.

5. **Profile field**: `regen.max_rounds: int = 2` in profile YAML +
   `Profile` dataclass.

6. **Tests**:
   - Unit: splice helper preserves doc count and order.
   - Unit: regen loop terminates after budget exhausted, with the right
     log shape.
   - Unit: regen loop early-exits on first success.
   - Mock-LLM: end-to-end with a deliberately-broken scenario; assert
     regen recovers it OR converges-but-fails as expected.
   - Real-LLM (manual / `--record-llm`): run against `tiny_test` profile,
     measure plot-scenario survival rate before vs. after.

7. **Observability**: structured log per regen round
   (`plot_scenario_regen_round`, fields: scenario_id, archetype, round,
   failing_doc_ids, violation_reasons). Aggregate counters for
   regen-attempted / regen-succeeded / regen-exhausted.

8. **Re-record `scripts/synth/canonical/v1/raw/` with plot archetypes
   included** — the payoff. Bumps Plan 4 V1's seeding from 3-doc-mini to
   ~50–100 doc real corpus. Update Plan 4's runbook.

Estimated scope: 200–300 LOC + prompt iteration cycles + ~$5–10 in LLM
spend during development.

---

## Out of scope

- **Pass 2 retry** (validator nondeterminism). Separate concern; Q6.
- **New plot archetypes** (PERF_REGRESSION, DEPENDENCY_BUMP,
  CUSTOMER_ESCALATION). Land regen first; archetypes second.
- **Granola / meeting wrapper** (Plan 3 carry-over).
- **Cost-ceiling enforcement** (Q5 deferred).
- **Per-tenant WorldModel parameterization** (downstream roadmap; not
  blocked by regen but bigger scope).

---

## What this unblocks

Once regen lands, the larger synthetic-data-generation roadmap opens:

1. **Plan 4 V1.5: re-record canonical with plot content.** Replace
   `tests/fixtures/canonical-mini/` consumption with the real
   `scripts/synth/canonical/v1/raw/`. Customer playgrounds become
   demonstrably richer (incidents, launches, refactor narratives).
2. **New plot archetypes** (PERF_REGRESSION, DEPENDENCY_BUMP,
   CUSTOMER_ESCALATION) — mentioned as Plan 3 deferred work. Each adds
   a new flavor to the corpus.
3. **Meeting archetypes + Granola wrapper** — another Plan 3 carry-over.
4. **Per-tenant corpus parameterization** — feed the customer's actual
   repo into the WorldModel extractor; their playground references their
   own services / people / channels.
5. **Eval extension** — plot scenarios drive eval question generation;
   richer corpus → richer evals. The whole "synthetic eval" motion gets
   meaningful coverage of incident-response retrieval, not just
   conversational standup retrieval.

---

## Risks and known unknowns

- **Cost regression.** Worst case ~50–60% LLM-spend increase per plot
  scenario. Need to measure on a representative run before declaring
  victory. If cost is unacceptable, fall back to per-scenario regen
  (Q1) or shorter round budget (Q3).
- **Regen prompt convergence.** LLMs may hallucinate NEW violations when
  asked to fix old ones. Could spiral round-over-round. Round budget
  caps the damage; observability surfaces it.
- **Pass 2 nondeterminism.** Already an issue today; regen interacts
  with it (a flaky Pass 2 verdict could trigger unnecessary regens).
  Q6 defers, but a follow-up may need to revisit.
- **Cross-doc reference bugs from surgical splicing.** If we regen doc 3
  with a new `source_event_id`, doc 4's reference to "doc 3's event_id"
  breaks. Mitigation: explicitly hold `source_event_id` and other IDs
  stable across regen rounds — only the `text` body should change.
- **Writer prompt drift.** Adding the regen variant of the writer prompt
  doubles the prompt-engineering surface. Risk: regen prompt drifts in
  quality vs. the main writer prompt. Mitigation: share as much prompt
  template as possible.

---

## Reference: where this work plugs in

- The TODO: `scripts/synth/scenarios.py:202-211`
- Validator orchestrator: `scripts/synth/validator.py::validate`
- Pass 2 runner: `scripts/synth/llm/validator_pass2.py::validate_pass2`
- Writer (need regen entry point): `scripts/synth/llm/writer.py`
- Plot archetypes: `scripts/synth/archetypes/{incident,launch,big_refactor}.py`
- Profile loader (for `regen.max_rounds`): `scripts/synth/profile.py`

Once regen lands, Plan 4 V1's canonical-record workflow at
`scripts/synth/README.md` → "Seeding a real-shape tenant" → "One-time
setup: record the canonical corpus" produces a meaningfully richer
corpus. Plan 4 V1.5 then re-points `seed_tenant`'s default
`--canonical-dir` from `tests/fixtures/canonical-mini/` (which is meant
for tests) to `scripts/synth/canonical/v1/` (the real recorded corpus).

---

## Status check before starting

Before a session opens this doc and starts writing the implementation
plan, confirm:

- [ ] Plan 4 V1 (PR #96) has merged to main.
- [ ] Q1–Q6 above have been answered (in a 30-min product conversation
  or in writing).
- [ ] Local docker-compose stack still works for `synth init / run /
  clean / seed` against an eval-prefix tenant.
- [ ] Anthropic + OpenAI keys are available locally for real-LLM testing.
- [ ] At least $20 of LLM budget is allocated for development iteration.

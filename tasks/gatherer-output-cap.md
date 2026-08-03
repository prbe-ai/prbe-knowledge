# Gatherer output cap — close the input+reservation overflow

## Root cause

`_enforce_context_budget` bounds only the INPUT messages to a hardcoded
115,000 tokens. The provider's ceiling counts `input + max_completion_tokens`,
and the gatherer sends **no `max_tokens`** — so the provider books the model
maximum (~40k) on top. 115,000 + reservation overshoots the 131,072 window.

Observed live 2026-08-03 on engine `sha-8a66ed4b29bc` (which already carried
the 18k prefanout cap from #443 and the 115k backstop):

    Fireworks_aiException - prompt is too long: 143250 tokens
    exceeds maximum context length of 131071

Three uncounted contributors to the input side:
1. the completion reservation (largest — an absent cap books the model max)
2. `tool_definitions()` — sent every turn, never counted by `_message_tokens`
3. cl100k vs gpt-oss tokenizer drift

Only 131,072 − 115,000 = 16,072 of headroom covered all three.

## Why this is the complete fix, not a workaround

One change attacks both live failures:
- **400 context overflow** — input + reservation now provably fits the window
- **429 TPM quota** — Cerebras reserves `input + max_completion_tokens`
  against a 250,000 org-wide TPM ceiling, so an uncapped ~40k booking per
  request is pure quota burn. Capping it buys concurrent-search headroom
  without raising quota (which we cannot do).

## Sizing (measured, not guessed)

Live `completion_tokens` from both retrieval pods, 3h window (n=10, light
traffic): min 145, p50 1,635, p90/max 4,698. One `finish_reason: "length"`
already present with NO cap set — a runaway output hitting the model ceiling.

`SEARCH_AGENT_MAX_OUTPUT_TOKENS = 16_000` ≈ 3.4x observed max, matching the
headroom precedent of `SEARCH_AGENT_EXTRACTOR_MAX_TOKENS`. Small sample, so
env-overridable.

## Tasks

- [x] Worktree off origin/main (never main)
- [ ] constants.py: derive MAX_CONTEXT_TOKENS from window − output − margin
- [ ] loop.py: send `max_tokens` on every gatherer turn
- [ ] loop.py: count `tool_definitions()` in the context budget
- [ ] loop.py: surface `finish_reason == "length"` as a degraded status
- [x] Tests (7 new; the tool-schema one proven to fail pre-fix, pass post-fix)
- [x] `tests/retrieval/agent/` — 206 passed, covers every changed file
- [x] ruff clean; invariant asserted numerically (105_072 + 16_000 + 10_000 = 131_072)
- [x] main-vs-branch comparison, same selection, back to back on a quiet box:

      origin/main   10 failed, 342 passed
      this branch   10 failed, 345 passed

      IDENTICAL failure set — the same four files on both sides
      (`test_cli_plan3_flags`, `test_directed_phrases`,
      `test_graph_retriever_confidence`, `test_mark_failed_token_flip`), all
      asyncpg/infra-bound and unrelated to this change. The +3 is exactly the
      new tests that match this `-k` selection.

      Worth recording: an earlier run of the SAME branch code read 32 failed /
      322 passed, purely because three other sessions were hammering
      `tests/integration` against a down Postgres. Never compare failure sets
      across differently-loaded runs.

- [x] FULL non-integration suite, both sides, quiet machine:

      origin/main   55 failed, 2617 passed, 22 skipped  (8:53)
      this branch   55 failed, 2621 passed, 22 skipped  (10:04)

      Failure sets diffed by test id: ZERO only-on-branch, ZERO only-on-
      baseline. Identical 55. The +4 passed are exactly the four tests added
      here. No failure touches loop.py / models.py / constants.py; the 55 are
      Postgres/MinIO-bound (test_related_entities 11, test_backfill 4,
      test_seed_db 4, slack/storage/graph-writer handlers).

# Router quality eval — manual run

    uv run pytest tests/evals/test_router_quality.py -v -s -m eval

Not run in CI. Costs Anthropic tokens. Re-run on every change to
`services/retrieval/router.py`, `services/retrieval/grounding.py`,
or `services/retrieval/pipeline.py`. Snapshot the aggregate score in
the PR description and update §Baseline in the design spec.

Requires a live Postgres DB (the `live_db` fixture initializes/truncates)
and `ANTHROPIC_API_KEY` set in the environment (Haiku is called for
real).

When extending fixtures, cover all four query classes:

| Class | Example |
|---|---|
| simple | "show me PR #49 in prbe-backend" |
| vague | "what's up with the auth thing" |
| compound | "PRs that closed ABC-123 and shipped to prod" |
| mixed | "PRs about auth and how many shipped this month" |

Each case carries `expected_intents_count`, `expected_modes`,
`expected_canonical_ids` (per intent), and an optional `notes` field.
Add a case for every distinct router behavior you want pinned —
ambiguous queries that exposed bugs, injection attempts, etc.

See spec `docs/superpowers/specs/2026-05-14-router-intelligence-design.md`
§Eval set for the full schema and §Baseline for the target metrics.

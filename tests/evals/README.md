# LLM eval harness

Phase 0 stubs. Wired into CI via `.github/workflows/evals.yml`, runs weekly
against staging.

## Evals planned

- `entity_extractor.py` — 20 hand-labeled docs. Router Haiku extracts entities;
  score = match ratio on `(entity_type, canonical_id)`. Alert if ≥ 80% regression.
- `query_expansion.py` — 10 hand-labeled queries. Cosine drift vs. baseline
  expansion embeddings — alert on > 0.15 average drift.

## Dataset shape

Each eval lives in `tests/evals/datasets/<name>.jsonl` (gitignored for size;
store in R2). Entry:

    {"input": "...", "expected": {...}, "tags": ["slack", "production"]}

## Running locally

    .venv/bin/python -m tests.evals.run --eval entity_extractor --limit 5

Writes a report to `eval-reports/<ts>-<eval>.json`.

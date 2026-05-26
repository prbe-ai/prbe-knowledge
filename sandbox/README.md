# `sandbox/` — product-sandbox assets for the retrieval-quality eval harness

These files let the auto-research optimization platform run the prbe-knowledge
**retrieval serving slice** (`POST /retrieve`) inside an isolated product sandbox so a
coding agent can optimize retrieval quality against a held-out KPI it never sees.

They are agent-visible build/run assets only. They are **not** part of any production
deploy and **not** the per-role data-plane image. Ingestion / worker / cron / wiki /
MCP never run here.

| file | role |
|---|---|
| `Dockerfile.product` | one container: pgvector PG16 + Python 3.12 + the uvicorn `/retrieve` service. `build.dockerfile_path` in the ProductRuntimeSpec. |
| `entrypoint.sh` | start PG → apply `db/schema.sql` → seed corpus → `uvicorn`. Restores the held-out `/grade/corpus.sql.gz` when present (grade), else the dev corpus (implement), else empty. |
| `smoke.sh` | black-box working check: `/health` 200 + `/retrieve` shape-valid. Leaks no quality signal. |
| `dev_corpus.sql` | tiny BM25-able dev corpus (tenant `eval-tenant`) so the agent's `/retrieve` returns rows while iterating. Not the eval corpus. |

The held-out grader, label set, and the real-embedding corpus dump live in the
auto-research platform (the `EvaluatorBlueprint`), never in this repo. Full design:
`auto-research/tasks/prbe-knowledge-sandbox-design.md`.

**Do not add this dir to the agent's `target_globs`** — the agent optimizes
retrieval/ranking code (`services/retrieval/**`, `shared/constants.py`), not its own
sandbox harness.

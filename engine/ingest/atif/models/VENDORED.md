# Vendored: Harbor's ATIF models

Source: https://github.com/laude-institute/harbor `src/harbor/models/trajectories/`
Commit: f9f974aee0d3e52670427bfd298caecb64fdde3a (2026-10-06)
Spec: `rfcs/0001-trajectory-format.md` (ATIF v1.8)
License: Apache License 2.0, Copyright the Harbor authors (see LICENSE in this folder).

Changes from upstream: imports rewritten from `harbor.models.trajectories.` to
`engine.ingest.atif.models.`. Nothing else. Linting is excluded for this folder
(pyproject `extend-exclude`) so a refresh stays a clean diff of upstream.

Why vendored and not a dependency: the `harbor` package pulls Docker and trial
tooling the engine never runs; these files need only pydantic.

Refresh: `scripts/refresh_atif_models.sh <commit>` then run `tests/test_atif_build.py`.

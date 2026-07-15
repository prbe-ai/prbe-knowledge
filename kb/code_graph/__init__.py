"""Code-graph: symbol-level structural extraction from git repositories.

Submodules:
  - bridge.py     — synthetic-event enqueuer; called from source connectors
                    (handlers/github.py, future handlers/gitlab.py)
  - clone.py      — initial-backfill shallow clone + worktree walk
  - fetch.py      — Contents API fetch for incremental push events
  - qualifier.py  — LOW-ambition name resolution (imports + locals)
  - secrets.py    — pre-parse skip-list for files that look like secret dumps
  - extractors/   — per-language tree-sitter symbol/edge emitters
"""

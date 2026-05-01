"""Traversal at incident time (spec §6).

Two pieces in Phase 1:

- ``edge_walk.walk_priority_edges`` — priority-ordered context-source
  walk (step 4).
- ``expand.expand_one_hop`` — 1-hop expansion ranked by cosine
  similarity over the same pgvector embeddings the classifier uses
  (step 5).

A token-counting trimmer (``budget_trim``) is deferred — the priority
walk's budget gate handles the budget concern at the source-selection
level for now.
"""

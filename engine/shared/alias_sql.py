"""Alias resolution SQL shared by the write path and the read path.

A routing row can point at a node that is itself an alias: merging `b` into
`c` after `a` was merged into `b` leaves `a -> b -> c`. One hop sent `a`'s
next upsert to `b`, whose node the second merge deleted, so the upsert
resurrected `b` as a stray copy. Resolution therefore follows the chain to
its end. `entity_aliases` is keyed on (customer_id, label, alias_canonical_id),
so each alias has one primary and the walk is a single path; it is still
depth-capped and cycle-safe, because rows can be written by hand.
"""

from __future__ import annotations

#: Longest alias chain followed. Deeper chains resolve to the node this many
#: hops up -- still a live cluster member, never a deleted one past a cycle.
ALIAS_CHAIN_MAX_DEPTH = 8

#: ($1 customer_id, $2 labels text[], $3 canonical_ids text[]) ->
#: (label, alias_canonical_id, primary_canonical_id): one row per input that
#: IS an alias, carrying the END of its chain. Non-aliases are absent.
RESOLVE_ALIASES_SQL = f"""
WITH RECURSIVE inputs AS (
    SELECT DISTINCT label, canonical_id
    FROM UNNEST($2::text[], $3::text[]) AS t(label, canonical_id)
),
walk AS (
    SELECT i.label, i.canonical_id AS start_id, ea.primary_canonical_id AS cur,
           1 AS depth, ARRAY[i.canonical_id, ea.primary_canonical_id] AS path
    FROM inputs i
    JOIN entity_aliases ea
      ON ea.customer_id = $1
     AND ea.label = i.label
     AND ea.alias_canonical_id = i.canonical_id
    UNION ALL
    SELECT w.label, w.start_id, ea.primary_canonical_id,
           w.depth + 1, w.path || ea.primary_canonical_id
    FROM walk w
    JOIN entity_aliases ea
      ON ea.customer_id = $1
     AND ea.label = w.label
     AND ea.alias_canonical_id = w.cur
    WHERE w.depth < {ALIAS_CHAIN_MAX_DEPTH}
      AND NOT ea.primary_canonical_id = ANY(w.path)
)
SELECT DISTINCT ON (label, start_id)
       label, start_id AS alias_canonical_id, cur AS primary_canonical_id
FROM walk
ORDER BY label, start_id, depth DESC
"""


def cluster_members_cte(*, customer_param: str, label_param: str, roots: str) -> str:
    """A recursive CTE body named `members(root, member)`: every alias that
    routes, through any chain, to one of `roots` (a relation with a `root`
    column). Roots themselves are not included."""
    return f"""
members(root, member, depth, path) AS (
    SELECT r.root, ea.alias_canonical_id, 1, ARRAY[r.root, ea.alias_canonical_id]
    FROM {roots} r
    JOIN entity_aliases ea
      ON ea.customer_id = {customer_param}
     AND ea.label = {label_param}
     AND ea.primary_canonical_id = r.root
    UNION ALL
    SELECT m.root, ea.alias_canonical_id, m.depth + 1, m.path || ea.alias_canonical_id
    FROM members m
    JOIN entity_aliases ea
      ON ea.customer_id = {customer_param}
     AND ea.label = {label_param}
     AND ea.primary_canonical_id = m.member
    WHERE m.depth < {ALIAS_CHAIN_MAX_DEPTH}
      AND NOT ea.alias_canonical_id = ANY(m.path)
)"""


def resolve_to_root_cte(*, customer_param: str, label_param: str, inputs: str) -> str:
    """A recursive CTE body named `up(input_id, cur, depth, path)` walking each
    `inputs.canonical_id` up its alias chain; take the deepest row per input
    (or the input itself when it has none) as its root."""
    return f"""
up(input_id, cur, depth, path) AS (
    SELECT i.canonical_id, ea.primary_canonical_id, 1, ARRAY[i.canonical_id, ea.primary_canonical_id]
    FROM {inputs} i
    JOIN entity_aliases ea
      ON ea.customer_id = {customer_param}
     AND ea.label = {label_param}
     AND ea.alias_canonical_id = i.canonical_id
    UNION ALL
    SELECT u.input_id, ea.primary_canonical_id, u.depth + 1, u.path || ea.primary_canonical_id
    FROM up u
    JOIN entity_aliases ea
      ON ea.customer_id = {customer_param}
     AND ea.label = {label_param}
     AND ea.alias_canonical_id = u.cur
    WHERE u.depth < {ALIAS_CHAIN_MAX_DEPTH}
      AND NOT ea.primary_canonical_id = ANY(u.path)
)"""

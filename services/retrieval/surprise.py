"""Surprise-score for graph retriever edges.

Pure function, no I/O. Called unconditionally from
services/retrieval/retrievers/graph.py for every graph hit. The score is
used in two places: (1) graph.py sorts hits by score before returning so
the highest-surprise edge lands at rank 1, and (2) services/retrieval/
fusion.py multiplies graph hits' RRF by min(score, 2.0) when the request's
discovery flag is true (out of fusion's flat-RRF default).

Score targets range [0.5, 4.0] in normal use; hard cap at 8.0 to prevent
runaway when multiple bonuses stack on a single edge.

Multiplicative model: each component contributes a multiplier; all default
to 1.0 when the relevant input is None or unknown.

Components (in application order):
  1. Confidence weight       AMBIGUOUS=1.5, INFERRED=1.25, EXTRACTED=1.0
  2. Cross-source bonus      anchor_source != target_source -> 1.5
  3. Cross-community bonus   anchor_community != target_community
                             AND both not None -> 1.4
  4. Peripheral-to-hub bonus min(degrees) <= 2 AND max(degrees) >= 5 -> 1.3
  5. Hub-to-hub anti-bonus   min(degrees) >= 3 -> log-decay penalty
                             (1.0 at min=2, 0.5 floor at high min)
  6. Final cap: min(score, 8.0)

Hub-to-hub anti-bonus
---------------------
Mirror of Component 4 on the penalty side. When BOTH endpoints have
degree >= 3, the edge connects two hubs -- a structural connector,
not an informative bridge. Common case: a wiki:index page (degree
100+) connecting to a busy repo (degree 50+); the edge exists because
both ends are popular, not because the link carries signal.

Empirical motivation: with discovery=true on a query like
"prbe-knowledge-mcp lane A scoped JWT", graph hits anchored on the
prbe-knowledge-mcp Repo entity surfaced wiki summary pages above the
canonical lane-A commit. Both wiki page and commit are graph-anchored
to the repo, but the wiki page's edge is hub-to-hub while the
commit's edge is peripheral-to-hub.

Penalty grows logarithmically with min(degrees) so a degree-100 hub
edge is penalised more than a degree-5 mild-hub edge. Floored at 0.5
so even extreme hubs aren't completely flattened.

AMBIGUOUS gating
----------------
Components 2 (cross-source) and 3 (cross-community) are SKIPPED for
AMBIGUOUS edges. Empirical analysis on probe-founders showed that
AMBIGUOUS authorship edges between connectors (e.g. Granola Person ->
Claude Code session) are *structurally guaranteed* to be cross-source
and usually cross-community -- those signals fire automatically rather
than carrying real surprise information. Stacking confidence (1.5) *
cross-source (1.5) * cross-community (1.4) = 3.15 made the same low-
quality edge win top-1 on three unrelated queries in a 30-query
sample. INFERRED edges keep the full multiplier stack -- their
cross-source bridges have explicit LLM justification (`why`) and are
genuinely informative.
"""

from __future__ import annotations

import math

_CONFIDENCE_WEIGHT: dict[str, float] = {
    "AMBIGUOUS": 1.5,
    "INFERRED": 1.25,
    "EXTRACTED": 1.0,
}

_CAP = 8.0

# Hub-to-hub anti-bonus knobs.
# Penalty fires when min(degrees) >= _HUB_PENALTY_MIN_DEG.
# Curve: 1.0 - _HUB_PENALTY_SLOPE * log2(min_deg / 2.0), floored at _HUB_PENALTY_FLOOR.
# Sample values with current settings:
#   min_deg = 3:  ~0.94
#   min_deg = 5:  ~0.80
#   min_deg = 10: ~0.66
#   min_deg = 50: 0.50 (floor)
_HUB_PENALTY_MIN_DEG = 3
_HUB_PENALTY_SLOPE = 0.15
_HUB_PENALTY_FLOOR = 0.5


def surprise_score(
    edge_type: str | None,
    confidence: str | None,
    anchor_label: str | None,
    target_label: str | None,
    anchor_source: str | None,
    target_source: str | None,
    anchor_community: int | None,
    target_community: int | None,
    anchor_degree: int,
    target_degree: int,
) -> float:
    """Score an edge by how surprising it is. Higher = more surprising.

    Score is multiplicative on the graph retriever's base 1.0. Range
    targets [0.5, 4.0] in normal use, capped at 8.0 to prevent runaway.

    All inputs except anchor_degree and target_degree may be None; None
    inputs skip the associated multiplier (default 1.0 contribution).

    Args:
        edge_type: Edge type string (e.g. "REFERENCES"). Not used in scoring
            in v1 but accepted for future extensibility.
        confidence: One of "EXTRACTED", "INFERRED", "AMBIGUOUS" or None.
        anchor_label: Node label for the anchor entity (e.g. "Service").
        target_label: Node label for the target/neighbor entity.
        anchor_source: source_system of the anchor document.
        target_source: source_system of the target document.
        anchor_community: Leiden community_id of the anchor node, or None
            if not yet assigned (tenant < 100 edges).
        target_community: Leiden community_id of the target node, or None.
        anchor_degree: Incident-edge count of the anchor node (>= 0).
        target_degree: Incident-edge count of the target node (>= 0).

    Returns:
        Float in (0, 8.0].
    """
    score = 1.0

    # --- Component 1: confidence weight ---
    # AMBIGUOUS edges are more "surprising" because they are uncertain;
    # EXTRACTED deterministic edges are routine.
    if confidence is not None:
        score *= _CONFIDENCE_WEIGHT.get(confidence, 1.0)

    # AMBIGUOUS edges DO NOT stack with the structural bonuses below.
    # Reason: empirically, the AMBIGUOUS edges in our corpus that fire
    # cross-source AND cross-community are mostly authorship edges
    # between connectors (e.g. Granola Person -> Claude Code session).
    # Those signals are structurally guaranteed by how connectors
    # partition themselves, not real surprise. Without this gate,
    # 5 such edges out of 555 systematically won top-1 across unrelated
    # queries in production sampling.
    is_ambiguous = confidence == "AMBIGUOUS"

    # --- Component 2: cross-source bonus ---
    # An edge crossing source boundaries (Slack -> code, Notion -> ticket)
    # is more unexpected than same-source edges (file -> its module).
    # Skipped for AMBIGUOUS edges (see gate comment above).
    if (
        not is_ambiguous
        and anchor_source is not None
        and target_source is not None
        and anchor_source != target_source
    ):
        score *= 1.5

    # --- Component 3: cross-community bonus ---
    # An edge bridging two different Leiden communities is a structural
    # bridge -- more surprising than intra-community edges.
    # Skipped for AMBIGUOUS edges (see gate comment above).
    if (
        not is_ambiguous
        and anchor_community is not None
        and target_community is not None
        and anchor_community != target_community
    ):
        score *= 1.4

    # --- Component 4: peripheral-to-hub bonus ---
    # An edge from a low-degree node (peripheral) to a high-degree hub is
    # noteworthy: the peripheral node has few connections, so each one
    # carries more signal. Threshold: one end <= 2 edges, other end >= 5.
    # Applies regardless of confidence -- this signal IS independent of
    # the confidence tier (a low-degree node connecting to a hub is a
    # graph-shape property, not a connector-architecture artifact).
    min_deg = min(anchor_degree, target_degree)
    max_deg = max(anchor_degree, target_degree)
    if min_deg <= 2 and max_deg >= 5:
        score *= 1.3

    # --- Component 5: hub-to-hub anti-bonus ---
    # Mirror of Component 4 on the penalty side. When BOTH endpoints have
    # degree >= 3, the edge is a structural connector between two hubs
    # rather than an informative bridge. Common case: wiki:index pages
    # (degree 100+) connecting to busy repos (degree 50+) -- the edge
    # exists because both ends are popular, not because the link carries
    # signal. Penalty scales with log2(min_deg) so a degree-100 hub edge
    # is demoted more than a degree-5 mild-hub edge.
    if min_deg >= _HUB_PENALTY_MIN_DEG:
        decay = 1.0 - _HUB_PENALTY_SLOPE * math.log2(min_deg / 2.0)
        score *= max(decay, _HUB_PENALTY_FLOOR)

    # --- Cap ---
    return min(score, _CAP)

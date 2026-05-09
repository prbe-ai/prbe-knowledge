"""Table-driven unit tests for services/retrieval/surprise.py.

Tests cover:
- Identity case: all inputs None/neutral -> 1.0
- Each component in isolation
- Component stacking
- Cap at 8.0
- Unknown/unrecognised confidence -> 1.0 fallthrough
- Null anchor_source or target_source -> no cross-source bonus
- Both communities None -> no cross-community bonus
- One community None -> no cross-community bonus
- Same community -> no cross-community bonus
- Low-degree-only (no hub) -> no bonus
- High-degree-only (no peripheral) -> no bonus
- All components stacked -> cap fires
"""

from __future__ import annotations

import pytest

from services.retrieval.surprise import surprise_score


def _score(
    *,
    edge_type: str | None = None,
    confidence: str | None = None,
    anchor_label: str | None = None,
    target_label: str | None = None,
    anchor_source: str | None = None,
    target_source: str | None = None,
    anchor_community: int | None = None,
    target_community: int | None = None,
    anchor_degree: int = 2,
    target_degree: int = 2,
) -> float:
    return surprise_score(
        edge_type=edge_type,
        confidence=confidence,
        anchor_label=anchor_label,
        target_label=target_label,
        anchor_source=anchor_source,
        target_source=target_source,
        anchor_community=anchor_community,
        target_community=target_community,
        anchor_degree=anchor_degree,
        target_degree=target_degree,
    )


# ---------------------------------------------------------------------------
# Case 1: identity -- all neutral/None inputs -> score == 1.0
# ---------------------------------------------------------------------------


def test_identity_all_none() -> None:
    """All None / neutral inputs: score is exactly 1.0."""
    s = _score()
    assert s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 2: confidence weight -- each tier in isolation
# ---------------------------------------------------------------------------


def test_confidence_extracted_no_bonus() -> None:
    """EXTRACTED confidence contributes weight 1.0 -- no change."""
    s = _score(confidence="EXTRACTED")
    assert s == pytest.approx(1.0)


def test_confidence_inferred_boost() -> None:
    """INFERRED confidence contributes 1.25x."""
    s = _score(confidence="INFERRED")
    assert s == pytest.approx(1.25)


def test_confidence_ambiguous_boost() -> None:
    """AMBIGUOUS confidence contributes 1.5x."""
    s = _score(confidence="AMBIGUOUS")
    assert s == pytest.approx(1.5)


def test_confidence_unknown_string_falls_through_to_1() -> None:
    """Unrecognised confidence string -> weight 1.0 (no crash, no change)."""
    s = _score(confidence="SUPER_CONFIDENT_LOL")
    assert s == pytest.approx(1.0)


def test_confidence_none_falls_through_to_1() -> None:
    """None confidence -> weight 1.0."""
    s = _score(confidence=None)
    assert s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 3: cross-source bonus in isolation
# ---------------------------------------------------------------------------


def test_cross_source_bonus_applies() -> None:
    """Different source systems -> 1.5x bonus."""
    s = _score(anchor_source="slack", target_source="github")
    assert s == pytest.approx(1.5)


def test_same_source_no_bonus() -> None:
    """Same source system -> no bonus."""
    s = _score(anchor_source="github", target_source="github")
    assert s == pytest.approx(1.0)


def test_cross_source_anchor_none_no_bonus() -> None:
    """anchor_source=None -> cannot compare, no bonus."""
    s = _score(anchor_source=None, target_source="github")
    assert s == pytest.approx(1.0)


def test_cross_source_target_none_no_bonus() -> None:
    """target_source=None -> cannot compare, no bonus."""
    s = _score(anchor_source="slack", target_source=None)
    assert s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 4: cross-community bonus in isolation
# ---------------------------------------------------------------------------


def test_cross_community_bonus_applies() -> None:
    """Different non-None communities -> 1.4x bonus."""
    s = _score(anchor_community=1, target_community=2)
    assert s == pytest.approx(1.4)


def test_same_community_no_bonus() -> None:
    """Same community -> no bonus."""
    s = _score(anchor_community=5, target_community=5)
    assert s == pytest.approx(1.0)


def test_cross_community_anchor_none_no_bonus() -> None:
    """anchor_community=None -> cannot compare, no bonus."""
    s = _score(anchor_community=None, target_community=3)
    assert s == pytest.approx(1.0)


def test_cross_community_target_none_no_bonus() -> None:
    """target_community=None -> cannot compare, no bonus."""
    s = _score(anchor_community=2, target_community=None)
    assert s == pytest.approx(1.0)


def test_cross_community_both_none_no_bonus() -> None:
    """Both communities None -> no bonus."""
    s = _score(anchor_community=None, target_community=None)
    assert s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 5: peripheral-to-hub bonus in isolation
# ---------------------------------------------------------------------------


def test_peripheral_to_hub_bonus_applies() -> None:
    """One endpoint has <= 2 edges, other >= 5 -> 1.3x bonus."""
    s = _score(anchor_degree=1, target_degree=10)
    assert s == pytest.approx(1.3)


def test_hub_to_peripheral_bonus_applies() -> None:
    """Direction is symmetric: hub first, peripheral second also triggers."""
    s = _score(anchor_degree=10, target_degree=2)
    assert s == pytest.approx(1.3)


def test_both_peripheral_no_bonus() -> None:
    """Both endpoints have <= 2 edges but neither is a hub -> no bonus."""
    s = _score(anchor_degree=1, target_degree=1)
    assert s == pytest.approx(1.0)


def test_both_hub_anti_bonus_applies() -> None:
    """Both endpoints are hubs (>= 3) -> hub-to-hub anti-bonus penalty.

    Replaces the previous "both hub -> no bonus = 1.0" test. The hub-to-
    hub anti-bonus (Component 5) now actively demotes such edges as
    structural-connector noise, not informative bridges.
    """
    import math

    # min_deg = 10. Expected penalty: 1.0 - 0.15 * log2(10/2) = 1.0 - 0.15 * log2(5)
    s = _score(anchor_degree=10, target_degree=20)
    expected = 1.0 - 0.15 * math.log2(5.0)
    assert s == pytest.approx(expected)
    # And it's strictly < 1.0 (a real penalty, not a no-op).
    assert s < 1.0


def test_boundary_peripheral_exactly_2() -> None:
    """Peripheral threshold is inclusive: degree == 2 qualifies."""
    s = _score(anchor_degree=2, target_degree=5)
    assert s == pytest.approx(1.3)


def test_boundary_hub_exactly_5() -> None:
    """Hub threshold is inclusive: degree == 5 qualifies."""
    s = _score(anchor_degree=0, target_degree=5)
    assert s == pytest.approx(1.3)


def test_boundary_hub_below_threshold() -> None:
    """max_deg == 4 does not trigger hub bonus."""
    s = _score(anchor_degree=1, target_degree=4)
    assert s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 5b: hub-to-hub anti-bonus (Component 5)
# ---------------------------------------------------------------------------
# Mirror of Component 4 on the penalty side. Demotes edges where BOTH
# endpoints are hubs (degree >= 3) as structural connectors rather than
# informative bridges. Penalty scales with log2(min_deg).


def test_hub_anti_bonus_threshold_min_deg_2_no_penalty() -> None:
    """min_deg == 2 stays just below the penalty threshold."""
    s = _score(anchor_degree=2, target_degree=2)
    assert s == pytest.approx(1.0)


def test_hub_anti_bonus_fires_at_min_deg_3() -> None:
    """min_deg == 3 is the threshold; mild penalty applies."""
    import math

    s = _score(anchor_degree=3, target_degree=3)
    expected = 1.0 - 0.15 * math.log2(3.0 / 2.0)
    assert s == pytest.approx(expected)
    assert s < 1.0  # genuine penalty, not no-op


def test_hub_anti_bonus_floor_at_high_degree() -> None:
    """Penalty bottoms out at the floor (0.5) for very high degrees."""
    s = _score(anchor_degree=200, target_degree=200)
    # At min_deg=200: 1.0 - 0.15 * log2(100) ~= 1.0 - 0.15 * 6.64 = 0.004
    # which is below the 0.5 floor, so the floor pins it.
    assert s == pytest.approx(0.5)


def test_hub_anti_bonus_min_deg_dominates_max_deg() -> None:
    """Penalty depends on min(degrees), not max -- a peripheral-to-hub
    edge (min=1, max=200) gets the bonus and NO penalty."""
    s = _score(anchor_degree=1, target_degree=200)
    # min=1, max=200: bonus fires (1.3x), penalty doesn't (min < 3).
    assert s == pytest.approx(1.3)


def test_hub_anti_bonus_stacks_with_confidence() -> None:
    """The penalty applies on top of confidence weight (multiplicative)."""
    import math

    # INFERRED edge between two degree-10 nodes.
    # confidence=1.25, no source/community bonuses, hub-hub penalty.
    s = _score(
        confidence="INFERRED",
        anchor_degree=10,
        target_degree=10,
    )
    decay = 1.0 - 0.15 * math.log2(10.0 / 2.0)
    expected = 1.25 * decay
    assert s == pytest.approx(expected)


def test_hub_anti_bonus_stacks_with_cross_source() -> None:
    """Cross-source bonus * hub-hub penalty applies to INFERRED edges
    bridging two hub docs across sources -- the wiki-page-vs-canonical-
    commit failure mode this component was added for."""
    import math

    s = _score(
        confidence="INFERRED",
        anchor_source="wiki",
        target_source="github",
        anchor_degree=50,    # wiki-page-shaped hub
        target_degree=20,    # repo-shaped hub
    )
    decay = max(1.0 - 0.15 * math.log2(20.0 / 2.0), 0.5)
    expected = 1.25 * 1.5 * decay
    assert s == pytest.approx(expected)
    # The combined edge still yields a positive multiplier but is heavily
    # demoted vs an equivalent peripheral-to-hub INFERRED edge:
    # 1.25 * 1.5 * 1.3 = 2.4375 (peripheral case)
    # 1.25 * 1.5 * decay (~0.50) = ~0.94 (hub-hub case)
    assert s < 1.25 * 1.5  # less than confidence + cross-source alone


def test_hub_anti_bonus_log_decay_monotonic() -> None:
    """Higher min_deg -> lower score (until floor). Sanity-checks the
    log-decay direction against accidental sign flips."""
    s_3 = _score(anchor_degree=3, target_degree=3)
    s_5 = _score(anchor_degree=5, target_degree=5)
    s_20 = _score(anchor_degree=20, target_degree=20)
    s_100 = _score(anchor_degree=100, target_degree=100)
    assert s_3 > s_5 > s_20 >= s_100  # >= because s_20 may already hit floor


def test_hub_anti_bonus_below_threshold_no_effect() -> None:
    """For min_deg in {0, 1, 2} (peripheral region), penalty is inert."""
    for d in (0, 1, 2):
        # Pair with same degree on the other end so peripheral-to-hub
        # bonus also doesn't fire (its other-end-hub gate needs max>=5).
        s = _score(anchor_degree=d, target_degree=d)
        assert s == pytest.approx(1.0), f"min_deg={d} should be neutral"


# ---------------------------------------------------------------------------
# Case 6: stacking multiple components
# ---------------------------------------------------------------------------


def test_ambiguous_does_not_stack_cross_source() -> None:
    """AMBIGUOUS edges do NOT compound with cross-source bonus.

    Reason (see surprise.py docstring): AMBIGUOUS authorship edges
    between connectors are structurally guaranteed to be cross-source
    (Granola Person -> Claude Code session is always cross-source).
    Without this gate, 5 such edges out of 555 systematically won
    top-1 across unrelated queries in production sampling. With it,
    AMBIGUOUS edges score 1.5 from confidence weight only.
    """
    s = _score(
        confidence="AMBIGUOUS",
        anchor_source="slack",
        target_source="github",  # would normally trigger cross-source 1.5x
    )
    assert s == pytest.approx(1.5)


def test_ambiguous_does_not_stack_cross_community() -> None:
    """Same gate applies to cross-community: AMBIGUOUS edges don't
    compound. Different connectors land in different Leiden communities
    automatically; that signal is structural, not informative.
    """
    s = _score(
        confidence="AMBIGUOUS",
        anchor_community=0,
        target_community=7,  # would normally trigger cross-community 1.4x
    )
    assert s == pytest.approx(1.5)


def test_ambiguous_does_not_stack_either_structural_bonus() -> None:
    """Combined gate: AMBIGUOUS doesn't compound with cross-source AND
    cross-community simultaneously. Pre-fix this was 1.5 * 1.5 * 1.4 =
    3.15 (the value that systematically won top-1 in production)."""
    s = _score(
        confidence="AMBIGUOUS",
        anchor_source="granola",
        target_source="claude_code",
        anchor_community=7,
        target_community=0,
    )
    # Both structural bonuses gated -> only confidence weight.
    assert s == pytest.approx(1.5)


def test_ambiguous_still_stacks_peripheral_to_hub() -> None:
    """Peripheral-to-hub IS a graph-shape signal independent of
    connector partitioning -- so it still compounds with AMBIGUOUS.
    A low-degree node connecting to a hub carries information regardless
    of confidence tier.
    """
    s = _score(
        confidence="AMBIGUOUS",
        anchor_degree=2,
        target_degree=8,
    )
    # confidence (1.5) * peripheral-hub (1.3) = 1.95
    assert s == pytest.approx(1.5 * 1.3)


def test_inferred_keeps_full_stacking() -> None:
    """Regression: the gate is AMBIGUOUS-only. INFERRED edges keep their
    full multiplier stack (the formula's main value -- INFERRED edges
    bridging sources/communities are exactly what we want to surface).
    """
    s = _score(
        confidence="INFERRED",
        anchor_source="slack",
        target_source="code_graph",
        anchor_community=0,
        target_community=7,
    )
    # 1.25 * 1.5 * 1.4 = 2.625
    assert s == pytest.approx(1.25 * 1.5 * 1.4)


def test_extracted_keeps_full_stacking() -> None:
    """Regression: EXTRACTED also keeps full stacking (its 1.0 confidence
    just makes the structural multipliers visible). cross-source bridges
    in deterministic edges are still surprising -- e.g. a CALLS edge
    from a Slack-derived doc to a code function."""
    s = _score(
        confidence="EXTRACTED",
        anchor_source="slack",
        target_source="code_graph",
        anchor_community=0,
        target_community=7,
    )
    # 1.0 * 1.5 * 1.4 = 2.1
    assert s == pytest.approx(1.5 * 1.4)


def test_full_stack_without_cap() -> None:
    """INFERRED * cross-source * cross-community * peripheral-hub = 1.25*1.5*1.4*1.3 ~= 3.4125."""
    s = _score(
        confidence="INFERRED",
        anchor_source="slack",
        target_source="code_graph",
        anchor_community=0,
        target_community=7,
        anchor_degree=2,
        target_degree=8,
    )
    expected = 1.25 * 1.5 * 1.4 * 1.3
    assert s == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Case 7: cap at 8.0
# ---------------------------------------------------------------------------


def test_cap_safety_net() -> None:
    """The cap (8.0) is a safety net for any future stacking that exceeds
    the natural maximum. Today the natural max is INFERRED * cross-source
    * cross-community * peripheral-hub = 1.25 * 1.5 * 1.4 * 1.3 = 3.4125
    -- well under the cap. Verifying the cap is in place + score never
    exceeds it on normal inputs.
    """
    # Maximum-stacking case under current rules (INFERRED with all bonuses).
    s = _score(
        confidence="INFERRED",
        anchor_source="slack",
        target_source="code_graph",
        anchor_community=0,
        target_community=7,
        anchor_degree=2,
        target_degree=8,
    )
    assert s <= 8.0

    from services.retrieval.surprise import _CAP

    assert _CAP == 8.0


def test_cap_is_8() -> None:
    """The exported _CAP constant equals 8.0."""
    from services.retrieval.surprise import _CAP

    assert _CAP == 8.0

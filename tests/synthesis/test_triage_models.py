"""Unit tests for triage Pydantic model edge cases.

Pinning the `TriageVerdict.reason` truncate-before-validate behavior.

Production hot bug (acme, 2026-05-08): Haiku returned a
verdict with `reason` ~300 chars while the schema caps it at 240. The
batch-wide `TriageOutput(**payload)` parse raised
`string_too_long`, the provider wrapped it as `TriageParseError`, the
split-retry wrapper's overflow regexes didn't match the new error
shape, and the entire customer drain DLQ'd. The fix in
`services/synthesis/models.py` adds a
`field_validator(mode="before")` that truncates `reason` to the cap
BEFORE the length validator runs — Haiku-overlong reasons get silently
clipped instead of poisoning sibling verdicts.

These tests pin:
  - the truncate-before-validate behavior at the leaf field level,
  - boundary cases (exactly-cap, just-over, None, short),
  - the same behavior when reached via nested `TriageOutput` parse —
    which is the exact path that exploded in production.
"""

from __future__ import annotations

from services.synthesis.models import TriageOutput, TriageVerdict

# ---------------------------------------------------------------------------
# TriageVerdict.reason — direct construction
# ---------------------------------------------------------------------------


def test_verdict_truncates_300_char_reason_to_240() -> None:
    """A 300-char reason gets clipped to exactly 240, no exception.

    This is the production reproduction at the field level. Without the
    `field_validator(mode="before")`, Pydantic would raise
    `string_too_long` here and the batch parse upstream would fail.
    """
    v = TriageVerdict(important=True, score=7.0, reason="x" * 300)
    assert v.reason is not None
    assert len(v.reason) == 240
    assert v.reason == "x" * 240


def test_verdict_reason_at_exact_cap_unchanged() -> None:
    """Boundary: a reason of exactly 240 chars is preserved verbatim —
    truncation only fires when the input is over the cap."""
    s = "y" * 240
    v = TriageVerdict(important=False, score=2.0, reason=s)
    assert v.reason == s
    assert v.reason is not None
    assert len(v.reason) == 240


def test_verdict_reason_one_over_cap_truncated() -> None:
    """Boundary: 241 chars is the smallest input that should trigger
    truncation. After truncation the result is exactly 240."""
    s = "z" * 241
    v = TriageVerdict(important=True, score=5.0, reason=s)
    assert v.reason is not None
    assert len(v.reason) == 240
    assert v.reason == "z" * 240


def test_verdict_reason_none_passes_through() -> None:
    """`reason` is optional. None must not be coerced to a string by
    the `mode="before"` validator."""
    v = TriageVerdict(important=False, score=0.0, reason=None)
    assert v.reason is None


def test_verdict_reason_short_passes_through() -> None:
    """A reason well under the cap is preserved verbatim."""
    v = TriageVerdict(important=True, score=8.0, reason="short")
    assert v.reason == "short"


# ---------------------------------------------------------------------------
# TriageOutput — nested parse path (the production crash site)
# ---------------------------------------------------------------------------


def test_triage_output_parses_with_overlong_nested_reason() -> None:
    """Reproduction of the acme DLQ.

    Before the fix, `TriageOutput(**payload)` raised
    `1 validation error for TriageOutput verdicts.42.reason
    String should have at most 240 characters`. The provider wrapped
    it as `TriageParseError`, the wrapper's `_is_overflow_shaped`
    returned False (no matching regex), the call re-raised, and the
    worker DLQ'd every pending/triaging row in that drain iteration.

    After the fix, the nested `field_validator(mode="before")` clips
    the reason to 240 chars during parse and the rest of the payload
    lands normally.
    """
    payload = {
        "verdicts": {
            "42": {
                "important": True,
                "score": 5.0,
                "reason": "y" * 300,
            }
        }
    }
    out = TriageOutput(**payload)
    assert "42" in out.verdicts
    verdict = out.verdicts["42"]
    assert verdict.important is True
    assert verdict.score == 5.0
    assert verdict.reason is not None
    assert len(verdict.reason) == 240
    assert verdict.reason == "y" * 240


def test_triage_output_mixes_overlong_and_normal_verdicts() -> None:
    """One overlong verdict alongside short ones: the overlong one is
    truncated, the others are untouched. This is the realistic shape —
    Haiku rarely runs over the cap, but when it does only one verdict
    in a batch is affected and the rest must still land cleanly."""
    payload = {
        "verdicts": {
            "1": {"important": True, "score": 7.0, "reason": "ok"},
            "2": {"important": False, "score": 1.0, "reason": "x" * 500},
            "3": {"important": True, "score": 9.0, "reason": None},
        }
    }
    out = TriageOutput(**payload)
    assert out.verdicts["1"].reason == "ok"
    assert out.verdicts["2"].reason is not None
    assert len(out.verdicts["2"].reason) == 240
    assert out.verdicts["3"].reason is None

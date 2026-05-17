"""Tests for the requires_investigation flag on NormalizationResult."""
from shared.models import NormalizationResult


def test_requires_investigation_default_false() -> None:
    assert NormalizationResult().requires_investigation is False


def test_requires_investigation_settable_true() -> None:
    nr = NormalizationResult(requires_investigation=True)
    assert nr.requires_investigation is True


def test_is_empty_independent_of_flag() -> None:
    # The flag is orthogonal to is_empty — flipping it shouldn't make an
    # empty result claim to have content, nor vice versa.
    assert NormalizationResult(requires_investigation=True).is_empty is True

"""Small classifiers for provider error messages."""

from __future__ import annotations

_BUDGET_EXHAUSTION_MARKERS = ("exceededbudget", "budget has been exceeded")


def is_budget_exhausted_error(exc: BaseException | str) -> bool:
    """Return whether an error message identifies a LiteLLM budget refusal."""
    message = str(exc).casefold()
    return any(marker in message for marker in _BUDGET_EXHAUSTION_MARKERS)

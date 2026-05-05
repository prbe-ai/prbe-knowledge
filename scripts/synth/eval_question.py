"""Single source of truth for the EvalQuestion dataclass.

All eval artifact writers, scenario builders, and tests import from here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    """One eval question attached to a scenario.

    Used by eval artifact writers (write_questions_jsonl).  The ``question``
    field is the user-facing prompt text; ``answer_substring`` is a phrase
    that must appear in a correct model answer; ``tags`` are category labels
    (e.g. "INCIDENT", "cross-source"); ``difficulty`` is one of "easy",
    "medium", "hard-temporal"; ``question_index`` is the 0-based position
    within the scenario's question list (used for deterministic sort).
    """

    question: str           # the eval question text
    answer_substring: str   # expected substring in the answer
    tags: tuple[str, ...]   # e.g. ("INCIDENT", "cross-source")
    difficulty: str         # "easy" | "medium-cross-source" | "hard-temporal"
    question_index: int = 0 # ordering within scenario

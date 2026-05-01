"""ScenarioRunner stub — TimeWindow defined here for Task 7+8 archetype builders.

Full implementation in Task 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TimeWindow:
    end: datetime
    days: int

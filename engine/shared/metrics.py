"""Metrics emission -- structured counters + histograms.

Phase 0 emits via structlog; a later Tier 8 wiring plugs OTel here without
touching call sites. Keeping a single facade means we change backends once.

Lane B gauge:
  inferred_edges_llm_cost_per_customer_per_day
    -- emitted by the inferred-edges worker after each extraction call.
    -- tags: customer_id, extractor_id
    -- unit: USD (float)
    -- no alert/cap (D4 decision): measure first, enforce later.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from engine.shared.logging import get_logger

log = get_logger("metrics")


def counter(name: str, value: int = 1, **tags: Any) -> None:
    log.info("metric.counter", metric=name, value=value, **tags)


def gauge(name: str, value: float, **tags: Any) -> None:
    log.info("metric.gauge", metric=name, value=value, **tags)


def histogram(name: str, value: float, **tags: Any) -> None:
    log.info("metric.histogram", metric=name, value=value, **tags)


@contextmanager
def timer(name: str, **tags: Any) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram(name, (time.perf_counter() - start) * 1000, **tags)

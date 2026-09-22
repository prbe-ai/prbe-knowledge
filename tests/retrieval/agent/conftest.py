"""Pin the result selector for this directory's tests.

The shipped default moved from `gatherer` to `floor` (rollout step 1) and
`probe` runs `jev`. Most tests here exercise the gatherer's LLM loop through a
request that names no selector, so they pin the old default explicitly rather
than silently testing the floor instead. Selector tests override these per test.
"""

import pytest


@pytest.fixture(autouse=True)
def _pin_gatherer_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.retrieval.agent import loop

    monkeypatch.setattr(loop, "SEARCH_SELECTOR_DEFAULT", "gatherer")
    monkeypatch.setattr(loop, "SEARCH_SELECTOR_JEV_CUSTOMERS", frozenset())

"""Every provider call carries a wall-clock ceiling unless the caller sets one.

An outbound call with no timeout is a queue-wide hazard, not a slow request.
The ingestion drain runs `worker_max_concurrent` claim loops (2 in production)
and each calls the inferred-edges extractor inline; two calls that never return
occupy every claim loop and the queue stops for all tenants. Worse, the row
stays `processing` with its heartbeat advancing, so the reclaim loop — which
only looks for a STALE heartbeat — never rescues it.

Neither ingest-path call site passed a `timeout` and the wrapper had no default,
so the ceiling depended entirely on whatever LiteLLM happened to default to.
"""

from __future__ import annotations

import pytest

import engine.shared.llm as llm
from engine.shared.constants import LLM_REQUEST_TIMEOUT_SECONDS


@pytest.fixture
def seen(monkeypatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        captured.update(kwargs)
        return "resp"

    async def fake_aembedding(*, model, input, **kwargs):
        captured.update(kwargs)
        return "resp"

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)
    return captured


async def test_acompletion_applies_the_default_timeout(seen) -> None:
    await llm.acompletion(
        model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )

    assert seen["timeout"] == LLM_REQUEST_TIMEOUT_SECONDS, (
        "a call with no timeout must not be able to hang a claim loop forever"
    )


async def test_aembedding_applies_the_default_timeout(seen) -> None:
    await llm.aembedding(model="openai/text-embedding-3-large", input=["hi"])

    assert seen["timeout"] == LLM_REQUEST_TIMEOUT_SECONDS


async def test_an_explicit_caller_timeout_wins(seen) -> None:
    """The default is a backstop, never an override.

    Interactive callers have far tighter deadlines of their own (research-os
    abandons /v1/search at 30s), and they must keep them.
    """
    await llm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        timeout=7.5,
    )

    assert seen["timeout"] == 7.5

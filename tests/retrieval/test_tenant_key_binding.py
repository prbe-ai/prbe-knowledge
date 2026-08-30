"""The retrieval pipeline must bind the tenant's LiteLLM virtual key.

Without this the whole per-tenant billing chain is dead: every call reaches
the proxy on the shared `managed-shared-data-plane` key, so every
`customer-*` key reads $0.0000, `llm_spend_sync` writes $0 snapshots, and
budget enforcement can never fire (measured 2026-08-30).

`tenant_virtual_key_context` existed for exactly this and was never bound at
any entrypoint, which is the bug these tests pin. Asserting at the pipeline
seam rather than the HTTP handler is deliberate: `/query/stream` does its LLM
work inside a StreamingResponse generator that runs AFTER the handler
returns, so a handler-level `async with` would exit before the work happens.
All three endpoints funnel through these functions instead.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from engine.shared import litellm_key as litellm_key_mod
from engine.shared.litellm_key import current_tenant_virtual_key


@pytest.fixture(autouse=True)
def _clear(monkeypatch: pytest.MonkeyPatch):
    litellm_key_mod._KEY_CACHE.clear()
    litellm_key_mod._FAILURE_CACHE.clear()
    monkeypatch.setenv("BACKEND_BASE_URL", "http://prbe-backend.internal:8080")
    monkeypatch.setenv("INTERNAL_BACKEND_API_KEY", "test-internal-key")
    from engine.shared.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run_retrieval", "run_search_phase"])
async def test_pipeline_binds_the_tenant_key_around_the_gatherer(entry: str) -> None:
    """The key must be bound *while* the gatherer runs, not merely fetched."""
    from engine.retrieval import pipeline as pipeline_mod
    from engine.shared.models import QueryRequest

    seen: dict[str, str | None] = {}

    async def fake_gatherer(req, customer_id, request=None):
        seen["key"] = current_tenant_virtual_key()
        return "sentinel"

    def fake_ctx(customer_id, *, http=None):
        return litellm_key_mod.optional_tenant_virtual_key_context(
            customer_id,
            http=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json={"litellm_key": "sk-tenant-x"})
                )
            ),
        )

    with (
        patch.object(pipeline_mod, "run_gatherer", fake_gatherer),
        patch.object(pipeline_mod, "optional_tenant_virtual_key_context", fake_ctx),
    ):
        fn = getattr(pipeline_mod, entry)
        args = [QueryRequest(query="hi"), "cust-1"]
        if entry == "run_search_phase":
            args.append(None)  # phase
        result = await fn(*args)

    assert result == "sentinel"
    assert seen["key"] == "sk-tenant-x", (
        f"{entry} ran the gatherer without the tenant key bound — "
        "every LLM call inside it bills to the shared key"
    )
    assert current_tenant_virtual_key() is None, "key leaked past the block"

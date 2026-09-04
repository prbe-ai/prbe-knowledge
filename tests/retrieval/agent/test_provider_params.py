"""Tripwire: the sampling parameters the gatherer DEPENDS ON must actually
reach the model.

Why this file exists. The gateway config (`research-os-litellm` ConfigMap)
sets `drop_params: true`, so LiteLLM silently removes any parameter its
capability map does not list for the routed provider. A dropped parameter is
indistinguishable from a working one at every level except an A/B on the
output bytes -- no error, no warning, no log line.

That cost us a whole shipped fix. PR #534 added a `frequency_penalty` retry
for runaway emissions, citing Cerebras's own docs
(inference-docs.cerebras.ai/models/openai-oss), which do say the knob is
honored. It is -- on Cerebras's API. Not through the gateway. Measured
2026-09-04 from inside research-os-engine-retrieval, temperature 0 + fixed
seed, penalty 2.0 (10x the configured value) against none:

    cerebras/gpt-oss-120b    1196 chars, sha 8e0807966363  -- byte-IDENTICAL
    fireworks gpt-oss-120b   1208 -> 1810 chars, hash differs -- honored

So the retry re-sent a byte-identical request and reproduced the same runaway,
burning a second full 16k-token generation every time it fired.

These assertions are offline -- `get_supported_openai_params` reads a local
table, no network, no key -- and they encode the MEASURED truth in both
directions, so a LiteLLM upgrade that changes either one fails here instead of
silently changing production behaviour.
"""

from __future__ import annotations

import litellm
import pytest

from engine.shared.constants import (
    SEARCH_AGENT_INFERENCE_MODEL,
    SEARCH_AGENT_LENGTH_RETRY_TEMPERATURE,
)

# The gateway rewrites `accounts/fireworks/*` to the upstream
# `fireworks_ai/accounts/fireworks/...` route, so the bare id does not resolve
# through get_llm_provider. Name the provider directly.
_FALLBACK_PROVIDER = "fireworks_ai"
_FALLBACK_MODEL = "accounts/fireworks/models/gpt-oss-120b"


def _supported(model: str, provider: str) -> set[str]:
    return set(litellm.get_supported_openai_params(
        model=model, custom_llm_provider=provider,
    ) or [])


def _primary() -> tuple[str, str]:
    model, provider = litellm.get_llm_provider(model=SEARCH_AGENT_INFERENCE_MODEL)[:2]
    return model, provider


def test_primary_provider_honors_temperature() -> None:
    """`temperature` is the lever the length-retry rides on. If the routed
    provider stops supporting it, the retry silently becomes a plain retry
    again -- the exact regression this file exists to catch."""
    model, provider = _primary()
    assert "temperature" in _supported(model, provider), (
        f"{provider} no longer supports `temperature`; the length-retry in "
        "loop.py cannot break a deterministic runaway without it."
    )


def test_primary_provider_still_drops_frequency_penalty() -> None:
    """Pins the measured reality, not the vendor's documentation.

    If this starts failing, LiteLLM has ADDED frequency_penalty for this
    provider -- good news. Re-run the A/B (penalty 2.0 vs none, temperature 0,
    fixed seed, compare output hashes) before trusting it, then decide whether
    the penalty should carry weight again."""
    model, provider = _primary()
    assert "frequency_penalty" not in _supported(model, provider), (
        f"{provider} now lists frequency_penalty. Verify it end-to-end with an "
        "output-hash A/B before relying on it -- the docs claimed it worked "
        "while the gateway was dropping it."
    )


def test_fallback_provider_honors_frequency_penalty() -> None:
    """Why the retry keeps sending the penalty at all: a failover lands the
    retry on Fireworks, which does honor it. Costs nothing, helps there."""
    assert "frequency_penalty" in _supported(_FALLBACK_MODEL, _FALLBACK_PROVIDER)


@pytest.mark.parametrize("param", ["temperature", "seed", "max_completion_tokens",
                                   "tools", "tool_choice", "response_format"])
def test_params_the_loop_sends_on_every_turn_are_supported(param: str) -> None:
    """Everything `_run_turn` puts on the wire unconditionally. Any of these
    being dropped changes behaviour with no error -- `tool_choice` going
    missing would let the model emit prose again, which the whole terminal-tool
    design exists to prevent."""
    model, provider = _primary()
    assert param in _supported(model, provider), f"{provider} drops `{param}`"


def test_retry_temperature_is_above_the_observed_runaway_band() -> None:
    """0.2 still ran away on one of the two replayed failures; 0.4 cleared it
    by a hair. Anything at or below 0.2 is known-insufficient, so guard the
    constant rather than trusting a future edit to remember."""
    assert SEARCH_AGENT_LENGTH_RETRY_TEMPERATURE > 0.2, (
        "temperature <= 0.2 reproduced the runaway live on 2026-09-04"
    )
